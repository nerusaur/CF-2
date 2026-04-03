package com.childfocus.service

import android.accessibilityservice.AccessibilityService
import android.content.Intent
import android.os.Handler
import android.os.Looper
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import android.widget.Toast
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import java.util.regex.Pattern

class ChildFocusAccessibilityService : AccessibilityService() {

    companion object {
        // ── Change this ONE line to switch targets ────────────────────────
        // Emulator (Pixel AVD) → "10.0.2.2"
        // Physical (same WiFi) → your PC's local IP e.g. "192.168.1.x"
        private const val FLASK_HOST = "192.168.100.13"
        private const val FLASK_PORT = 5000
        private const val BASE_URL = "http://$FLASK_HOST:$FLASK_PORT"

        private const val TITLE_RESET_MS = 5 * 60 * 1000L
        private const val DEBOUNCE_MS = 1500L

        // ── Screen-time testing flag ──────────────────────────────────────
        // Keep DEBUG_SCREEN_TIME = false in production.
        // Set to true only when you want the ticker to fire every 10 s
        // instead of 60 s for quick local testing.
        const val DEBUG_SCREEN_TIME         = false
        const val SCREEN_TIME_TEST_LIMIT_MS = 1 * 60 * 1000L
        private val TICKER_INTERVAL_MS      = if (DEBUG_SCREEN_TIME) 10_000L else 60_000L

        private const val PRIORITY_PLAYING = 3
        private const val PRIORITY_ACTIVE  = 2

        /**
         * Packages that must NEVER be counted as foreground app time or
         * subject to screen-time enforcement. These are launchers, system UI,
         * input methods, and our own app — none of which the child is
         * "using" in the supervised sense.
         *
         * Without this list, two phantom-enforcement scenarios occur:
         *
         *  A) The screenTimeTicker fires while the launcher is in front
         *     (e.g. right after a prior enforcement sent the user home).
         *     tick() sees the previously-tracked app is still over-limit and
         *     returns true. getCurrentForegroundPkg() returns the launcher.
         *     enforceScreenTimeLimit(launcher) fires → random home + toast. ✗
         *
         *  B) A TYPE_WINDOW_STATE_CHANGED arrives for the system notification
         *     shade or an IME. onAppForeground() starts accruing time against
         *     that package. If it ever crosses a limit (or alreadyExceeded is
         *     set for it by accident) → phantom enforcement while idle. ✗
         */
        private val EXEMPT_PACKAGES = setOf(
            // Common AOSP / Pixel launchers
            "com.android.launcher",
            "com.android.launcher2",
            "com.android.launcher3",
            "com.google.android.apps.nexuslauncher",
            "com.sec.android.app.launcher",          // Samsung One UI
            "com.miui.home",                          // MIUI
            "com.huawei.android.launcher",            // EMUI
            "com.oppo.launcher",
            "com.vivo.launcher",
            "com.oneplus.launcher",
            // System UI / overlays
            "com.android.systemui",
            "android",
            "com.android.settings",
            // Input methods
            "com.google.android.inputmethod.latin",
            "com.samsung.android.honeyboard",
            "com.swiftkey.swiftkeyapp",
            // Our own service app — never block ourselves
            "com.childfocus",
        )

        private val SKIP_TITLES = listOf(
            "Shorts", "Sponsored", "Advertisement", "Ad ·", "Skip Ads",
            "My Mix", "Trending", "Explore", "Subscriptions", "Library",
            "Home", "Video player", "Minimized player", "Minimize", "Cast",
            "More options", "Hide controls", "Enter fullscreen", "Rewind",
            "Fast forward", "Navigate up", "Voice search", "Choose Premium",
            "More actions", "YouTube makes for you", "Drag to reorder",
            "Action menu", "Options menu", "Overflow menu",
            "Close Repeat", "Shuffle Menu", "Sign up", "re playlists",
            "Queue", "ylists", "Add to queue", "Save to playlist", "Share",
            "Report", "Not interested", "Don't recommend channel", "Next:",
            "Subscribe", "Subscribed", "Join", "Bell", "notifications",
            "minutes, ", "seconds", "Go to channel",
            "Music for you", "TikTok Lite", "Top podcasts", "Recommended",
            "Continue watching", "Up next", "Playing next", "Autoplay is",
            "Pause autoplay", "Mix -", "Topic",
            "Why this ad", "Stop seeing this ad", "Visit advertiser",
            "Ad ", "Promoted", "Sponsored content",
            "K views", "M views", "B views", "months ago", "years ago",
            "days ago", "hours ago", "weeks ago", "See #", "videos ...more",
            "...more",
            "Add a comment", "@mention", "comment or @", "Reply", "replies",
            "Pinned comment", "View all comments", "Comments are turned off",
            "Top comments", "Newest first", "Sort comments", "likes",
            "like this", "liked by", "Liked by creator", "Show more replies",
            "Hide replies", "Load more comments", "Be the first to comment",
            "No comments yet",
            "See more videos using this sound", "using this sound",
            "Original audio", "Original sound", "Collaboration channels",
            "View product", "Shop now", "Swipe up", "Add yours", "Remix this",
            "(Official", "- Topic", "♪", "♫", "🎵", "🎶",
            "Premium Lite", "Try Premium", "YouTube Premium",
            "you'll want to try", "Ad-free", "Get Premium",
            "Feature not available", "not available for this video",
            "New content available", "content is available",
            "Subscriptions:", "new content",
        )

        private val SKIP_TITLES_WHOLE_WORD = setOf(
            "Subscribe", "Subscribed", "Join", "Bell", "Share", "Reply",
            "Report", "Queue", "Topic", "Shorts", "Explore", "Library",
            "Home", "Cast", "Minimize", "likes", "seconds", "Recommended",
        )

        private val SKIP_TITLES_WHOLE_WORD_RE: Regex = run {
            val pattern = SKIP_TITLES_WHOLE_WORD
                .joinToString("|") { Regex.escape(it) }
            Regex("(?i)\\b($pattern)\\b")
        }

        private val SKIP_NODE_CLASSES = listOf(
            "android.widget.ImageButton",
            "android.widget.ImageView",
            "android.widget.ProgressBar",
            "android.widget.SeekBar",
            "android.widget.CheckBox",
            "android.widget.Switch",
            "android.widget.RadioButton",
        )

        private val CHANNEL_HANDLE_RE = Regex("^[A-Z][a-zA-Z0-9]{4,40}$")
    }

    private val scope = CoroutineScope(Dispatchers.IO)

    @Volatile private var currentJob: Job = Job().also { it.cancel() }
    @Volatile private var currentPriority = 0
    @Volatile private var currentTarget = ""

    private var lastSentTitle   = ""
    private var lastSentTimeMs  = 0L
    private var pendingTitle    = ""
    private var lastEventTimeMs = 0L

    @Volatile private var lastGuardText      = ""
    @Volatile private var lastGuardResult    = false
    @Volatile private var shortsPendingTitle = ""
    @Volatile private var lastExtractedShortsKey  = ""
    @Volatile private var lastShortsScreenHash   = 0

    // ── Screen-time enforcement state ─────────────────────────────────────
    // lastEnforcedTimeMs: timestamp of the most recent GLOBAL_ACTION_HOME
    //   triggered by screen-time. Used as a simple cooldown so rapid bursts
    //   of TYPE_WINDOW_STATE_CHANGED events (launcher transitions) don't spam
    //   the home action multiple times per enforcement.
    @Volatile private var lastEnforcedTimeMs       = 0L
    @Volatile private var lastEnforcedPackage      = ""
    private  val ENFORCE_COOLDOWN_MS               = 2_000L
    // Set to true as soon as the foreground changes AWAY from the enforced
    // package (i.e. the app actually went to background). This lets us
    // distinguish a genuine re-open from duplicate burst events so the
    // cooldown never blocks a legitimate second enforcement.
    @Volatile private var enforcedPkgWentBackground = false

    private val http = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    private val VIEWS_PATTERN = Pattern.compile(
        "([A-Z][^\\n]{10,150})\\s+[\\d.,]+[KMBkm]?\\s+views",
        Pattern.CASE_INSENSITIVE
    )
    private val AT_CHANNEL_PATTERN = Pattern.compile(
        "([A-Z][^\\n@]{10,150})\\s{1,4}@([\\w]{2,50})(?:\\s|$)"
    )
    private val URL_PATTERN = Pattern.compile(
        "(?:v=|youtu\\.be/|shorts/)([a-zA-Z0-9_-]{11})"
    )

    // ── Package exemption helper ─────────────────────────────────────────────
    // Returns true for launchers, system UI, IMEs, and our own package —
    // anything that must never be treated as a child-supervised foreground app.
    private fun isExemptPackage(pkg: String): Boolean {
        if (pkg.isEmpty()) return true
        return EXEMPT_PACKAGES.any { pkg == it || pkg.startsWith("$it.") }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // DATA CLASS
    // ═══════════════════════════════════════════════════════════════════════

    private data class ShortsInfo(val title: String, val channel: String)

    // ═══════════════════════════════════════════════════════════════════════
    // LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════

    override fun onServiceConnected() {
        println("[CF_SERVICE] ✓ Connected — monitoring YouTube + screen time")
        mainHandler.postDelayed(screenTimeTicker, TICKER_INTERVAL_MS)
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        event ?: return

        // ── Screen-time enforcement ──────────────────────────────────────
        if (event.eventType == AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED) {
            val pkg = event.packageName?.toString() ?: ""
            if (pkg.isNotEmpty() && !isExemptPackage(pkg)) {

                // If the foreground just moved AWAY from the previously enforced
                // package, record it. This tells us the app actually went to
                // background so the next open is a genuine re-launch, not a
                // duplicate event from the same enforcement burst.
                if (lastEnforcedPackage.isNotEmpty() && pkg != lastEnforcedPackage) {
                    enforcedPkgWentBackground = true
                }

                // Always update foreground tracking so usage accumulates.
                val overLimit = ScreenTimeManager.onAppForeground(this, pkg)

                // isExceeded() reads from SharedPrefs — persisted across bursts.
                // overLimit reflects a first-time breach detected this tick.
                val alreadyExceeded = ScreenTimeManager.isExceeded(this, pkg)

                // ── FIX #2: enforce ONLY for packages the user has configured.
                //    Without this guard, any non-exempt package could be blocked
                //    if isExceeded() or overLimit ever returned true for it —
                //    e.g. due to a stale SharedPrefs entry or an event arriving
                //    while foregroundPkg was pointing at a tracked package.
                val shouldBlock = (alreadyExceeded || overLimit) &&
                        pkg in ScreenTimeManager.TRACKED_PACKAGES

                if (shouldBlock) {
                    val now = System.currentTimeMillis()

                    val cooldownExpired      = (now - lastEnforcedTimeMs) > ENFORCE_COOLDOWN_MS
                    val differentPkg         = lastEnforcedPackage != pkg
                    // Only bypass the cooldown when the app genuinely went to
                    // background after the last enforcement and is now coming
                    // back — i.e. the child deliberately re-opened it.
                    // Do NOT bypass just because alreadyExceeded is true: that
                    // flag stays set all day and would cause every stray
                    // TYPE_WINDOW_STATE_CHANGED for this package (including
                    // system-generated ones while the phone is idle) to fire
                    // a home action and toast, producing the "random
                    // notification while doing nothing" symptom.
                    val isReopenAfterEnforce = enforcedPkgWentBackground && lastEnforcedPackage == pkg

                    if (cooldownExpired || differentPkg || isReopenAfterEnforce) {
                        enforcedPkgWentBackground = false   // reset for next burst
                        enforceScreenTimeLimit(pkg)
                    }
                    return
                }
            }
        }

        val now = System.currentTimeMillis()
        if (lastSentTitle.isNotEmpty() && (now - lastSentTimeMs) > TITLE_RESET_MS) {
            lastSentTitle          = ""
            lastSentTimeMs         = 0L
            lastExtractedShortsKey  = ""
            lastShortsScreenHash   = 0
        }

        val root    = rootInActiveWindow ?: return
        val allText = collectAllNodeText(root)
        root.recycle()

        if (allText == lastGuardText && lastGuardResult) return
        val isAdOrComment = isAdPlaying(allText) || isCommentSectionVisible(allText)
        lastGuardText   = allText
        lastGuardResult = isAdOrComment
        if (isAdOrComment) return

        val isShorts = allText.contains("Shorts") &&
                !allText.contains("views", ignoreCase = true) &&
                isRealShortsScreen(allText)

        if (isShorts) {
            val screenHash = allText.hashCode()
            if (screenHash != lastShortsScreenHash) {
                lastShortsScreenHash = screenHash
                val info = extractShortsInfo(allText)
                if (info != null &&
                    info.title != lastSentTitle &&
                    info.title != shortsPendingTitle
                ) {
                    shortsPendingTitle = info.title
                    val extractedId = tryExtractShortsVideoId(allText)
                    if (extractedId != null) {
                        println("[CF_SERVICE] ✓ [SHORTS] ID extracted from tree: $extractedId")
                        enqueue(extractedId, PRIORITY_PLAYING) { doHandleVideoId(extractedId) }
                    } else {
                        println("[CF_SERVICE] ✓ [SHORTS] title='${info.title}' channel='${info.channel}' (no ID in tree — using title search)")
                        enqueue(info.title, PRIORITY_ACTIVE) { doDispatchTitle(info.title, info.channel) }
                    }
                }
            }
            return
        }

        val eventText = event.text?.joinToString(" ") ?: ""
        val urlMatch  = URL_PATTERN.matcher(eventText)
        if (urlMatch.find()) {
            val videoId = urlMatch.group(1) ?: return
            enqueue(videoId, PRIORITY_PLAYING) { doHandleVideoId(videoId) }
            return
        }

        val urlInTree = URL_PATTERN.matcher(allText)
        if (urlInTree.find()) {
            val videoId = urlInTree.group(1) ?: return
            enqueue(videoId, PRIORITY_PLAYING) { doHandleVideoId(videoId) }
            return
        }

        val viewsMatch = VIEWS_PATTERN.matcher(allText)
        if (viewsMatch.find()) {
            val t = viewsMatch.group(1)?.trim() ?: return
            if (isCleanTitle(t)) {
                enqueue(t, PRIORITY_ACTIVE) { doDispatchTitle(t) }
                return
            }
        }

        val atMatch = AT_CHANNEL_PATTERN.matcher(allText)
        if (atMatch.find()) {
            val t  = atMatch.group(1)?.trim() ?: return
            val ch = atMatch.group(2)?.trim() ?: ""
            if (isCleanTitle(t)) {
                enqueue(t, PRIORITY_ACTIVE) { doDispatchTitle(t, ch) }
                return
            }
        }
    }

    override fun onInterrupt() {
        currentJob.cancel()
        mainHandler.removeCallbacks(screenTimeTicker)
        ScreenTimeManager.onAppBackground(this)
        lastSentTitle            = ""
        lastSentTimeMs           = 0L
        pendingTitle             = ""
        lastEventTimeMs          = 0L
        currentPriority          = 0
        currentTarget            = ""
        shortsPendingTitle       = ""
        lastExtractedShortsKey   = ""
        lastShortsScreenHash     = 0
        lastEnforcedPackage      = ""
        lastEnforcedTimeMs       = 0L
        enforcedPkgWentBackground = false
        // Re-schedule the ticker so it survives brief service interruptions.
        mainHandler.postDelayed(screenTimeTicker, TICKER_INTERVAL_MS)
    }

    override fun onDestroy() {
        super.onDestroy()
        mainHandler.removeCallbacks(screenTimeTicker)
        ScreenTimeManager.onAppBackground(this)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SHORTS VIDEO ID EXTRACTION
    // ═══════════════════════════════════════════════════════════════════════

    private fun tryExtractShortsVideoId(allText: String): String? {
        val root = rootInActiveWindow ?: return null
        val nodes = collectTextNodes(root)
        root.recycle()

        for (node in nodes) {
            val m = URL_PATTERN.matcher(node.text)
            if (m.find()) {
                val id = m.group(1) ?: continue
                if (id.length == 11 && id.matches(Regex("[a-zA-Z0-9_-]{11}"))) {
                    return id
                }
            }
        }

        val m = Pattern.compile("shorts/([a-zA-Z0-9_-]{11})").matcher(allText)
        if (m.find()) {
            val id = m.group(1) ?: return null
            if (id.length == 11) return id
        }

        return null
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SHORTS TITLE + CHANNEL EXTRACTION
    // ═══════════════════════════════════════════════════════════════════════

    private data class TextNode(val text: String, val top: Int, val bottom: Int)

    private fun collectTextNodes(node: AccessibilityNodeInfo): List<TextNode> {
        val result = mutableListOf<TextNode>()
        try {
            val rect = android.graphics.Rect()
            node.getBoundsInScreen(rect)

            val texts = mutableListOf<String>()
            node.text?.toString()?.trim()?.takeIf { it.isNotEmpty() }?.let { texts += it }
            node.contentDescription?.toString()?.trim()?.takeIf { it.isNotEmpty() }?.let { texts += it }

            for (t in texts) {
                if (t.matches(Regex("[\\d.,:\\s]+[KMBkm]?"))) continue
                result += TextNode(text = t, top = rect.top, bottom = rect.bottom)
            }

            for (i in 0 until node.childCount) {
                val child = node.getChild(i) ?: continue
                result += collectTextNodes(child)
                child.recycle()
            }
        } catch (_: Exception) { }
        return result
    }

    private fun extractShortsInfo(allText: String): ShortsInfo? {
        val root = rootInActiveWindow ?: return null
        val nodes = collectTextNodes(root)
        root.recycle()

        if (nodes.isEmpty()) return null

        val screenHeight = resources.displayMetrics.heightPixels

        val SETTLED_BOTTOM_MIN   = (screenHeight * 0.72).toInt()
        val SETTLED_BOTTOM_MAX   = (screenHeight * 0.97).toInt()
        val CHANNEL_ROW_TARGET   = (screenHeight * 0.87).toInt()
        val NEIGHBOUR_TOLERANCE  = (screenHeight * 0.18).toInt()

        val uiExact = setOf(
            "shorts", "home", "explore", "subscriptions", "library", "you",
            "like", "dislike", "comment", "share", "subscribe", "subscribed",
            "follow", "more", "pause", "play", "mute", "unmute", "youtube",
            "search", "remix", "add yours", "save", "reels", "create",
            "video progress", "progress", "seek bar", "playback",
        )
        val uiContains = listOf(
            "skip ad", "why this ad", "visit advertiser",
            "new content available", "subscriptions:",
            "see more videos", "feature not available",
            "like this video", "dislike this video", "share this video",
            "comments disabled", "subscribe to @",
            "please wait", "action menu", "go to channel",
        )

        val filtered = nodes.filter { n ->
            val lower = n.text.lowercase()
            !uiExact.any { lower == it } &&
                    !uiContains.any { lower.contains(it) } &&
                    n.text.length >= 2
        }

        fun isMusicLabel(text: String): Boolean {
            val lower = text.lowercase()
            return lower.contains("♪") || lower.contains("♫") ||
                    lower.contains("🎵") || lower.contains("🎶") ||
                    lower.contains("original audio") || lower.contains("original sound")
        }

        val seekBarRe = Regex(
            "^\\d+\\s+minutes?\\s+\\d+\\s+seconds?\\s+of\\s+\\d+\\s+minutes?\\s+\\d+\\s+seconds?$",
            RegexOption.IGNORE_CASE
        )

        val channelNode = filtered
            .filter { it.text.startsWith("@") && it.bottom > 0 }
            .minByOrNull { Math.abs(it.bottom - CHANNEL_ROW_TARGET) }
            ?: return null

        val maxAllowedBottom = minOf(channelNode.bottom + NEIGHBOUR_TOLERANCE, SETTLED_BOTTOM_MAX)

        val candidates = filtered
            .filter { it.text != channelNode.text && !it.text.startsWith("@") }
            .filterNot { it.text.trimStart().startsWith("#") }
            .filterNot { it.text.lowercase().startsWith("search ") }
            .filterNot { isMusicLabel(it.text) }
            .filterNot { it.text.contains(" · ") }
            .filterNot { seekBarRe.matches(it.text.trim()) }
            .filter { it.bottom in SETTLED_BOTTOM_MIN..maxAllowedBottom }
            .filter { it.top < it.bottom }
            .filter { it.text.length >= 3 }
            .sortedByDescending { it.bottom }

        if (candidates.isEmpty()) return null

        val titleRaw = candidates[0].text

        var titleClean = titleRaw.replace(Regex("[​‌‍﻿]"), "").trim()

        val hashIdx = titleClean.indexOf('#')
        if (hashIdx > 0) titleClean = titleClean.substring(0, hashIdx).trim()

        titleClean = titleClean
            .replace(Regex("\\s+[A-Za-z][A-Za-z0-9_-]{3,39}$"), "")
            .trim()

        if (titleClean.length < 3) return null

        val key = "${channelNode.text}|$titleClean"
        if (key == lastExtractedShortsKey) return null
        lastExtractedShortsKey = key

        println("[CF_SERVICE] ✓ [SHORTS] screenH=$screenHeight settled=$SETTLED_BOTTOM_MIN..$SETTLED_BOTTOM_MAX channelRow=$CHANNEL_ROW_TARGET")
        return ShortsInfo(title = titleClean, channel = channelNode.text)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // GUARDS
    // ═══════════════════════════════════════════════════════════════════════

    private fun isRealShortsScreen(allText: String): Boolean {
        val root = rootInActiveWindow ?: return false
        val nodes = collectTextNodes(root)
        root.recycle()
        val combined = nodes.joinToString(" ") { it.text }.lowercase()
        val shortsUiSignals = listOf(
            "like", "dislike", "comment", "share", "remix",
            "add yours", "mute", "pause", "play", "seek bar", "video progress"
        )
        val matchCount = shortsUiSignals.count { combined.contains(it) }
        return matchCount >= 2
    }

    private fun isAdPlaying(allText: String): Boolean =
        listOf("Skip Ads", "Skip ad", "Ad ·", "Why this ad",
            "Stop seeing this ad", "Visit advertiser", "Skip in")
            .any { allText.contains(it, ignoreCase = true) }

    private fun isCommentSectionVisible(allText: String): Boolean =
        listOf("Add a comment", "Top comments", "Newest first", "Sort comments",
            "Be the first to comment", "Pinned comment",
            "Show more replies", "Load more comments")
            .any { allText.contains(it, ignoreCase = true) }

    // ═══════════════════════════════════════════════════════════════════════
    // HELPERS
    // ═══════════════════════════════════════════════════════════════════════

    private fun collectAllNodeText(node: AccessibilityNodeInfo): String {
        val sb = StringBuilder()
        try {
            val className    = node.className?.toString() ?: ""
            val isSkipped    = SKIP_NODE_CLASSES.any { className.endsWith(it.substringAfterLast('.')) }
            val textLen      = node.text?.length ?: 0
            val isButtonLike = node.isClickable && textLen in 1..25 && node.childCount == 0
            if (!isSkipped && !isButtonLike) {
                node.text?.let { sb.append(it).append("\n") }
                node.contentDescription?.let { sb.append(it).append("\n") }
            }
            for (i in 0 until node.childCount) {
                val child = node.getChild(i) ?: continue
                sb.append(collectAllNodeText(child))
                child.recycle()
            }
        } catch (_: Exception) { }
        return sb.toString()
    }

    private fun skipTitleMatches(text: String, skipTerm: String): Boolean {
        return if (skipTerm in SKIP_TITLES_WHOLE_WORD) {
            SKIP_TITLES_WHOLE_WORD_RE.containsMatchIn(text)
                    && Regex("(?i)\\b${Regex.escape(skipTerm)}\\b").containsMatchIn(text)
        } else {
            text.contains(skipTerm, ignoreCase = true)
        }
    }

    private fun isCleanTitle(text: String): Boolean {
        if (text.length < 8 || text.length > 200) return false
        if (SKIP_TITLES.any { skipTitleMatches(text, it) }) return false
        val lowerText = text.lowercase()
        if (lowerText.contains(" thousand views") || lowerText.contains("- play short") ||
            lowerText.contains("affiliate")       || lowerText.contains("shopee") ||
            lowerText.contains("lazada")           || lowerText.contains("best seller") ||
            lowerText.contains("buy now")          || lowerText.contains("order now") ||
            lowerText.contains("link in bio")      ||
            lowerText.contains("say goodbye to")   || lowerText.contains("one pair for") ||
            lowerText.contains("years younger")    || lowerText.contains("skin look") ||
            lowerText.contains("suitable for all") || lowerText.contains("all skin types") ||
            lowerText.startsWith("search ") ||
            (lowerText.startsWith("view ") && lowerText.contains("comment")) ||
            lowerText.contains("palawan")          || lowerText.contains("sangla") ||
            lowerText.startsWith("helps ")         ||
            lowerText.startsWith("get ") && text.length < 60 ||
            lowerText.startsWith("try ") && text.length < 50
        ) return false
        if (Regex("[A-Za-z]{1,4}-?\\d{4,}[A-Za-z0-9]*").containsMatchIn(text)) return false
        if (Regex("^[A-Z][a-zA-Z]+ [·•] [A-Z][a-zA-Z]+$").containsMatchIn(text)) return false
        if (CHANNEL_HANDLE_RE.matches(text.trim())) return false
        if (text.trim().split(Regex("\\s+")).size < 2) return false
        return true
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PAUSE CONTROL
    // ═══════════════════════════════════════════════════════════════════════

    private val mainHandler = Handler(Looper.getMainLooper())

    private val screenTimeTicker = object : Runnable {
        override fun run() {
            val overLimit = ScreenTimeManager.tick(this@ChildFocusAccessibilityService)
            if (overLimit) {
                val pkg = ScreenTimeManager.getCurrentForegroundPkg()
                if (pkg.isNotEmpty() && !isExemptPackage(pkg)) {

                    // ── FIX #1: verify the package ScreenTimeManager thinks is
                    //    in the foreground actually IS the foreground right now.
                    //
                    //    Problem: foregroundPkg is only updated when a
                    //    TYPE_WINDOW_STATE_CHANGED event arrives. Some apps do
                    //    not emit this event reliably, so foregroundPkg can stay
                    //    pointing at the last tracked app (e.g. YouTube) even
                    //    after the user has opened Gmail or the camera.
                    //
                    //    When the ticker then fires and sees YouTube is exceeded,
                    //    enforceScreenTimeLimit("youtube") calls
                    //    performGlobalAction(HOME) — which kicks whatever app is
                    //    ACTUALLY in the foreground, not YouTube.
                    //
                    //    Fix: read the real foreground package from the window
                    //    before enforcing. If it doesn't match, update
                    //    ScreenTimeManager and skip this enforcement cycle.
                    val actualPkg = rootInActiveWindow?.packageName?.toString() ?: ""
                    when {
                        // Real foreground matches tracked pkg — safe to enforce.
                        actualPkg.isEmpty() || actualPkg == pkg -> {
                            enforceScreenTimeLimit(pkg)
                        }
                        // Real foreground is a different app — foregroundPkg is
                        // stale. Update tracking so future ticks are accurate,
                        // but do NOT kick the user out of the unrelated app.
                        else -> {
                            println("[SCREEN TIME] ⚠ Ticker stale: tracked=$pkg actual=$actualPkg — skipping enforcement, syncing state")
                            ScreenTimeManager.onAppForeground(
                                this@ChildFocusAccessibilityService, actualPkg
                            )
                        }
                    }
                }
            }
            mainHandler.postDelayed(this, TICKER_INTERVAL_MS)
        }
    }

    /**
     * Sends the user to the home screen when a package exceeds its daily limit.
     *
     * ── FIX (original) ───────────────────────────────────────────────────────
     * Calls ScreenTimeManager.markExceeded() BEFORE performing the home action.
     * This persists the exceeded state to SharedPreferences so that even after
     * the in-memory screenTimeLimitEnforced flag is reset (which happens as soon
     * as the user navigates to the launcher), the next onAccessibilityEvent for
     * this package will hit the isExceeded() check at the top and block the
     * re-open immediately.
     *
     * ── FIX (reopen bypass) ──────────────────────────────────────────────────
     * enforcedPkgWentBackground is set to true as soon as a different package
     * comes to the foreground after this enforcement fires. When the enforced
     * package then re-appears, isReopenAfterEnforce = true overrides the
     * cooldown gate so the home action fires again regardless of timing.
     *
     * Without markExceeded():
     *   1. Limit hit → enforce → go home → flag resets
     *   2. Reopen YouTube → onAppForeground → isOverLimit()
     *      → if usedMs is 1 ms under limit due to flush lag → returns false
     *      → YouTube proceeds ✗
     *
     * With markExceeded():
     *   1. Limit hit → enforce → markExceeded("com.google.android.youtube")
     *      → go home → in-memory flag resets
     *   2. Reopen YouTube → isExceeded() → true → block fires immediately ✓
     * ─────────────────────────────────────────────────────────────────────────
     */
    private fun enforceScreenTimeLimit(packageName: String) {
        if (packageName.isEmpty()) return

        // ── Atomic dedup ────────────────────────────────────────────────────
        // The ticker and onAccessibilityEvent run on different paths and can
        // both decide to enforce in the same millisecond (e.g. the ticker fires
        // exactly as the child opens the app). Without this gate, both calls
        // go through, producing a double toast and a double home action.
        // We use the same lastEnforcedTimeMs / lastEnforcedPackage pair that
        // the event path already maintains, but guard it with synchronized so
        // concurrent calls from IO (ticker coroutine) and main thread are safe.
        val now = System.currentTimeMillis()
        synchronized(this) {
            val alreadyFiredRecently =
                lastEnforcedPackage == packageName &&
                        (now - lastEnforcedTimeMs) < ENFORCE_COOLDOWN_MS
            if (alreadyFiredRecently) return
            lastEnforcedPackage = packageName
            lastEnforcedTimeMs  = now
        }

        // Persist exceeded state to SharedPrefs so every future re-open
        // of this app is blocked for the rest of the day, regardless of
        // whether the accessibility service was restarted in between.
        ScreenTimeManager.markExceeded(this, packageName)

        mainHandler.post {
            performGlobalAction(GLOBAL_ACTION_HOME)
            Toast.makeText(
                this,
                "⏱️ Daily screen time limit reached for this app.",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    /**
     * Pauses YouTube then navigates home to fully remove the blocked video.
     */
    private fun blockYouTubeVideo() {
        mainHandler.post {
            val root = rootInActiveWindow
            if (root != null) {
                root.findAccessibilityNodeInfosByViewId(
                    "com.google.android.youtube:id/player_control_play_pause_replay_button"
                ).firstOrNull()?.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                root.recycle()
            }
            mainHandler.postDelayed({
                performGlobalAction(GLOBAL_ACTION_HOME)
            }, 300)
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PRIORITY QUEUE
    // ═══════════════════════════════════════════════════════════════════════

    private fun enqueue(target: String, priority: Int, block: suspend () -> Unit) {
        synchronized(this) {
            when {
                target == currentTarget && currentJob.isActive -> Unit
                priority > currentPriority && currentJob.isActive -> {
                    currentJob.cancel()
                    startJob(target, priority, block)
                }
                !currentJob.isActive -> startJob(target, priority, block)
                else -> Unit
            }
        }
    }

    private fun startJob(target: String, priority: Int, block: suspend () -> Unit) {
        currentTarget   = target
        currentPriority = priority
        currentJob = scope.launch {
            try { block() }
            finally {
                synchronized(this@ChildFocusAccessibilityService) {
                    if (currentTarget == target) {
                        currentPriority = 0
                        currentTarget   = ""
                    }
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // WORKERS
    // ═══════════════════════════════════════════════════════════════════════

    private suspend fun doHandleVideoId(videoId: String) {
        if (videoId == lastSentTitle) return
        lastSentTitle  = videoId
        lastSentTimeMs = System.currentTimeMillis()
        println("[CF_SERVICE] ✓ [PLAYING] $videoId")
        broadcastResult(videoId, "Analyzing", 0f, false)
        if (!currentJob.isActive) return
        classifyByUrl(
            videoId,
            "https://www.youtube.com/watch?v=$videoId",
            "https://i.ytimg.com/vi/$videoId/hqdefault.jpg"
        )
    }

    private suspend fun doDispatchTitle(title: String, channel: String = "") {
        if (title.length < 8 || title == lastSentTitle) return
        pendingTitle    = title
        lastEventTimeMs = System.currentTimeMillis()
        delay(DEBOUNCE_MS)
        if (!currentJob.isActive) return
        if (pendingTitle != title) return
        if ((System.currentTimeMillis() - lastEventTimeMs) < DEBOUNCE_MS) return
        lastSentTitle      = title
        shortsPendingTitle = ""
        lastSentTimeMs     = System.currentTimeMillis()
        println("[CF_SERVICE] ✓ [ACTIVE] title='$title' channel='$channel'")
        broadcastResult(title, "Analyzing", 0f, false)
        if (!currentJob.isActive) return
        classifyByTitle(title, channel)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // NETWORK
    // ═══════════════════════════════════════════════════════════════════════

    private fun classifyByTitle(title: String, channel: String = "") {
        try {
            val body = JSONObject().apply {
                put("title", title)
                if (channel.isNotEmpty()) put("channel", channel)
            }
            val request = Request.Builder()
                .url("$BASE_URL/classify_by_title")
                .post(body.toString().toRequestBody("application/json".toMediaType()))
                .build()
            val response = http.newCall(request).execute()
            val json = JSONObject(response.body?.string() ?: return)
            handleClassificationResult(json)
        } catch (e: Exception) {
            println("[CF_SERVICE] ✗ classify_by_title: ${e.message}")
        }
    }

    private fun classifyByUrl(videoId: String, videoUrl: String, thumbUrl: String) {
        try {
            val body = JSONObject().apply {
                put("video_url", videoUrl)
                put("thumbnail_url", thumbUrl)
            }
            val request = Request.Builder()
                .url("$BASE_URL/classify_full")
                .post(body.toString().toRequestBody("application/json".toMediaType()))
                .build()
            val response = http.newCall(request).execute()
            val json = JSONObject(response.body?.string() ?: return)
            handleClassificationResult(json)
        } catch (e: Exception) {
            println("[CF_SERVICE] ✗ classify_full: ${e.message}")
        }
    }

    private fun handleClassificationResult(json: JSONObject) {
        val label   = json.optString("oir_label", "Neutral")
        val score   = json.optDouble("score_final", 0.5)
        val cached  = json.optBoolean("cached", false)
        val videoId = json.optString("video_id", "unknown")
        println("[CF_SERVICE] $videoId → $label ($score) cached=$cached")

        val isBlocked = label.equals("Overstimulating", ignoreCase = true) ||
                label.equals("Overstimulation",  ignoreCase = true)
        if (isBlocked) {
            blockYouTubeVideo()
        }

        broadcastResult(videoId, label, score.toFloat(), cached)
    }

    private fun broadcastResult(videoId: String, label: String, score: Float, cached: Boolean) {
        val intent = Intent("com.childfocus.CLASSIFICATION_RESULT").apply {
            putExtra("video_id", videoId)
            putExtra("oir_label", label)
            putExtra("score_final", score)
            putExtra("cached", cached)
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }
}