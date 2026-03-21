package com.childfocus.service

import android.accessibilityservice.AccessibilityService
import android.content.Intent
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
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
        // Emulator (Pixel AVD)     → "10.0.2.2"
        // Physical (same WiFi)     → your PC's local IP e.g. "192.168.1.x"
        private const val FLASK_HOST = "10.0.2.2"
        private const val FLASK_PORT = 5000
        private const val BASE_URL   = "http://$FLASK_HOST:$FLASK_PORT"

        private const val TITLE_RESET_MS = 5 * 60 * 1000L
        private const val DEBOUNCE_MS    = 1500L

        // ── Priority levels ───────────────────────────────────────────────
        // PLAYING = direct video ID from URL — most accurate, highest priority
        // ACTIVE  = title detected next to view count while video is playing
        private const val PRIORITY_PLAYING = 3
        private const val PRIORITY_ACTIVE  = 2

        // ── UI strings that are never real video titles ───────────────────
        private val SKIP_TITLES = listOf(
            // Player controls
            "Shorts", "Sponsored", "Advertisement", "Ad ·", "Skip Ads",
            "My Mix", "Trending", "Explore", "Subscriptions",
            "Library", "Home", "Video player", "Minimized player",
            "Minimize", "Cast", "More options", "Hide controls",
            "Enter fullscreen", "Rewind", "Fast forward", "Navigate up",
            "Voice search", "Choose Premium",
            // Action menus
            "More actions", "YouTube makes for you", "Drag to reorder",
            "Close Repeat", "Shuffle Menu", "Sign up", "re playlists",
            "Queue", "ylists", "Add to queue", "Save to playlist",
            "Share", "Report", "Not interested", "Don't recommend channel", "Next:",
            // Channel/account UI
            "Subscribe", "Subscribed", "Join", "Bell", "notifications",
            // Timestamp noise
            "minutes, ", "seconds", "Go to channel",
            // Feed noise
            "Music for you", "TikTok Lite", "Top podcasts", "Recommended",
            "Continue watching", "Up next", "Playing next",
            "Autoplay is", "Pause autoplay", "Mix -", "Topic",
            // Ad labels
            "Why this ad", "Stop seeing this ad", "Visit advertiser",
            "Ad ", "Promoted", "Sponsored content",
            // View count noise
            "K views", "M views", "B views",
            "months ago", "years ago", "days ago", "hours ago", "weeks ago",
            "See #", "videos ...more", "...more",
            // Comment UI
            "Add a comment", "@mention", "comment or @", "Reply", "replies",
            "Pinned comment", "View all comments", "Comments are turned off",
            "Top comments", "Newest first", "Sort comments",
            "likes", "like this", "liked by", "Liked by creator",
            "Show more replies", "Hide replies", "Load more comments",
            "Be the first to comment", "No comments yet",
            // Shorts UI
            "See more videos using this sound", "using this sound",
            "Original audio", "Original sound", "Collaboration channels",
            "View product", "Shop now", "Swipe up", "Add yours", "Remix this",
            // Music indicators
            "(Official", "- Topic", "♪", "♫", "🎵", "🎶",
            // YouTube Premium popups
            "Premium Lite", "Try Premium", "YouTube Premium",
            "you'll want to try", "Ad-free", "Get Premium",
            // Feature unavailable
            "Feature not available",
            "not available for this video",
        )

        private val SKIP_NODE_CLASSES = listOf(
            "android.widget.ImageButton", "android.widget.ImageView",
            "android.widget.ProgressBar", "android.widget.SeekBar",
            "android.widget.CheckBox", "android.widget.Switch",
            "android.widget.RadioButton",
        )

        private val CHANNEL_HANDLE_RE = Regex("^[A-Z][a-zA-Z0-9]{4,40}$")
    }

    private val scope = CoroutineScope(Dispatchers.IO)

    // ── Priority Queue ────────────────────────────────────────────────────────
    @Volatile private var currentJob: Job = Job().also { it.cancel() }
    @Volatile private var currentPriority = 0
    @Volatile private var currentTarget   = ""

    private var lastSentTitle   = ""
    private var lastSentTimeMs  = 0L
    private var pendingTitle    = ""
    private var lastEventTimeMs = 0L

    // Guard dedup — skip re-checking same screen state
    @Volatile private var lastGuardText   = ""
    @Volatile private var lastGuardResult = false

    // Shorts spam lock — set immediately on first detection, cleared after dispatch
    @Volatile private var shortsPendingTitle = ""

    private val http = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    // Pattern 1: video title is always shown next to view count
    private val VIEWS_PATTERN = Pattern.compile(
        "([A-Z][^\\n]{10,150})\\s+[\\d.,]+[KMBkm]?\\s+views",
        Pattern.CASE_INSENSITIVE
    )

    // Pattern 2: title before @ChannelName during playback
    private val AT_CHANNEL_PATTERN = Pattern.compile(
        "([A-Z][^\\n@]{10,150})\\s{1,4}@[\\w]{2,50}(?:\\s|$)"
    )

    // Pattern 3: direct video ID from URL — most reliable
    private val URL_PATTERN = Pattern.compile("(?:v=|youtu\\.be/|shorts/)([a-zA-Z0-9_-]{11})")

    // Pattern 4: Shorts title — appears before like/dislike/share buttons
    override fun onServiceConnected() {
        println("[CF_SERVICE] ✓ Connected — monitoring YouTube")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        event ?: return

        val now = System.currentTimeMillis()
        if (lastSentTitle.isNotEmpty() && (now - lastSentTimeMs) > TITLE_RESET_MS) {
            println("[CF_SERVICE] ↺ Reset title memory after timeout")
            lastSentTitle  = ""
            lastSentTimeMs = 0L
        }

        // ── Strategy 0: Direct video ID from event text ───────────────────
        val eventText = event.text?.joinToString(" ") ?: ""
        val urlMatch  = URL_PATTERN.matcher(eventText)
        if (urlMatch.find()) {
            val videoId = urlMatch.group(1) ?: return
            enqueue(videoId, PRIORITY_PLAYING) { doHandleVideoId(videoId) }
            return
        }

        val root    = rootInActiveWindow ?: return
        val allText = collectAllNodeText(root)
        root.recycle()

        // ── Guard: skip duplicate screen states ───────────────────────────
        if (allText == lastGuardText && lastGuardResult) return
        val isAdOrComment = isAdPlaying(allText) || isCommentSectionVisible(allText)
        lastGuardText   = allText
        lastGuardResult = isAdOrComment
        if (isAdOrComment) return

        // ── Strategy 1: Video ID in full node tree ────────────────────────
        val urlInTree = URL_PATTERN.matcher(allText)
        if (urlInTree.find()) {
            val videoId = urlInTree.group(1) ?: return
            enqueue(videoId, PRIORITY_PLAYING) { doHandleVideoId(videoId) }
            return
        }

        // ── Strategy 2: Shorts detection ─────────────────────────────────
        // Shorts fires typeWindowStateChanged (32) with text="YouTube" and
        // empty content description on every swipe. The title is ONLY in
        // the node tree — never in the event text.
        // We detect Shorts by checking if the tree contains Shorts-specific
        // UI elements ("Shorts" tab, like/comment buttons without view count)
        // and then extract the title from the tree directly.
        val isShorts = allText.contains("Shorts") &&
                !allText.contains("views", ignoreCase = true)
        if (isShorts) {
            val shortsTitle = extractShortsTitle(allText)
            // Block immediately if same title is already pending or sent
            if (shortsTitle != null
                && isCleanTitle(shortsTitle)
                && shortsTitle != lastSentTitle
                && shortsTitle != shortsPendingTitle) {
                shortsPendingTitle = shortsTitle  // lock immediately
                println("[CF_SERVICE] ✓ [SHORTS] $shortsTitle")
                enqueue(shortsTitle, PRIORITY_ACTIVE) { doDispatchTitle(shortsTitle) }
                return
            }
        }

        // ── Strategy 3: Title before view count (regular video playing) ───
        val viewsMatch = VIEWS_PATTERN.matcher(allText)
        if (viewsMatch.find()) {
            val t = viewsMatch.group(1)?.trim() ?: return
            if (isCleanTitle(t)) {
                enqueue(t, PRIORITY_ACTIVE) { doDispatchTitle(t) }
                return
            }
        }

        // ── Strategy 4: Title before @channel ────────────────────────────
        val atMatch = AT_CHANNEL_PATTERN.matcher(allText)
        if (atMatch.find()) {
            val t = atMatch.group(1)?.trim() ?: return
            if (isCleanTitle(t)) {
                enqueue(t, PRIORITY_ACTIVE) { doDispatchTitle(t) }
                return
            }
        }


    }

    /**
     * Extracts the Shorts video title from the accessibility tree text.
     *
     * From the debug log, we know the Shorts tree contains:
     * - "Shorts" tab name
     * - Action button labels: "like", "dislike", "comment", "share"
     * - The video title appears as a text node somewhere in the tree
     *
     * Strategy: split allText into lines, find the longest clean line
     * that is NOT a UI element label — that is the title.
     */
    private fun extractShortsTitle(allText: String): String? {
        val uiLabels = setOf(
            "shorts", "home", "explore", "subscriptions", "library",
            "like", "dislike", "comment", "share", "subscribe", "subscribed",
            "follow", "more", "pause", "play", "mute", "unmute",
            "youtube", "search", "remix", "add yours", "save",
            // Progress/player UI labels
            "video progress", "progress", "seek bar", "playback",
            // Watch prompt
            "watch full video", "watch more", "watch now",
            // Shopping/affiliate labels
            "affiliate", "shopee", "lazada", "tiktok shop",
            "best seller", "shop now", "buy now", "order now",
            "add to cart", "check link", "link in bio",
            // Comment/search UI
            "view comments", "comments", "search",
        )
        val lines = allText.split("\n")
            .map { it.trim() }
            .filter { it.length >= 8 }
            .filter { line ->
                val lower = line.lowercase()
                // Not a known UI label
                !uiLabels.any { lower == it } &&
                        // Not a pure number (like count)
                        !line.matches(Regex("[\\d.,]+[KMBkm]?")) &&
                        // Not a hashtag string
                        !line.startsWith("#") &&
                        // Not a timestamp
                        !line.matches(Regex("\\d+:\\d+.*")) &&
                        // Must have at least 3 words (filters channel names like "Apple Philippines")
                        line.trim().split(" ").filter { it.isNotEmpty() }.size >= 3
            }

        // Return the first line that passes isCleanTitle
        return lines.firstOrNull { isCleanTitle(it) }
    }

    private fun searchNodeForShortsId(node: AccessibilityNodeInfo, pattern: Regex): String? {
        // Check this node's text and content description — no class filtering
        val text = node.text?.toString() ?: ""
        val desc = node.contentDescription?.toString() ?: ""
        for (str in listOf(text, desc)) {
            val match = pattern.find(str)
            if (match != null) return match.groupValues[1]
        }
        // Recurse into children
        for (i in 0 until node.childCount) {
            val child = node.getChild(i) ?: continue
            val result = searchNodeForShortsId(child, pattern)
            child.recycle()
            if (result != null) return result
        }
        return null
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PRIORITY QUEUE
    // ═══════════════════════════════════════════════════════════════════════

    private fun enqueue(target: String, priority: Int, block: suspend () -> Unit) {
        synchronized(this) {
            when {
                target == currentTarget && currentJob.isActive -> Unit

                priority > currentPriority && currentJob.isActive -> {
                    println("[QUEUE] ⬆ P$priority cancels: ${currentTarget.take(35)}")
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
        println("[QUEUE] ▶ [P$priority]: ${target.take(40)}")
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

    private suspend fun doDispatchTitle(title: String) {
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
        println("[CF_SERVICE] ✓ [ACTIVE] $title")

        broadcastResult(title, "Analyzing", 0f, false)
        if (!currentJob.isActive) return

        classifyByTitle(title)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // GUARDS
    // ═══════════════════════════════════════════════════════════════════════

    private fun isAdPlaying(allText: String): Boolean =
        listOf("Skip Ads", "Skip ad", "Ad ·", "Why this ad",
            "Stop seeing this ad", "Visit advertiser", "Skip in")
            .any { allText.contains(it, ignoreCase = true) }

    private fun isCommentSectionVisible(allText: String): Boolean =
        listOf("Add a comment", "Top comments", "Newest first",
            "Sort comments", "Be the first to comment",
            "Pinned comment", "Show more replies", "Load more comments")
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
                node.text?.let               { sb.append(it).append("\n") }
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

    private fun isCleanTitle(text: String): Boolean {
        if (text.length < 8 || text.length > 200) return false
        if (SKIP_TITLES.any { text.contains(it, ignoreCase = true) }) return false
        // Block product/ad titles
        val lowerText = text.lowercase()
        // Affiliate and shopping signals
        if (lowerText.contains("affiliate") ||
            lowerText.contains("shopee") ||
            lowerText.contains("lazada") ||
            lowerText.contains("best seller") ||
            lowerText.contains("buy now") ||
            lowerText.contains("order now") ||
            lowerText.contains("link in bio") ||
            // Ad copy patterns
            lowerText.contains("say goodbye to") ||
            lowerText.contains("one pair for") ||
            lowerText.contains("years younger") ||
            lowerText.contains("skin look") ||
            lowerText.contains("suitable for all") ||
            lowerText.contains("all skin types") ||
            // YouTube UI strings
            lowerText.startsWith("search ") ||
            (lowerText.startsWith("view ") && lowerText.contains("comment")) ||
            lowerText.contains("palawan") ||
            lowerText.contains("sangla") ||
            // Generic ad openers
            lowerText.startsWith("helps ") ||
            lowerText.startsWith("get ") && text.length < 60 ||
            lowerText.startsWith("try ") && text.length < 50) {
            return false
        }
        // Product model numbers: letter(s) + 4+ digits + more alphanumerics
        // e.g. "ks-9000h", "A0058N010926A"
        if (Regex("[A-Za-z]{1,4}-?\\d{4,}[A-Za-z0-9]*").containsMatchIn(text)) {
            return false
        }
        // Dot-separated product/app names: "Banana · Tobii", "Word · App"
        if (Regex("^[A-Z][a-zA-Z]+ [·•] [A-Z][a-zA-Z]+$").containsMatchIn(text)) {
            return false
        }
        if (CHANNEL_HANDLE_RE.matches(text.trim())) return false
        if (text.trim().split(Regex("\\s+")).size < 2) return false
        return true
    }

    // ═══════════════════════════════════════════════════════════════════════
    // NETWORK
    // ═══════════════════════════════════════════════════════════════════════

    private fun classifyByTitle(title: String) {
        try {
            val body    = JSONObject().apply { put("title", title) }
            val request = Request.Builder()
                .url("$BASE_URL/classify_by_title")
                .post(body.toString().toRequestBody("application/json".toMediaType()))
                .build()
            val response = http.newCall(request).execute()
            val json     = JSONObject(response.body?.string() ?: return)
            handleClassificationResult(json)
        } catch (e: Exception) {
            println("[CF_SERVICE] ✗ classify_by_title: ${e.message}")
        }
    }

    private fun classifyByUrl(videoId: String, videoUrl: String, thumbUrl: String) {
        try {
            val body = JSONObject().apply {
                put("video_url",     videoUrl)
                put("thumbnail_url", thumbUrl)
            }
            val request = Request.Builder()
                .url("$BASE_URL/classify_full")
                .post(body.toString().toRequestBody("application/json".toMediaType()))
                .build()
            val response = http.newCall(request).execute()
            val json     = JSONObject(response.body?.string() ?: return)
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
        broadcastResult(videoId, label, score.toFloat(), cached)
    }

    private fun broadcastResult(videoId: String, label: String, score: Float, cached: Boolean) {
        val intent = Intent("com.childfocus.CLASSIFICATION_RESULT").apply {
            putExtra("video_id",    videoId)
            putExtra("oir_label",   label)
            putExtra("score_final", score)
            putExtra("cached",      cached)
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }

    override fun onInterrupt() {
        println("[CF_SERVICE] Interrupted")
        currentJob.cancel()
        lastSentTitle   = ""
        lastSentTimeMs  = 0L
        pendingTitle    = ""
        lastEventTimeMs = 0L
        currentPriority    = 0
        currentTarget      = ""
        shortsPendingTitle = ""
    }
}