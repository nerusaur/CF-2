package com.childfocus.service

import android.accessibilityservice.AccessibilityService
import android.content.Intent
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
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
        // ── Change this ONE line to switch targets ────────────────────────────
        // Emulator (Pixel 3a AVD)  → "10.0.2.2"
        // Physical (Infinix WiFi)  → "192.168.100.136"
        private const val FLASK_HOST = "10.0.2.2"
        private const val FLASK_PORT = 5000
        private const val BASE_URL   = "http://$FLASK_HOST:$FLASK_PORT"

        // After 5 minutes without a new title, reset so the same
        // video re-classifies (hits cache = instant response)
        private const val TITLE_RESET_MS = 5 * 60 * 1000L

        // ── YouTube UI chrome — never a real video title ──────────────────────
        private val SKIP_TITLES = listOf(
            // Player controls
            "Shorts", "Sponsored", "Advertisement", "Ad ·", "Skip Ads",
            "My Mix", "Trending", "Explore", "Subscriptions",
            "Library", "Home", "Video player", "Minimized player",
            "Minimize", "Cast", "More options", "Hide controls",
            "Enter fullscreen", "Rewind", "Fast forward", "Navigate up",
            "Voice search", "Choose Premium",
            // ── Seen in Logcat 2026-03-15 ────────────────────────────────────
            "More actions",            // ← was missing; kebab-menu button label
            "YouTube makes for you",
            "Drag to reorder",
            "Close Repeat",
            "Shuffle Menu",
            "Sign up",
            "re playlists",
            "Queue",
            "ylists",
            "Add to queue",
            "Save to playlist",
            "Share",
            "Report",
            "Not interested",
            "Don't recommend channel",
            "Next:",
            // Channel/account UI
            "Subscribe",
            "Subscribed",
            "Join",
            "Bell",
            "notifications",
        )

        // ── System/notification noise ──────────────────────────────────────────
        private val SKIP_SYSTEM = listOf(
            "PTE. LTD", "Installed", "Open app", "App image",
            "Update", "Install", "Download", "Notification",
            "Allow", "Deny", "Permission", "Settings",
            "Battery", "Charging", "Wi-Fi", "Bluetooth",
        )

        // ── Widget classes that never contain video titles ─────────────────────
        private val SKIP_NODE_CLASSES = listOf(
            "android.widget.ImageButton",
            "android.widget.ImageView",
            "android.widget.ProgressBar",
            "android.widget.SeekBar",
            "android.widget.CheckBox",
            "android.widget.Switch",
            "android.widget.RadioButton",
        )

        // ── Looks like a channel handle: CamelCase, no spaces, no punctuation ──
        // Matches: "RickAstleyYT", "MichaelJackson", "VEVOofficial"
        // Does NOT match: "Michael Jackson - Billie Jean (Official Video)"
        private val CHANNEL_HANDLE_RE = Regex("^[A-Z][a-zA-Z0-9]{4,40}$")
    }

    private val scope = CoroutineScope(Dispatchers.IO)
    private var lastSentTitle  = ""
    private var lastSentTimeMs = 0L

    private val http = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    // ── Pattern 1: title before "views" count ─────────────────────────────────
    // e.g. "Michael Jackson - Billie Jean (Official Video)\n1.2B views"
    private val VIEWS_PATTERN = Pattern.compile(
        "([A-Z][^\\n]{10,150})\\s+[\\d.,]+[KMBkm]?\\s+views",
        Pattern.CASE_INSENSITIVE
    )

    // ── Pattern 2: title before @ChannelName ──────────────────────────────────
    // e.g. "Billie Jean (Official Video)\n@MichaelJackson"
    private val AT_CHANNEL_PATTERN = Pattern.compile(
        "([A-Z][^\\n@]{10,150})\\s{1,4}@[\\w]{2,50}(?:\\s|$)"
    )

    // ── Pattern 3: repeated title (minimized player) handled by extractRepeatedTitle()

    // ── Pattern 4: direct URL video ID ───────────────────────────────────────
    private val URL_PATTERN = Pattern.compile("(?:v=|youtu\\.be/)([a-zA-Z0-9_-]{11})")

    override fun onServiceConnected() {
        println("[CF_SERVICE] ✓ Connected — monitoring YouTube")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        event ?: return

        // Auto-reset after 5 min so revisiting same video hits cache
        val now = System.currentTimeMillis()
        if (lastSentTitle.isNotEmpty() && (now - lastSentTimeMs) > TITLE_RESET_MS) {
            println("[CF_SERVICE] ↺ Reset title memory after timeout")
            lastSentTitle  = ""
            lastSentTimeMs = 0L
        }

        // Strategy 0: direct URL in event text (fastest path)
        val eventText = event.text?.joinToString(" ") ?: ""
        val urlMatch  = URL_PATTERN.matcher(eventText)
        if (urlMatch.find()) { handleVideoId(urlMatch.group(1) ?: return); return }

        val root    = rootInActiveWindow ?: return
        val allText = collectAllNodeText(root)
        root.recycle()

        // Strategy 1: direct URL in tree
        val urlInTree = URL_PATTERN.matcher(allText)
        if (urlInTree.find()) { handleVideoId(urlInTree.group(1) ?: return); return }

        // Strategy 2: title before "views" — most accurate
        val viewsMatch = VIEWS_PATTERN.matcher(allText)
        if (viewsMatch.find()) {
            val t = viewsMatch.group(1)?.trim() ?: return
            if (isCleanTitle(t)) { dispatchTitle(t); return }
        }

        // Strategy 3: title before @channel — fallback during playback
        val atMatch = AT_CHANNEL_PATTERN.matcher(allText)
        if (atMatch.find()) {
            val t = atMatch.group(1)?.trim() ?: return
            if (isCleanTitle(t)) { dispatchTitle(t); return }
        }

        // Strategy 4: repeated title scan (minimized player)
        val repeated = extractRepeatedTitle(allText)
        if (repeated != null && isCleanTitle(repeated)) {
            dispatchTitle(repeated)
        }
    }

    /**
     * Filtered node text collector.
     * Skips: known chrome widget classes, short clickable leaf nodes.
     * Separates text with \n so line-anchored regex patterns work correctly.
     */
    private fun collectAllNodeText(node: AccessibilityNodeInfo): String {
        val sb = StringBuilder()
        try {
            val className = node.className?.toString() ?: ""

            // (a) Skip known non-title widget classes entirely
            val isSkippedClass = SKIP_NODE_CLASSES.any {
                className.endsWith(it.substringAfterLast('.'))
            }

            // (b) Skip short clickable leaf nodes (buttons, icon labels)
            val textLen      = node.text?.length ?: 0
            val isButtonLike = node.isClickable && textLen in 1..25 && node.childCount == 0

            if (!isSkippedClass && !isButtonLike) {
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

    /**
     * Gate-keeper: returns true only if the candidate is a real video title.
     *
     * Rules (in order of cost, cheapest first):
     *  1. Length: 8–200 chars
     *  2. Not in SKIP_TITLES / SKIP_SYSTEM
     *  3. Not a channel handle (CamelCase, no spaces, no punctuation)
     *  4. Must contain at least 2 words  ← NEW: blocks "Rick Astley", "MichaelJackson"
     *  5. No 2+ uppercase abbreviations  (e.g. "PTE. LTD. SG")
     */
    private fun isCleanTitle(text: String): Boolean {
        // 1. Length gate
        if (text.length < 8 || text.length > 200) return false

        // 2. Blocklists
        if (SKIP_TITLES.any { text.contains(it, ignoreCase = true) }) return false
        if (SKIP_SYSTEM.any { text.contains(it, ignoreCase = true) }) return false

        // 3. Channel handle: CamelCase single token with no spaces or punctuation
        //    e.g. "RickAstleyYT", "MichaelJackson", "VEVOofficial"
        if (CHANNEL_HANDLE_RE.matches(text.trim())) {
            println("[CF_SERVICE] ⛔ Skipped channel handle: $text")
            return false
        }

        // 4. Must contain at least 2 whitespace-separated words.
        //    Real titles: "Rick Astley – Never Gonna Give You Up" ✓
        //    Channel names / single labels: "Rick Astley" ✗  "More actions" ✗
        //    Exception: allow single-word titles with digits/symbols (e.g. "Minecraft: 1.21")
        val wordCount = text.trim().split(Regex("\\s+")).size
        if (wordCount < 2) {
            println("[CF_SERVICE] ⛔ Skipped single-word string: $text")
            return false
        }

        // 5. Reject strings that look like system/corp labels (2+ ALLCAPS abbreviations)
        val abbrCount = Regex("[A-Z]{2,4}\\.").findAll(text).count()
        if (abbrCount >= 2) return false

        return true
    }

    /**
     * Scan-based repeated title detection for the minimized player,
     * where the title node appears twice in the accessibility tree.
     */
    private fun extractRepeatedTitle(text: String): String? {
        val candidates = text
            .split(Regex("[\\n|•·–—]+"))
            .map { it.trim() }
            .filter { it.length in 12..200 }
            .distinctBy { it.lowercase() }

        for (candidate in candidates) {
            if (!isCleanTitle(candidate)) continue
            var idx = 0; var count = 0
            while (true) {
                idx = text.indexOf(candidate, idx)
                if (idx == -1) break
                count++; idx += candidate.length
            }
            if (count >= 2) return candidate
        }
        return null
    }

    private fun dispatchTitle(title: String) {
        if (title.length < 8) return          // raised from 5 → 8 to match isCleanTitle
        if (title == lastSentTitle) return

        lastSentTitle  = title
        lastSentTimeMs = System.currentTimeMillis()
        println("[CF_SERVICE] ✓ Detected title: $title")

        // Immediately show spinner in UI while backend classifies
        broadcastResult(videoId = title, label = "Analyzing", score = 0f, cached = false)
        scope.launch { classifyByTitle(title) }
    }

    private fun handleVideoId(videoId: String) {
        if (videoId == lastSentTitle) return
        lastSentTitle  = videoId
        lastSentTimeMs = System.currentTimeMillis()
        broadcastResult(videoId = videoId, label = "Analyzing", score = 0f, cached = false)
        scope.launch {
            classifyByUrl(
                videoId  = videoId,
                videoUrl = "https://www.youtube.com/watch?v=$videoId",
                thumbUrl = "https://i.ytimg.com/vi/$videoId/hqdefault.jpg"
            )
        }
    }

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
            println("[CF_SERVICE] ✗ classify_by_title error: ${e.message}")
        }
    }

    private fun classifyByUrl(videoId: String, videoUrl: String, thumbUrl: String) {
        try {
            val body    = JSONObject().apply {
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
            println("[CF_SERVICE] ✗ classify_full error: ${e.message}")
        }
    }

    private fun handleClassificationResult(json: JSONObject) {
        val label   = json.optString("oir_label", "Neutral")
        val score   = json.optDouble("score_final", 0.5)
        val cached  = json.optBoolean("cached", false)
        val videoId = json.optString("video_id", "unknown")
        println("[CF_SERVICE] $videoId → $label ($score) cached=$cached")
        broadcastResult(videoId = videoId, label = label, score = score.toFloat(), cached = cached)
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
        lastSentTitle  = ""
        lastSentTimeMs = 0L
    }
}
