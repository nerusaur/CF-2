package com.childfocus

import android.app.*
import android.content.Context
import android.content.Intent
import android.graphics.BitmapFactory
import android.graphics.PixelFormat
import android.net.Uri
import android.os.*
import android.provider.Settings
import android.view.*
import android.widget.ImageView
import android.widget.TextView
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*
import java.net.URL

class BlockOverlayService : Service() {

    private lateinit var windowManager: WindowManager
    private var overlayView: View? = null
    private val serviceScope = CoroutineScope(Dispatchers.IO)
    private val mainHandler = Handler(Looper.getMainLooper())

    // Keeps the last ACTION_SHOW extras so onTaskRemoved restart can re-show
    // the overlay without losing context.
    private var lastVideoId = ""
    private var lastLabel   = "Overstimulating"
    private var lastScore   = 0f

    private fun loadThumbnail(videoId: String, imageView: ImageView) {
        serviceScope.launch {
            try {
                val url    = URL("https://i.ytimg.com/vi/$videoId/mqdefault.jpg")
                val bitmap = BitmapFactory.decodeStream(url.openStream())
                mainHandler.post { imageView.setImageBitmap(bitmap) }
            } catch (_: Exception) {}
        }
    }

    data class RecommendedVideo(
        val title: String,
        val videoId: String,
        val category: String
    )

    companion object {
        const val ACTION_SHOW    = "com.childfocus.overlay.SHOW"
        const val ACTION_HIDE    = "com.childfocus.overlay.HIDE"
        const val EXTRA_VIDEO_ID = "video_id"
        const val EXTRA_LABEL    = "label"
        const val EXTRA_SCORE    = "score"

        val RECOMMENDED_VIDEOS = listOf(
            // Educational
            RecommendedVideo("Sesame Street: Do De Rubber Duck",    "KpM6oFHmBBQ", "Educational"),
            RecommendedVideo("Sesame Street: Elmo's Got the Moves", "FHmK8aiwXs8", "Educational"),
            RecommendedVideo("Khan Academy Kids: Adding Numbers",   "gA2B_kelXpg", "Educational"),
            RecommendedVideo("Numberblocks: One",                   "cpFHMNSPNTk", "Educational"),
            RecommendedVideo("Alphablocks: A",                      "BJCGMMWpEWo", "Educational"),
            RecommendedVideo("SciShow Kids: Why Do We Dream?",      "GiBGwFOL8qA", "Educational"),
            RecommendedVideo("Nat Geo Kids: Amazing Animals",       "iqHLDEMoiKQ", "Educational"),
            RecommendedVideo("Crash Course Kids: Ecosystems",       "IDV8ou3FbwI", "Educational"),
            RecommendedVideo("PBS Kids: Curious George",            "g_i_tVYTYp4", "Educational"),
            RecommendedVideo("Peekaboo Kidz: Solar System",         "mQrlgH97v94", "Educational"),
            // Neutral
            RecommendedVideo("Peppa Pig: Muddy Puddles",            "keXn2HB4MkI", "Neutral"),
            RecommendedVideo("Peppa Pig: The Playground",           "0SYcr5Qv0mg", "Neutral"),
            RecommendedVideo("Bluey: Camping",                      "UkH4kGzBPCk", "Neutral"),
            RecommendedVideo("Bluey: The Pool",                     "Ek5J3rEMjOU", "Neutral"),
            RecommendedVideo("Paw Patrol: Pups Save Ryder",         "bLzCVIFhZYE", "Neutral"),
            RecommendedVideo("Thomas & Friends: Go Go Thomas",      "S-b_RmMjqF8", "Neutral"),
            RecommendedVideo("Mr Bean: Do-It-Yourself Mr Bean",     "b-Kd9MuFMBo", "Neutral"),
            RecommendedVideo("Shaun the Sheep: Off the Baa",        "YK86H2Mc9kc", "Neutral"),
            RecommendedVideo("Lego: City Mini Movies",              "L4LqBNKBVgA", "Neutral"),
            RecommendedVideo("Paw Patrol: Sea Patrol",              "1R3VoAkHPEI", "Neutral"),
        )

        fun show(context: Context, videoId: String, label: String, score: Float) {
            val intent = Intent(context, BlockOverlayService::class.java).apply {
                action = ACTION_SHOW
                putExtra(EXTRA_VIDEO_ID, videoId)
                putExtra(EXTRA_LABEL, label)
                putExtra(EXTRA_SCORE, score)
            }
            context.startForegroundService(intent)
        }

        fun hide(context: Context) {
            val intent = Intent(context, BlockOverlayService::class.java).apply {
                action = ACTION_HIDE
            }
            context.startService(intent)
        }
    }

    override fun onCreate() {
        super.onCreate()
        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager
        startForegroundWithNotification()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_SHOW -> {
                lastVideoId = intent.getStringExtra(EXTRA_VIDEO_ID) ?: ""
                lastLabel   = intent.getStringExtra(EXTRA_LABEL)    ?: "Overstimulating"
                lastScore   = intent.getFloatExtra(EXTRA_SCORE, 0f)
                closeYouTube()
                showOverlay(lastVideoId, lastLabel, lastScore)
            }
            ACTION_HIDE -> {
                removeOverlay()
                stopSelf()
            }
            // null intent = system restarted us after swipe-kill → re-show last overlay
            null -> {
                if (lastVideoId.isNotEmpty()) {
                    showOverlay(lastVideoId, lastLabel, lastScore)
                }
            }
        }

        // START_STICKY → system will restart this service automatically if killed.
        // stopWithTask="false" in the manifest ensures a swipe-kill does not stop it.
        return START_STICKY
    }

    /**
     * Called when the user swipes the app away from Recents.
     * AlarmManager schedules a self-restart ~1 second later so the overlay
     * reappears even if the system tears down the process.
     */
    override fun onTaskRemoved(rootIntent: Intent?) {
        super.onTaskRemoved(rootIntent)

        val restartIntent = Intent(applicationContext, BlockOverlayService::class.java).apply {
            action = ACTION_SHOW
            putExtra(EXTRA_VIDEO_ID, lastVideoId)
            putExtra(EXTRA_LABEL, lastLabel)
            putExtra(EXTRA_SCORE, lastScore)
        }

        val pendingIntent = PendingIntent.getService(
            applicationContext,
            20,
            restartIntent,
            PendingIntent.FLAG_ONE_SHOT or PendingIntent.FLAG_IMMUTABLE
        )

        val alarmManager = getSystemService(ALARM_SERVICE) as AlarmManager
        alarmManager.set(
            AlarmManager.ELAPSED_REALTIME,
            SystemClock.elapsedRealtime() + 1_000L,
            pendingIntent
        )
    }

    private fun closeYouTube() {
        // 1. Send YouTube to background by going home first
        val homeIntent = Intent(Intent.ACTION_MAIN).apply {
            addCategory(Intent.CATEGORY_HOME)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        startActivity(homeIntent)

        // 2. Force-stop YouTube after a brief delay (gives HOME intent time to fire)
        mainHandler.postDelayed({
            val activityManager = getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
            activityManager.killBackgroundProcesses("com.google.android.youtube")
        }, 300)
    }

    private fun showOverlay(videoId: String, label: String, score: Float) {
        if (overlayView != null) return

        // ── FIX: guard against missing SYSTEM_ALERT_WINDOW permission ──────
        // TYPE_APPLICATION_OVERLAY requires the user to grant "Draw over other
        // apps" permission at runtime. Without this check, addView() throws a
        // WindowManager$BadTokenException and the overlay silently never appears.
        // If permission is missing, log it and bail — OverlayPermissionHelper
        // in MainActivity should have already prompted the user for this.
        if (!Settings.canDrawOverlays(this)) {
            println("[BlockOverlay] ✗ SYSTEM_ALERT_WINDOW not granted — overlay skipped. " +
                    "Ensure OverlayPermissionHelper.requestIfNeeded() is called from MainActivity.")
            return
        }

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or
                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                    WindowManager.LayoutParams.FLAG_WATCH_OUTSIDE_TOUCH,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
        }

        overlayView = android.view.LayoutInflater.from(this)
            .inflate(R.layout.overlay_blocked, null)

        overlayView?.apply {
            findViewById<TextView>(R.id.tvScore)?.text =
                "Score: ${"%.2f".format(score)} (threshold: 0.20)"

            val container = findViewById<android.widget.LinearLayout>(R.id.llSuggestions)

            val picks =
                RECOMMENDED_VIDEOS.filter { it.category == "Educational" }.shuffled().take(3) +
                        RECOMMENDED_VIDEOS.filter { it.category == "Neutral" }.shuffled().take(3)

            picks.forEach { video ->
                val card = android.view.LayoutInflater.from(context)
                    .inflate(R.layout.item_recommendation, container, false)

                card.findViewById<TextView>(R.id.tv_video_title)?.text    = video.title
                card.findViewById<TextView>(R.id.tv_video_category)?.text = video.category
                card.findViewById<ImageView>(R.id.iv_thumbnail)
                    ?.let { loadThumbnail(video.videoId, it) }

                card.setOnClickListener {
                    removeOverlay()
                    openRecommendedVideoClean(video.title)
                    stopSelf()
                }
                container?.addView(card)
            }
        }

        try {
            windowManager.addView(overlayView, params)
        } catch (e: Exception) {
            // Catch any remaining WindowManager exceptions (e.g. permission
            // revoked between the check above and addView()) so the service
            // doesn't crash the whole app.
            println("[BlockOverlay] ✗ addView failed: ${e.message}")
            overlayView = null
        }
    }

    private fun openRecommendedVideoClean(title: String) {
        val resetIntent = packageManager
            .getLaunchIntentForPackage("com.google.android.youtube")
            ?.apply { addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK) }
        if (resetIntent != null) startActivity(resetIntent)

        mainHandler.postDelayed({
            val query = Uri.encode("$title for kids")
            val searchIntent = Intent(Intent.ACTION_SEARCH).apply {
                `package` = "com.google.android.youtube"
                putExtra("query", "$title for kids")
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            val webFallback = Intent(
                Intent.ACTION_VIEW,
                Uri.parse("https://www.youtube.com/results?search_query=$query")
            ).apply { addFlags(Intent.FLAG_ACTIVITY_NEW_TASK) }
            try { startActivity(searchIntent) }
            catch (_: Exception) { try { startActivity(webFallback) } catch (_: Exception) {} }
        }, 900)
    }

    private fun removeOverlay() {
        overlayView?.let {
            try { windowManager.removeView(it) } catch (_: Exception) {}
            overlayView = null
        }
    }

    private fun startForegroundWithNotification() {
        val channelId = "childfocus_overlay"
        val channel = NotificationChannel(
            channelId,
            "ChildFocus Safety Overlay",
            NotificationManager.IMPORTANCE_LOW
        )
        getSystemService(NotificationManager::class.java)
            .createNotificationChannel(channel)

        val notification: Notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("ChildFocus is active")
            .setContentText("Monitoring YouTube for overstimulating content")
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setOngoing(true) // Persistent — cannot be dismissed by the user
            .build()

        startForeground(1, notification)
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        super.onDestroy()
        removeOverlay()
        serviceScope.cancel()
    }
}