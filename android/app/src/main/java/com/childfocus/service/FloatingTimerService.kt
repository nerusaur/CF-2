package com.childfocus.service

import android.app.*
import android.content.Intent
import android.graphics.PixelFormat
import android.os.*
import android.view.*
import android.widget.TextView
import androidx.core.app.NotificationCompat

class FloatingTimerService : Service() {

    companion object {
        const val EXTRA_LIMIT   = "limit"
        const val EXTRA_PACKAGE = "package"

        private const val DEFAULT_PACKAGE = "com.google.android.youtube"

        // Sentinel value meaning "limit was not passed in — derive from prefs"
        private const val LIMIT_NOT_PROVIDED = -99
    }

    private lateinit var windowManager: WindowManager
    private lateinit var timerView: TextView

    private var timeLeftMillis: Long = 0
    private var trackedPackage: String = DEFAULT_PACKAGE

    // ── FIX #2: guard flag so rapid window events don't keep restarting ──
    @Volatile private var isRunning = false

    private val handler = Handler(Looper.getMainLooper())

    private val updateRunnable = object : Runnable {
        override fun run() {
            timeLeftMillis -= 1000

            if (timeLeftMillis <= 0) {
                timerView.text = "⛔ Time's up"
                onTimerExpired()
                return
            }

            val minutes = timeLeftMillis / 60_000
            val seconds = (timeLeftMillis % 60_000) / 1000
            timerView.text = if (minutes > 0) "⏱ ${minutes}m ${seconds}s" else "⏱ ${seconds}s"

            handler.postDelayed(this, 1000)
        }
    }

    override fun onCreate() {
        super.onCreate()

        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager

        timerView = TextView(this).apply {
            textSize = 18f
            setBackgroundColor(0xAA000000.toInt())
            setTextColor(0xFFFFFFFF.toInt())
            setPadding(20, 10, 20, 10)
        }

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.END
            x = 50
            y = 100
        }

        windowManager.addView(timerView, params)

        startForegroundNotification()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {

        // ── FIX #1: null intent means Android restarted a START_STICKY
        //    service with no data. timeLeftMillis would stay 0 and
        //    onTimerExpired() would fire immediately — causing a random
        //    home-screen kick. Stop cleanly instead.
        if (intent == null) {
            println("[TIMER] Restarted with null intent — stopping cleanly")
            stopSelf()
            return START_NOT_STICKY
        }

        val incomingPkg = intent.getStringExtra(EXTRA_PACKAGE) ?: DEFAULT_PACKAGE

        // ── FIX #2: if we are already counting down for this exact package,
        //    ignore the redundant start (e.g. rapid TYPE_WINDOW_STATE_CHANGED
        //    events from the accessibility service). Without this guard the
        //    countdown resets on every window event and never reaches zero
        //    normally.
        if (isRunning && incomingPkg == trackedPackage) {
            println("[TIMER] Already running for $trackedPackage — ignoring redundant start")
            return START_NOT_STICKY
        }

        isRunning      = true
        trackedPackage = incomingPkg

        val limitMinutes = intent.getIntExtra(EXTRA_LIMIT, LIMIT_NOT_PROVIDED)

        // ── FIX #3: when limitMinutes is not provided (e.g., onTaskRemoved
        //    restart path), fall back to the value we persisted in prefs
        //    rather than defaulting to 1 minute, which caused premature expiry.
        val resolvedLimitMinutes = when (limitMinutes) {
            LIMIT_NOT_PROVIDED -> getSavedLimit()
            else               -> limitMinutes.also { saveLimitToPrefs(it) }
        }

        timeLeftMillis = when (resolvedLimitMinutes) {
            -1   -> {
                // 10-second test mode — still respect any remaining usage
                val remaining = ScreenTimeManager.getRemainingMs(this, trackedPackage)
                if (remaining == Long.MAX_VALUE) 10_000L else minOf(10_000L, remaining)
            }
            else -> {
                val remaining = ScreenTimeManager.getRemainingMs(this, trackedPackage)
                if (remaining == Long.MAX_VALUE) resolvedLimitMinutes * 60_000L else remaining
            }
        }

        println("[TIMER] $trackedPackage remaining: ${timeLeftMillis / 1000}s")

        handler.removeCallbacks(updateRunnable)

        if (timeLeftMillis <= 0) {
            // Already exceeded — fire immediately without showing the timer.
            onTimerExpired()
        } else {
            handler.post(updateRunnable)
        }

        // ── FIX #1 (continued): use START_NOT_STICKY so Android does NOT
        //    auto-restart this service with a null intent after it is killed.
        //    The onTaskRemoved() path below handles intentional restarts.
        return START_NOT_STICKY
    }

    /**
     * Called when the countdown reaches zero.
     * Persists the exceeded flag, sends the user home, and stops the service.
     */
    private fun onTimerExpired() {
        ScreenTimeManager.markExceeded(this, trackedPackage)
        println("[TIMER] ✓ Marked exceeded + navigating home: $trackedPackage")

        val homeIntent = Intent(Intent.ACTION_MAIN).apply {
            addCategory(Intent.CATEGORY_HOME)
            flags = Intent.FLAG_ACTIVITY_NEW_TASK
        }
        startActivity(homeIntent)

        stopSelf()
    }

    /**
     * Schedules a clean self-restart via AlarmManager when the user swipes
     * the app away from Recents. The restart intent now includes EXTRA_LIMIT
     * (restored from SharedPreferences) so onStartCommand never falls back
     * to the wrong default of 1 minute.
     */
    override fun onTaskRemoved(rootIntent: Intent?) {
        super.onTaskRemoved(rootIntent)

        if (ScreenTimeManager.isExceeded(this, trackedPackage)) return

        val restartIntent = Intent(applicationContext, FloatingTimerService::class.java).apply {
            putExtra(EXTRA_PACKAGE, trackedPackage)
            // ── FIX #3: always include the saved limit so the restart path
            //    does not fall back to getIntExtra default of 1 minute.
            putExtra(EXTRA_LIMIT, getSavedLimit())
        }

        val pendingIntent = PendingIntent.getService(
            applicationContext,
            10,
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

    // ── FIX #3 helpers: persist and restore the limit across service
    //    restarts so onTaskRemoved always has the correct value to pass back.
    private fun saveLimitToPrefs(limitMinutes: Int) {
        getSharedPreferences("timer_prefs", MODE_PRIVATE)
            .edit()
            .putInt("limit_$trackedPackage", limitMinutes)
            .apply()
    }

    private fun getSavedLimit(): Int =
        getSharedPreferences("timer_prefs", MODE_PRIVATE)
            .getInt("limit_$trackedPackage", 60)   // 60 min is a safe fallback

    private fun startForegroundNotification() {
        val channelId = "timer_channel"

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId,
                "Timer Service",
                NotificationManager.IMPORTANCE_LOW
            )
            getSystemService(NotificationManager::class.java)
                .createNotificationChannel(channel)
        }

        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("Screen Time Running")
            .setContentText("ChildFocus is monitoring screen time")
            .setSmallIcon(android.R.drawable.ic_lock_idle_alarm)
            .setOngoing(true)
            .build()

        startForeground(2, notification)
    }

    override fun onDestroy() {
        super.onDestroy()
        isRunning = false   // ── FIX #2: reset guard so future starts work
        handler.removeCallbacks(updateRunnable)
        if (::timerView.isInitialized) {
            try { windowManager.removeView(timerView) } catch (_: Exception) {}
        }
    }

    override fun onBind(intent: Intent?) = null
}