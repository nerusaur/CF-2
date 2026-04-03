package com.childfocus.service

import android.app.*
import android.content.Intent
import android.graphics.PixelFormat
import android.os.*
import android.view.*
import android.widget.TextView
import androidx.core.app.NotificationCompat

class FloatingTimerService : Service() {

    private lateinit var windowManager: WindowManager
    private lateinit var timerView: TextView

    private var timeLeftMillis: Long = 0
    private val handler = Handler(Looper.getMainLooper())

    private val updateRunnable = object : Runnable {
        override fun run() {
            timeLeftMillis -= 1000

            if (timeLeftMillis <= 0) {
                timerView.text = "⛔ Time's up"
                showBlockOverlay()
                stopSelf()
                return
            }

            val seconds = timeLeftMillis / 1000
            timerView.text = "⏱ $seconds s"

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
        )

        params.gravity = Gravity.TOP or Gravity.END
        params.x = 50
        params.y = 100

        windowManager.addView(timerView, params)

        startForegroundService()
    }

    private fun startForegroundService() {
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
            .setOngoing(true) // Persistent — cannot be dismissed by the user
            .build()

        startForeground(2, notification)
    }

    private fun showBlockOverlay() {
        val blockView = TextView(this).apply {
            text = "🚫 App blocked"
            textSize = 24f
            setBackgroundColor(0xFF000000.toInt())
            setTextColor(0xFFFFFFFF.toInt())
            gravity = Gravity.CENTER
        }

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
            PixelFormat.TRANSLUCENT
        )

        windowManager.addView(blockView, params)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Only read the extra when it's a fresh start, not a system restart.
        // On system restart (intent == null), timeLeftMillis stays at whatever
        // it was before — ideally you'd persist it to SharedPreferences for
        // a perfect resume, but this prevents an instant crash-restart loop.
        intent?.let {
            val minutes = it.getIntExtra("limit", 1)
            timeLeftMillis = if (minutes == -1) 10_000L else minutes * 60 * 1000L
        }

        handler.removeCallbacks(updateRunnable)
        handler.post(updateRunnable)

        // START_STICKY → Android will restart this service automatically if
        // the system kills it (e.g. low memory). Combined with stopWithTask="false"
        // in the manifest this also survives swipe-kill.
        return START_STICKY
    }

    /**
     * Called when the user swipes the app away from Recents.
     * We schedule a self-restart via AlarmManager so the timer
     * picks back up ~1 second after the swipe.
     */
    override fun onTaskRemoved(rootIntent: Intent?) {
        super.onTaskRemoved(rootIntent)

        val restartIntent = Intent(applicationContext, FloatingTimerService::class.java)
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

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacks(updateRunnable)
        if (::timerView.isInitialized) {
            try { windowManager.removeView(timerView) } catch (_: Exception) {}
        }
    }

    override fun onBind(intent: Intent?) = null
}