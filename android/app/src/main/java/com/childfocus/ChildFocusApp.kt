package com.childfocus

import android.app.Application
import android.content.Intent
import com.childfocus.service.WebBlockerManager

/**
 * Application class — initialises singletons and ensures long-running
 * foreground services are running whenever the app process starts.
 *
 * Registered in AndroidManifest.xml:
 *   <application android:name=".ChildFocusApp" ...>
 */
class ChildFocusApp : Application() {
    override fun onCreate() {
        super.onCreate()

        // Initialise WebBlockerManager with an application-level Context
        // before any Service, Activity or ViewModel tries to use it.
        WebBlockerManager.init(this)

        // Ensure BlockOverlayService is running as soon as the process starts.
        // startForegroundService is safe to call even if the service is already
        // running — Android just delivers another onStartCommand() call.
        startForegroundService(Intent(this, BlockOverlayService::class.java))
    }
}