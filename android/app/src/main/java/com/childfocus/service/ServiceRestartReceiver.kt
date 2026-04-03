package com.childfocus.service

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import com.childfocus.BlockOverlayService

/**
 * Restores ChildFocus services after:
 *  - Device reboot (BOOT_COMPLETED)
 *  - App update / re-install (MY_PACKAGE_REPLACED)
 *  - Manual restart broadcast (com.childfocus.RESTART_SERVICES)
 *
 * Registered in AndroidManifest.xml with all three intent-filter actions.
 */
class ServiceRestartReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        when (intent.action) {
            Intent.ACTION_BOOT_COMPLETED,
            Intent.ACTION_MY_PACKAGE_REPLACED,
            "com.childfocus.RESTART_SERVICES" -> {
                restoreServices(context)
            }
        }
    }

    private fun restoreServices(context: Context) {
        // BlockOverlayService — always keep alive so the overlay
        // can reappear the moment the accessibility service fires again.
        context.startForegroundService(
            Intent(context, BlockOverlayService::class.java)
        )

        // FloatingTimerService is only meaningful when a session is active,
        // so we do NOT auto-restart it here (it would start with 0 time left
        // and immediately call showBlockOverlay). It will be re-started by
        // ScreenTimeManager when the next session begins.
    }
}