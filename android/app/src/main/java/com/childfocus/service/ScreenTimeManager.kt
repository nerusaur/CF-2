package com.childfocus.service

import android.content.Context
import android.content.SharedPreferences
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Tracks per-app foreground time for the current calendar day and checks it
 * against the limits saved by ScreenTimeScreen.kt.
 *
 * ─── How it fits together ───────────────────────────────────────────────────
 *  ScreenTimeScreen.kt  →  writes limits to "screen_time_prefs"
 *  ScreenTimeManager    →  reads those limits, writes today's usage to
 *                          "screen_time_usage", exposes isOverLimit()
 *  ChildFocusAccessibilityService  →  calls onAppForeground() / onAppBackground()
 *                                     on every TYPE_WINDOW_STATE_CHANGED event
 * ────────────────────────────────────────────────────────────────────────────
 *
 * ─── FIX: Persistent exceeded flag ─────────────────────────────────────────
 *  Previously, exceeded state lived only in the accessibility service's memory
 *  (screenTimeLimitEnforced). This flag was reset the moment the user left the
 *  blocked app (home press), so reopening YouTube would pass the isOverLimit()
 *  check if ScreenTimeManager's flushed usage was fractionally under the limit
 *  (e.g., due to up-to-60-second flush lag from the ticker interval).
 *
 *  Now markExceeded() writes a boolean to "screen_time_exceeded" SharedPrefs.
 *  onAppForeground() checks isExceeded() BEFORE computing usage, so a package
 *  that has been marked exceeded is ALWAYS blocked on re-open — regardless of
 *  the exact millisecond difference between the timer and the persisted usage.
 *
 *  The flag is cleared:
 *    • At midnight (maybeResetDay)
 *    • Per-package when the parent explicitly calls clearExceeded() — e.g.,
 *      when the user turns screen time OFF for that app in ScreenTimeScreen.
 *    • All packages via clearAllExceeded() — used by ScreenTimeScreen when the
 *      master screen-time toggle is disabled.
 * ────────────────────────────────────────────────────────────────────────────
 *
 * Thread-safety: all public methods are @Synchronized.
 */
object ScreenTimeManager {

    // ── SharedPreferences keys (must match ScreenTimeScreen.kt exactly) ───
    private const val CONFIG_PREFS    = "screen_time_prefs"
    private const val USAGE_PREFS     = "screen_time_usage"
    private const val EXCEEDED_PREFS  = "screen_time_exceeded"   // ← NEW
    private const val KEY_TOTAL_LIMIT = "total_daily_limit_minutes"
    private const val KEY_DATE        = "usage_date"

    private fun limitKey(pkg: String)    = "limit_$pkg"
    private fun enabledKey(pkg: String)  = "enabled_$pkg"
    private fun usageKey(pkg: String)    = "usage_ms_$pkg"
    private fun exceededKey(pkg: String) = "exceeded_$pkg"       // ← NEW
    private const val KEY_TOTAL_USAGE    = "usage_ms_total"

    // ── Packages whose screen time is managed (mirrors ScreenTimeScreen) ──
    val TRACKED_PACKAGES = setOf(
        "com.google.android.youtube",
        "com.zhiliaoapp.musically",
        "com.instagram.android",
        "com.facebook.katana",
        "com.roblox.client",
        "com.android.chrome",
        "com.netflix.mediaclient",
        "com.snapchat.android",
    )

    // ── In-memory foreground tracking ─────────────────────────────────────
    @Volatile private var foregroundPkg   = ""
    @Volatile private var foregroundStart = 0L   // epoch ms when current app entered fg

    // ─────────────────────────────────────────────────────────────────────
    // PUBLIC API — EXCEEDED FLAG  (NEW)
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Permanently marks [pkg] as having exceeded its daily limit.
     * Persisted to SharedPreferences so it survives service restarts and
     * app re-opens within the same calendar day.
     *
     * Call this from:
     *  • ChildFocusAccessibilityService.enforceScreenTimeLimit()
     *  • FloatingTimerService when its countdown reaches zero
     */
    @Synchronized
    fun markExceeded(context: Context, pkg: String) {
        exceededPrefs(context).edit()
            .putBoolean(exceededKey(pkg), true)
            .apply()
        println("[SCREEN TIME] ✓ Marked exceeded: $pkg")
    }

    /**
     * Returns true if [pkg] was previously marked as exceeded today.
     * Checked at the TOP of onAppForeground() so the app is blocked on
     * re-open even if the accumulated usage milliseconds fell just short of
     * the limit due to flush-timing gaps.
     *
     * ── FIX #3: guard with TRACKED_PACKAGES ──────────────────────────────
     * Without this guard, a stale or accidentally-written SharedPrefs entry
     * for an untracked package could cause it to be blocked. Only packages
     * the user has explicitly configured can ever be "exceeded".
     */
    @Synchronized
    fun isExceeded(context: Context, pkg: String): Boolean {
        if (pkg !in TRACKED_PACKAGES) return false   // ← FIX #3
        return exceededPrefs(context).getBoolean(exceededKey(pkg), false)
    }

    /**
     * Clears the exceeded flag for a single package.
     * Call this when the user turns screen time OFF for [pkg] in the UI,
     * so they can open the app again immediately.
     */
    @Synchronized
    fun clearExceeded(context: Context, pkg: String) {
        exceededPrefs(context).edit()
            .putBoolean(exceededKey(pkg), false)
            .apply()
    }

    /**
     * Clears exceeded flags for ALL tracked packages.
     * Call this when the master screen-time toggle is disabled in the UI.
     */
    @Synchronized
    fun clearAllExceeded(context: Context) {
        exceededPrefs(context).edit().clear().apply()
    }

    // ─────────────────────────────────────────────────────────────────────
    // PUBLIC API — FOREGROUND TRACKING
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Call every time a new package comes to the foreground
     * (TYPE_WINDOW_STATE_CHANGED with a new packageName).
     *
     * ── FIX ──────────────────────────────────────────────────────────────
     * isExceeded() is now checked BEFORE computing live usage so that a
     * package that was marked exceeded earlier today is ALWAYS blocked on
     * re-open, regardless of flush-timing discrepancies.
     * ─────────────────────────────────────────────────────────────────────
     *
     * @return true if this app has now exceeded its configured daily limit.
     */
    @Synchronized
    fun onAppForeground(context: Context, packageName: String): Boolean {
        val now = System.currentTimeMillis()

        // Flush elapsed time for whatever was previously in the foreground.
        flushCurrent(context, now)

        // Only track packages the user has configured.
        return if (packageName in TRACKED_PACKAGES) {
            foregroundPkg   = packageName
            foregroundStart = now

            // ── FIX: check persisted exceeded flag first ──────────────────
            // This is the key guard that prevents re-opening a blocked app.
            // Even if the live usedMs is fractionally below the limit, a
            // previously enforced block will still fire here.
            if (isExceeded(context, packageName)) {
                println("[SCREEN TIME] ✗ Re-open blocked (exceeded flag set): $packageName")
                true
            } else {
                isOverLimit(context, packageName)
            }
        } else {
            foregroundPkg   = ""
            foregroundStart = 0L
            false
        }
    }

    /**
     * Call when the monitored app goes to the background (home pressed, etc.).
     * The accessibility service can call this on any non-tracked package switch.
     */
    @Synchronized
    fun onAppBackground(context: Context) {
        flushCurrent(context, System.currentTimeMillis())
        foregroundPkg   = ""
        foregroundStart = 0L
    }

    /**
     * Periodic tick — call every ~60 s from the accessibility service.
     * Flushes the current session window and re-checks the limit so that the
     * block fires mid-session, not just when the user switches apps.
     *
     * @return true if the currently-foregrounded app is now over its limit.
     */
    @Synchronized
    fun tick(context: Context): Boolean {
        if (foregroundPkg.isEmpty()) return false
        val now = System.currentTimeMillis()
        // Flush and restart the window so we don't double-count on the next tick.
        flushCurrent(context, now)
        foregroundStart = now
        // isExceeded is implicitly checked via onAppForeground on re-open;
        // here we still check isOverLimit for mid-session enforcement.
        return isExceeded(context, foregroundPkg) || isOverLimit(context, foregroundPkg)
    }

    /** How many milliseconds of [pkg] have been used today. */
    @Synchronized
    fun getUsageMs(context: Context, pkg: String): Long {
        maybeResetDay(context)
        return usagePrefs(context).getLong(usageKey(pkg), 0L)
    }

    /** How many minutes of [pkg] have been used today (rounded). */
    fun getUsageMinutes(context: Context, pkg: String): Int =
        (getUsageMs(context, pkg) / 60_000L).toInt()

    /**
     * Returns how many milliseconds remain for [pkg] today.
     * Used by FloatingTimerService to start from the REMAINING time instead
     * of always resetting to the full limit.
     *
     * Returns 0 if the limit has already been reached or exceeded.
     */
    @Synchronized
    fun getRemainingMs(context: Context, pkg: String): Long {
        val cfg = configPrefs(context)
        if (!cfg.getBoolean(enabledKey(pkg), false)) return Long.MAX_VALUE

        val limitMinutes = cfg.getInt(limitKey(pkg), 60)
        val limitMs = when (limitMinutes) {
            -1   -> 10_000L
            else -> limitMinutes * 60_000L
        }
        if (limitMs <= 0) return Long.MAX_VALUE

        val usedMs = getUsageMs(context, pkg)
        return maxOf(0L, limitMs - usedMs)
    }

    /** The package currently tracked as being in the foreground (empty if none). */
    fun getCurrentForegroundPkg(): String = foregroundPkg

    // ─────────────────────────────────────────────────────────────────────
    // PRIVATE HELPERS
    // ─────────────────────────────────────────────────────────────────────

    /** Returns true if [pkg] has a limit enabled AND usage >= limit. */
    private fun isOverLimit(context: Context, pkg: String): Boolean {
        val cfg = configPrefs(context)

        if (!cfg.getBoolean(enabledKey(pkg), false)) return false

        val limitMinutes = cfg.getInt(limitKey(pkg), 60)

        val limitMs = when (limitMinutes) {
            -1   -> 10_000L   // 10-second test option
            else -> limitMinutes * 60_000L
        }

        if (limitMs <= 0) return false

        val usedMs = getUsageMs(context, pkg)

        println("[SCREEN TIME] $pkg used: ${usedMs / 1000}s / limit: ${limitMs / 1000}s")

        return usedMs >= limitMs
    }

    /** Flush [foregroundPkg]'s elapsed time up to [nowMs] into storage. */
    private fun flushCurrent(context: Context, nowMs: Long) {
        val pkg   = foregroundPkg
        val start = foregroundStart
        if (pkg.isEmpty() || start <= 0L) return
        val delta = nowMs - start
        if (delta > 0) addUsage(context, pkg, delta)
    }

    private fun addUsage(context: Context, pkg: String, deltaMs: Long) {
        if (deltaMs <= 0) return
        maybeResetDay(context)
        val prefs     = usagePrefs(context)
        val prev      = prefs.getLong(usageKey(pkg), 0L)
        val totalPrev = prefs.getLong(KEY_TOTAL_USAGE, 0L)
        prefs.edit()
            .putLong(usageKey(pkg), prev + deltaMs)
            .putLong(KEY_TOTAL_USAGE, totalPrev + deltaMs)
            .apply()
    }

    /**
     * Clears all usage counters AND exceeded flags when the calendar date
     * changes (midnight reset).
     */
    private fun maybeResetDay(context: Context) {
        val today = SimpleDateFormat("yyyyMMdd", Locale.US).format(Date())
        val prefs = usagePrefs(context)
        if (prefs.getString(KEY_DATE, "") != today) {
            prefs.edit().clear().putString(KEY_DATE, today).apply()
            // ── FIX: also wipe exceeded flags at the daily reset ──────────
            clearAllExceeded(context)
            println("[SCREEN TIME] ✓ Daily reset — usage + exceeded flags cleared")
        }
    }

    private fun configPrefs(context: Context): SharedPreferences =
        context.getSharedPreferences(CONFIG_PREFS, Context.MODE_PRIVATE)

    private fun usagePrefs(context: Context): SharedPreferences =
        context.getSharedPreferences(USAGE_PREFS, Context.MODE_PRIVATE)

    private fun exceededPrefs(context: Context): SharedPreferences =
        context.getSharedPreferences(EXCEEDED_PREFS, Context.MODE_PRIVATE)
}