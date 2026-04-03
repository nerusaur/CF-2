package com.childfocus.ui

import android.content.Context
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Block
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Timer
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.childfocus.ui.theme.*

// ── Data ──────────────────────────────────────────────────────────────────────

private data class AppLimit(
    val label: String,
    val packageName: String,
    val icon: String
)

private val TRACKED_APPS = listOf(
    AppLimit("YouTube",   "com.google.android.youtube",  "▶"),
    AppLimit("TikTok",    "com.zhiliaoapp.musically",    "♪"),
    AppLimit("Instagram", "com.instagram.android",       "📷"),
    AppLimit("Facebook",  "com.facebook.katana",         "f"),
    AppLimit("Roblox",    "com.roblox.client",           "🎮"),
    AppLimit("Chrome",    "com.android.chrome",          "🌐"),
    AppLimit("Netflix",   "com.netflix.mediaclient",     "N"),
    AppLimit("Snapchat",  "com.snapchat.android",        "👻"),
)

private const val SCREEN_TIME_PREFS_NAME = "screen_time_prefs"
private const val KEY_TOTAL_LIMIT        = "total_daily_limit_minutes"

private fun limitKey(pkg: String)   = "limit_$pkg"
private fun enabledKey(pkg: String) = "enabled_$pkg"

private fun getPrefs(context: Context) =
    context.getSharedPreferences(SCREEN_TIME_PREFS_NAME, Context.MODE_PRIVATE)

private fun getTotalLimit(context: Context): Int =
    getPrefs(context).getInt(KEY_TOTAL_LIMIT, 120)

private fun setTotalLimit(context: Context, minutes: Int) =
    getPrefs(context).edit().putInt(KEY_TOTAL_LIMIT, minutes).apply()

private fun getAppLimitMinutes(context: Context, pkg: String): Int =
    getPrefs(context).getInt(limitKey(pkg), 60)

private fun setAppLimitMinutes(context: Context, pkg: String, minutes: Int) =
    getPrefs(context).edit().putInt(limitKey(pkg), minutes).apply()

private fun isAppLimitEnabled(context: Context, pkg: String): Boolean =
    getPrefs(context).getBoolean(enabledKey(pkg), false)

private fun setAppLimitEnabled(context: Context, pkg: String, enabled: Boolean) =
    getPrefs(context).edit().putBoolean(enabledKey(pkg), enabled).apply()

private fun formatMinutes(minutes: Int): String {
    val h = minutes / 60; val m = minutes % 60
    return when { h == 0 -> "${m}m"; m == 0 -> "${h}h"; else -> "${h}h ${m}m" }
}

// ── Screen ────────────────────────────────────────────────────────────────────

@Composable
fun ScreenTimeScreen() {
    val context = LocalContext.current

    var totalLimitMinutes by remember { mutableIntStateOf(getTotalLimit(context)) }
    var showTotalDialog   by remember { mutableStateOf(false) }

    val appEnabled = remember {
        mutableStateMapOf<String, Boolean>().also { map ->
            TRACKED_APPS.forEach { map[it.packageName] = isAppLimitEnabled(context, it.packageName) }
        }
    }
    val appLimits = remember {
        mutableStateMapOf<String, Int>().also { map ->
            TRACKED_APPS.forEach { map[it.packageName] = getAppLimitMinutes(context, it.packageName) }
        }
    }
    var editingApp by remember { mutableStateOf<AppLimit?>(null) }

    val bgGradient = Brush.verticalGradient(colors = listOf(CfBgTop, CfBgBottom))

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(bgGradient)
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Header
        item {
            Text(
                text       = "⏱️  Screen Time",
                fontSize   = 22.sp,
                fontWeight = FontWeight.Bold,
                color      = CfTextPrimary
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text     = "Set daily limits for apps on this device.",
                color    = CfTextSecond,
                fontSize = 13.sp
            )
        }

        // Total daily limit card
        item {
            Card(
                shape     = RoundedCornerShape(16.dp),
                colors    = CardDefaults.cardColors(containerColor = CfPurpleLight),
                elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
                modifier  = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier              = Modifier.padding(horizontal = 20.dp, vertical = 16.dp),
                    verticalAlignment     = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Column {
                        Text(
                            text       = "Total Daily Limit",
                            color      = CfIdleText,
                            fontWeight = FontWeight.SemiBold,
                            fontSize   = 14.sp
                        )
                        Text(
                            text       = formatMinutes(totalLimitMinutes),
                            color      = CfPurple,
                            fontSize   = 26.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                    Button(
                        onClick = { showTotalDialog = true },
                        shape   = RoundedCornerShape(12.dp),
                        colors  = ButtonDefaults.buttonColors(containerColor = CfPurple)
                    ) {
                        Text("Edit", color = CfTextOnDark, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }

        // Section label
        item {
            Text(
                text       = "Per-App Limits",
                color      = CfTextSecond,
                fontSize   = 12.sp,
                fontWeight = FontWeight.SemiBold
            )
        }

        // App rows
        items(TRACKED_APPS, key = { it.packageName }) { app ->
            val enabled  = appEnabled[app.packageName] == true
            val limitMin = appLimits[app.packageName] ?: 60
            AppLimitRow(
                app         = app,
                enabled     = enabled,
                limitMin    = limitMin,
                onToggle    = {
                    appEnabled[app.packageName] = it
                    setAppLimitEnabled(context, app.packageName, it)
                },
                onEditClick = { editingApp = app }
            )
        }

        item { Spacer(modifier = Modifier.height(8.dp)) }
    }

    // Total limit dialog
    if (showTotalDialog) {
        LimitPickerDialog(
            title      = "Total Daily Limit",
            currentMin = totalLimitMinutes,
            onConfirm  = { minutes -> totalLimitMinutes = minutes; setTotalLimit(context, minutes); showTotalDialog = false },
            onDismiss  = { showTotalDialog = false }
        )
    }

    // Per-app limit dialog
    editingApp?.let { app ->
        LimitPickerDialog(
            title      = "${app.label} Daily Limit",
            currentMin = appLimits[app.packageName] ?: 60,
            onConfirm  = { minutes -> appLimits[app.packageName] = minutes; setAppLimitMinutes(context, app.packageName, minutes); editingApp = null },
            onDismiss  = { editingApp = null }
        )
    }
}

// ── App row ───────────────────────────────────────────────────────────────────

@Composable
private fun AppLimitRow(
    app: AppLimit,
    enabled: Boolean,
    limitMin: Int,
    onToggle: (Boolean) -> Unit,
    onEditClick: () -> Unit
) {
    Card(
        shape    = RoundedCornerShape(12.dp),
        colors   = CardDefaults.cardColors(
            containerColor = if (enabled) CfSurface else CfSurfaceMuted
        ),
        border   = androidx.compose.foundation.BorderStroke(
            width = 1.dp,
            color = if (enabled) CfPurple.copy(alpha = 0.2f) else CfBorder
        ),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier          = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // App icon
            Surface(
                shape    = RoundedCornerShape(8.dp),
                color    = if (enabled) CfPurpleLight else CfSurfaceMuted,
                modifier = Modifier.size(40.dp)
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Text(text = app.icon, fontSize = 18.sp, textAlign = TextAlign.Center)
                }
            }

            Spacer(modifier = Modifier.width(12.dp))

            // Label + limit
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text       = app.label,
                    color      = if (enabled) CfTextPrimary else CfTextSecond,
                    fontWeight = FontWeight.SemiBold,
                    fontSize   = 14.sp
                )
                if (enabled) {
                    Row(
                        verticalAlignment     = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Icon(
                            imageVector        = Icons.Default.Timer,
                            contentDescription = null,
                            tint               = CfPurple,
                            modifier           = Modifier.size(12.dp)
                        )
                        Text(
                            text     = formatMinutes(limitMin) + " / day",
                            color    = CfPurple,
                            fontSize = 12.sp
                        )
                    }
                } else {
                    Text(text = "No limit", color = CfTextHint, fontSize = 12.sp)
                }
            }

            // Edit button
            if (enabled) {
                TextButton(
                    onClick        = onEditClick,
                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text("Edit", color = CfPurple, fontSize = 12.sp)
                }
            }

            // Toggle
            Switch(
                checked         = enabled,
                onCheckedChange = onToggle,
                colors          = SwitchDefaults.colors(
                    checkedThumbColor   = CfTextOnDark,
                    checkedTrackColor   = CfGreen,
                    uncheckedThumbColor = CfTextHint,
                    uncheckedTrackColor = CfBorder
                )
            )
        }
    }
}

// ── Limit picker dialog ───────────────────────────────────────────────────────

@Composable
private fun LimitPickerDialog(
    title: String,
    currentMin: Int,
    onConfirm: (Int) -> Unit,
    onDismiss: () -> Unit
) {
    val presets = listOf(-1, 15, 30, 45, 60, 90, 120, 180, 240)
    var selected by remember {
        mutableIntStateOf(if (currentMin in presets) currentMin else presets[4])
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor   = CfDialogBg,
        title = {
            Text(
                text       = title,
                color      = CfTextPrimary,
                fontWeight = FontWeight.Bold,
                fontSize   = 16.sp
            )
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(text = "Choose a daily time limit:", color = CfTextSecond, fontSize = 13.sp)
                Spacer(modifier = Modifier.height(4.dp))
                presets.chunked(4).forEach { row ->
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        modifier              = Modifier.fillMaxWidth()
                    ) {
                        row.forEach { preset ->
                            val isSelected = selected == preset
                            Surface(
                                onClick  = { selected = preset },
                                shape    = RoundedCornerShape(8.dp),
                                color    = if (isSelected) CfPurple else CfPurpleLight,
                                modifier = Modifier.weight(1f)
                            ) {
                                Text(
                                    text       = if (preset == -1) "10s" else formatMinutes(preset),
                                    color      = if (isSelected) CfTextOnDark else CfIdleText,
                                    fontSize   = 12.sp,
                                    fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                                    textAlign  = TextAlign.Center,
                                    modifier   = Modifier.padding(vertical = 10.dp)
                                )
                            }
                        }
                        repeat(4 - row.size) { Spacer(modifier = Modifier.weight(1f)) }
                    }
                }
            }
        },
        confirmButton = {
            Button(
                onClick = { onConfirm(selected) },
                colors  = ButtonDefaults.buttonColors(containerColor = CfGreen)
            ) {
                Icon(
                    imageVector        = Icons.Default.CheckCircle,
                    contentDescription = null,
                    tint               = CfTextOnDark,
                    modifier           = Modifier.size(16.dp)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text("Set Limit", color = CfTextOnDark, fontWeight = FontWeight.Bold)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel", color = CfTextSecond)
            }
        }
    )
}