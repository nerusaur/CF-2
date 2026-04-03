package com.childfocus.ui

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.Settings
import android.text.TextUtils
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.outlined.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import com.childfocus.service.ChildFocusAccessibilityService
import com.childfocus.service.WebBlockerAccessibilityService

// ─────────────────────────────────────────────
//  Helper utilities
// ─────────────────────────────────────────────

fun hasOverlayPermission(context: Context): Boolean =
    Settings.canDrawOverlays(context)

fun hasAccessibilityPermission(context: Context): Boolean {
    val enabledServices = Settings.Secure.getString(
        context.contentResolver,
        Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES
    ) ?: return false

    // Drain the splitter into a plain List to avoid Kotlin overload ambiguity
    val splitter = TextUtils.SimpleStringSplitter(':').also { it.setString(enabledServices) }
    val enabledList = mutableListOf<String>()
    while (splitter.hasNext()) enabledList.add(splitter.next())

    val packageName = context.packageName
    val services = listOf(
        "$packageName/${ChildFocusAccessibilityService::class.java.name}",
        "$packageName/${WebBlockerAccessibilityService::class.java.name}"
    )
    return services.all { service ->
        enabledList.any { it.equals(service, ignoreCase = true) }
    }
}

fun openOverlaySettings(context: Context) {
    context.startActivity(
        Intent(
            Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
            Uri.parse("package:${context.packageName}")
        )
    )
}

fun openAccessibilitySettings(context: Context) {
    context.startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
}

// ─────────────────────────────────────────────
//  Screen
// ─────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(onBack: () -> Unit = {}) {
    val context        = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var overlayGranted       by remember { mutableStateOf(hasOverlayPermission(context)) }
    var accessibilityGranted by remember { mutableStateOf(hasAccessibilityPermission(context)) }

    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) {
                overlayGranted       = hasOverlayPermission(context)
                accessibilityGranted = hasAccessibilityPermission(context)
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Permissions & Settings", fontWeight = FontWeight.SemiBold) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text     = "Required Permissions",
                style    = MaterialTheme.typography.labelLarge,
                color    = MaterialTheme.colorScheme.primary,
                modifier = Modifier.padding(start = 4.dp, bottom = 4.dp)
            )

            PermissionCard(
                title         = "Display Over Other Apps",
                description   = "Allows ChildFocus to show the floating timer and block overlays on top of any app.",
                isGranted     = overlayGranted,
                onActionClick = { openOverlaySettings(context) }
            )

            PermissionCard(
                title         = "Accessibility Service",
                description   = "Lets ChildFocus monitor app usage and enforce web-blocking rules automatically.",
                isGranted     = accessibilityGranted,
                onActionClick = { openAccessibilitySettings(context) }
            )

            Spacer(modifier = Modifier.height(8.dp))

            if (overlayGranted && accessibilityGranted) {
                AllPermissionsGrantedBanner()
            } else {
                MissingPermissionsNote()
            }
        }
    }
}

// ─────────────────────────────────────────────
//  Permission Card
// ─────────────────────────────────────────────

@Composable
private fun PermissionCard(
    title: String,
    description: String,
    isGranted: Boolean,
    onActionClick: () -> Unit
) {
    val cardColor by animateColorAsState(
        targetValue   = if (isGranted) MaterialTheme.colorScheme.primaryContainer
        else MaterialTheme.colorScheme.errorContainer,
        animationSpec = tween(400),
        label         = "cardColor"
    )

    Card(
        shape    = RoundedCornerShape(16.dp),
        colors   = CardDefaults.cardColors(containerColor = cardColor),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier              = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment     = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            StatusDot(isGranted)

            Column(modifier = Modifier.weight(1f)) {
                Text(text = title, fontWeight = FontWeight.SemiBold, fontSize = 15.sp)
                Spacer(modifier = Modifier.height(3.dp))
                Text(
                    text       = description,
                    fontSize   = 13.sp,
                    color      = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 18.sp
                )
            }

            if (!isGranted) {
                Button(
                    onClick        = onActionClick,
                    shape          = RoundedCornerShape(10.dp),
                    contentPadding = PaddingValues(horizontal = 14.dp, vertical = 8.dp)
                ) {
                    Text("Enable", fontSize = 13.sp)
                }
            } else {
                Icon(
                    imageVector        = Icons.Default.CheckCircle,
                    contentDescription = "Granted",
                    tint               = MaterialTheme.colorScheme.primary,
                    modifier           = Modifier.size(28.dp)
                )
            }
        }
    }
}

@Composable
private fun StatusDot(isGranted: Boolean) {
    val color by animateColorAsState(
        targetValue   = if (isGranted) Color(0xFF4CAF50) else Color(0xFFF44336),
        animationSpec = tween(400),
        label         = "dotColor"
    )
    Box(
        modifier = Modifier
            .size(12.dp)
            .clip(CircleShape)
            .background(color)
    )
}

@Composable
private fun AllPermissionsGrantedBanner() {
    Card(
        shape    = RoundedCornerShape(12.dp),
        colors   = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.secondaryContainer
        ),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier              = Modifier.padding(14.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment     = Alignment.CenterVertically
        ) {
            Icon(Icons.Default.CheckCircle, null, tint = MaterialTheme.colorScheme.secondary)
            Text(
                text  = "All permissions granted — ChildFocus is fully active.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSecondaryContainer
            )
        }
    }
}

@Composable
private fun MissingPermissionsNote() {
    Card(
        shape    = RoundedCornerShape(12.dp),
        colors   = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        ),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier              = Modifier.padding(14.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment     = Alignment.Top
        ) {
            Icon(
                Icons.Outlined.Warning, null,
                tint     = MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(top = 2.dp)
            )
            Text(
                text       = "Some permissions are missing. Tap Enable to open system settings, grant the permission, then return here — the status updates automatically.",
                style      = MaterialTheme.typography.bodySmall,
                color      = MaterialTheme.colorScheme.onSurfaceVariant,
                lineHeight = 18.sp
            )
        }
    }
}