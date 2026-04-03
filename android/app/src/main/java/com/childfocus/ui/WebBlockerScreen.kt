package com.childfocus.ui

import android.accessibilityservice.AccessibilityServiceInfo
import android.content.Context
import android.content.Intent
import android.provider.Settings
import android.view.accessibility.AccessibilityManager
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.childfocus.service.WebBlockerManager
import com.childfocus.ui.theme.*
import kotlinx.coroutines.delay

// ─── Presets ─────────────────────────────────────────────────────────────────
private data class Preset(val label: String, val domains: List<String>)

private val PRESETS = listOf(
    Preset("🔞 Adult",    listOf("pornhub.com","xvideos.com","xnxx.com","onlyfans.com","redtube.com")),
    Preset("🎰 Gambling", listOf("bet365.com","pokerstars.com","casino.com","draftkings.com","fanduel.com")),
    Preset("⚔️ Violence", listOf("liveleak.com","bestgore.com","goregrish.com")),
    Preset("🎮 Gaming",   listOf("roblox.com","fortnite.com","miniclip.com","poki.com","crazygames.com")),
    Preset("📱 Social",   listOf("tiktok.com","instagram.com","snapchat.com","twitter.com","x.com")),
)

internal const val DEFAULT_PIN   = "1234"
internal const val PREFS_NAME    = "web_blocker_prefs"
internal const val PREFS_PIN_KEY = "parent_pin"

fun isWebBlockerServiceEnabled(context: Context): Boolean {
    val am      = context.getSystemService(Context.ACCESSIBILITY_SERVICE) as AccessibilityManager
    val enabled = am.getEnabledAccessibilityServiceList(AccessibilityServiceInfo.FEEDBACK_ALL_MASK)
    return enabled.any {
        it.resolveInfo.serviceInfo.packageName == context.packageName &&
                it.resolveInfo.serviceInfo.name.contains("WebBlocker", ignoreCase = true)
    }
}

@Composable
fun WebBlockerScreen() {
    WebBlockerDashboard(onLock = { SessionAuthManager.lock() })
}

// ─── PIN gate screen ──────────────────────────────────────────────────────────

@Composable
internal fun PinGateScreen(
    storedPin    : String,
    onPinCorrect : () -> Unit,
    subtitle     : String = "Enter your PIN to manage blocked sites",
) {
    val pinState      = remember { mutableStateOf("") }
    val hasErrorState = remember { mutableStateOf(false) }
    val offsetX       = remember { Animatable(0f) }

    val pin      = pinState.value
    val hasError = hasErrorState.value

    LaunchedEffect(hasError) {
        if (hasError) {
            repeat(4) {
                offsetX.animateTo( 10f, tween(50))
                offsetX.animateTo(-10f, tween(50))
            }
            offsetX.animateTo(0f, tween(50))
        }
    }

    fun submit() {
        if (pinState.value == storedPin) onPinCorrect()
        else { hasErrorState.value = true; pinState.value = "" }
    }

    val bgGradient = Brush.verticalGradient(colors = listOf(CfBgTop, CfBgBottom))

    Box(
        Modifier
            .fillMaxSize()
            .background(bgGradient)
            .navigationBarsPadding(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier
                .padding(32.dp)
                .offset(x = offsetX.value.dp)
        ) {
            Icon(Icons.Default.Lock, null, tint = CfPurple, modifier = Modifier.size(52.dp))
            Spacer(Modifier.height(16.dp))
            Text("Parent Access", fontSize = 22.sp, fontWeight = FontWeight.Bold, color = CfTextPrimary)
            Spacer(Modifier.height(6.dp))
            Text(subtitle, fontSize = 13.sp, color = CfTextSecond, textAlign = TextAlign.Center)
            Spacer(Modifier.height(32.dp))

            // PIN dots
            Row(horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                repeat(4) { i ->
                    Box(
                        Modifier
                            .size(16.dp)
                            .clip(RoundedCornerShape(50))
                            .background(
                                when {
                                    i < pin.length -> CfPinDotFilled
                                    hasError       -> CfRed.copy(alpha = 0.4f)
                                    else           -> CfPinDotEmpty
                                }
                            )
                    )
                }
            }
            Spacer(Modifier.height(8.dp))
            AnimatedVisibility(visible = hasError) {
                Text("Incorrect PIN — try again", color = CfRed, fontSize = 12.sp)
            }
            Spacer(Modifier.height(28.dp))

            // Numpad
            listOf(
                listOf("1","2","3"),
                listOf("4","5","6"),
                listOf("7","8","9"),
                listOf("" ,"0","⌫"),
            ).forEach { row ->
                Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                    row.forEach { key ->
                        Surface(
                            onClick = {
                                when (key) {
                                    "" -> Unit
                                    "⌫" -> {
                                        if (pinState.value.isNotEmpty()) {
                                            pinState.value = pinState.value.dropLast(1)
                                        }
                                        hasErrorState.value = false
                                    }
                                    else -> {
                                        if (pinState.value.length < 4) {
                                            pinState.value += key
                                            hasErrorState.value = false
                                            if (pinState.value.length == 4) submit()
                                        }
                                    }
                                }
                            },
                            shape    = RoundedCornerShape(50),
                            color    = if (key.isEmpty()) Color.Transparent else CfNumpadKey,
                            modifier = Modifier.size(72.dp)
                        ) {
                            Box(contentAlignment = Alignment.Center) {
                                if (key.isNotEmpty()) {
                                    Text(
                                        key,
                                        fontSize   = 22.sp,
                                        color      = CfTextPrimary,
                                        fontWeight = FontWeight.Medium
                                    )
                                }
                            }
                        }
                    }
                }
                Spacer(Modifier.height(12.dp))
            }
        }
    }
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

@Composable
internal fun WebBlockerDashboard(onLock: () -> Unit) {
    val context      = LocalContext.current
    val focusManager = LocalFocusManager.current
    val focusReq     = remember { FocusRequester() }

    // Explicit MutableState objects — no `by` delegation, safe inside LazyColumn lambdas
    val newDomainState     = remember { mutableStateOf("") }
    val blockedSitesState  = remember { mutableStateOf<List<String>>(WebBlockerManager.getBlockedSites().toList()) }
    val serviceActiveState = remember { mutableStateOf(isWebBlockerServiceEnabled(context)) }
    val showPinDialogState = remember { mutableStateOf(false) }

    // Read into plain vals — these are what item{} lambdas capture
    val newDomain     = newDomainState.value
    val blockedSites  = blockedSitesState.value
    val serviceActive = serviceActiveState.value
    val showPinDialog = showPinDialogState.value

    fun refresh() {
        blockedSitesState.value = WebBlockerManager.getBlockedSites().toList()
    }

    LaunchedEffect(Unit) {
        while (true) {
            serviceActiveState.value = isWebBlockerServiceEnabled(context)
            delay(2_000)
        }
    }

    if (showPinDialog) {
        ChangePinDialog(
            onDismiss = { showPinDialogState.value = false },
            onConfirm = { showPinDialogState.value = false }
        )
    }

    val bgGradient = Brush.verticalGradient(colors = listOf(CfBgTop, CfBgBottom))

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(bgGradient)
            .padding(horizontal = 20.dp, vertical = 20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // ── Header ─────────────────────────────────────────────────────────────
        item {
            Row(
                modifier          = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text       = "🌐  Web Blocker",
                        fontSize   = 22.sp,
                        fontWeight = FontWeight.Bold,
                        color      = CfTextPrimary
                    )
                    Text(
                        text     = "Block harmful websites for your child.",
                        color    = CfTextSecond,
                        fontSize = 13.sp
                    )
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    IconButton(onClick = { showPinDialogState.value = true }) {
                        Icon(Icons.Default.Settings, "Change PIN", tint = CfTextSecond)
                    }
                    IconButton(onClick = onLock) {
                        Icon(Icons.Default.Lock, "Lock", tint = CfTextSecond)
                    }
                }
            }
        }

        // ── Service status banner ──────────────────────────────────────────────
        item {
            Card(
                shape  = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(
                    containerColor = if (serviceActive) CfAllowedBg else CfAmberLight
                ),
                elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
                modifier  = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier          = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector        = if (serviceActive) Icons.Default.CheckCircle else Icons.Default.Warning,
                        contentDescription = null,
                        tint               = if (serviceActive) CfGreen else CfAmber,
                        modifier           = Modifier.size(20.dp)
                    )
                    Spacer(Modifier.width(10.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text       = if (serviceActive) "Service Active" else "Service Disabled",
                            color      = if (serviceActive) CfAllowedText else CfAnalyzingText,
                            fontWeight = FontWeight.SemiBold,
                            fontSize   = 13.sp
                        )
                        Text(
                            text     = if (serviceActive) "Blocking is running" else "Tap to enable in Settings",
                            color    = if (serviceActive) CfAllowedText.copy(alpha = .7f) else CfAnalyzingText.copy(alpha = .7f),
                            fontSize = 12.sp
                        )
                    }
                    if (!serviceActive) {
                        TextButton(onClick = {
                            context.startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
                        }) {
                            Text("Enable", color = CfPurple, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                        }
                    }
                }
            }
        }

        // ── Add domain row ─────────────────────────────────────────────────────
        item {
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment     = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value         = newDomain,
                    onValueChange = { newDomainState.value = it.lowercase().trim() },
                    placeholder   = { Text("e.g. example.com", color = CfTextHint, fontSize = 14.sp) },
                    singleLine    = true,
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Uri,
                        imeAction    = ImeAction.Done
                    ),
                    keyboardActions = KeyboardActions(onDone = {
                        if (newDomainState.value.isNotBlank()) {
                            WebBlockerManager.addSite(newDomainState.value.trim())
                            newDomainState.value = ""
                            focusManager.clearFocus()
                            refresh()
                        }
                    }),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor   = CfPurple,
                        unfocusedBorderColor = CfBorder,
                        focusedTextColor     = CfTextPrimary,
                        unfocusedTextColor   = CfTextPrimary,
                        cursorColor          = CfPurple,
                    ),
                    shape    = RoundedCornerShape(12.dp),
                    modifier = Modifier
                        .weight(1f)
                        .focusRequester(focusReq)
                )
                Button(
                    onClick = {
                        if (newDomainState.value.isNotBlank()) {
                            WebBlockerManager.addSite(newDomainState.value.trim())
                            newDomainState.value = ""
                            focusManager.clearFocus()
                            refresh()
                        }
                    },
                    colors   = ButtonDefaults.buttonColors(containerColor = CfGreen),
                    shape    = RoundedCornerShape(12.dp),
                    modifier = Modifier.height(56.dp)
                ) {
                    Icon(Icons.Default.Add, "Add", tint = CfTextOnDark)
                }
            }
        }

        // ── Quick presets ──────────────────────────────────────────────────────
        item {
            Text(
                "Quick Presets",
                color      = CfTextSecond,
                fontSize   = 12.sp,
                fontWeight = FontWeight.SemiBold,
                modifier   = Modifier.padding(start = 4.dp)
            )
            Spacer(Modifier.height(8.dp))
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                PRESETS.chunked(2).forEach { row ->
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        row.forEach { preset ->
                            PresetChip(
                                preset   = preset,
                                onClick  = {
                                    preset.domains.forEach { WebBlockerManager.addSite(it) }
                                    refresh()
                                },
                                modifier = Modifier.weight(1f)
                            )
                        }
                        if (row.size == 1) Spacer(Modifier.weight(1f))
                    }
                }
            }
        }

        // ── Blocked list header ────────────────────────────────────────────────
        item {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier          = Modifier.padding(start = 4.dp, top = 4.dp)
            ) {
                Text(
                    "Blocked Sites",
                    color      = CfTextSecond,
                    fontSize   = 12.sp,
                    fontWeight = FontWeight.SemiBold
                )
                Spacer(Modifier.width(8.dp))
                Surface(
                    shape = RoundedCornerShape(20.dp),
                    color = CfPurpleLight
                ) {
                    Text(
                        text       = blockedSites.size.toString(),
                        color      = CfPurple,
                        fontSize   = 11.sp,
                        fontWeight = FontWeight.Bold,
                        modifier   = Modifier.padding(horizontal = 8.dp, vertical = 2.dp)
                    )
                }
                if (blockedSites.isNotEmpty()) {
                    Spacer(Modifier.weight(1f))
                    TextButton(onClick = {
                        WebBlockerManager.clearAll()
                        refresh()
                    }) {
                        Text("Clear All", color = CfRed, fontSize = 12.sp)
                    }
                }
            }
        }

        // ── Empty state ────────────────────────────────────────────────────────
        item {
            if (blockedSites.isEmpty()) {
                Box(
                    Modifier
                        .fillMaxWidth()
                        .padding(vertical = 32.dp),
                    Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Icon(
                            Icons.Default.CheckCircle,
                            null,
                            tint     = CfTextHint,
                            modifier = Modifier.size(40.dp)
                        )
                        Spacer(Modifier.height(8.dp))
                        Text("No sites blocked yet", color = CfTextSecond, fontSize = 14.sp)
                        Text(
                            "Add a domain above or use a quick preset",
                            color    = CfTextHint,
                            fontSize = 12.sp
                        )
                    }
                }
            }
        }

        // ── Blocked site rows ──────────────────────────────────────────────────
        items(items = blockedSites, key = { it }) { site ->
            BlockedSiteRow(
                site     = site,
                onRemove = {
                    WebBlockerManager.removeSite(site)
                    refresh()
                }
            )
        }
    }
}

// ─── Change PIN dialog ────────────────────────────────────────────────────────

@Composable
private fun ChangePinDialog(onDismiss: () -> Unit, onConfirm: () -> Unit) {
    val context  = LocalContext.current
    val prefs    = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    val currentState  = remember { mutableStateOf("") }
    val newPinState   = remember { mutableStateOf("") }
    val confirmState  = remember { mutableStateOf("") }
    val errorMsgState = remember { mutableStateOf<String?>(null) }

    AlertDialog(
        onDismissRequest  = onDismiss,
        containerColor    = CfDialogBg,
        titleContentColor = CfTextPrimary,
        title = { Text("Change Parent PIN", fontWeight = FontWeight.Bold) },
        text  = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                PinTextField("Current PIN", currentState.value) {
                    currentState.value = it.filter(Char::isDigit).take(4)
                }
                PinTextField("New PIN (4 digits)", newPinState.value) {
                    newPinState.value = it.filter(Char::isDigit).take(4)
                }
                PinTextField("Confirm New PIN", confirmState.value) {
                    confirmState.value = it.filter(Char::isDigit).take(4)
                }
                errorMsgState.value?.let {
                    Text(it, color = CfRed, fontSize = 12.sp)
                }
            }
        },
        confirmButton = {
            TextButton(onClick = {
                val stored = prefs.getString(PREFS_PIN_KEY, DEFAULT_PIN) ?: DEFAULT_PIN
                when {
                    currentState.value != stored            -> errorMsgState.value = "Current PIN is incorrect."
                    newPinState.value.length != 4           -> errorMsgState.value = "New PIN must be exactly 4 digits."
                    newPinState.value != confirmState.value -> errorMsgState.value = "PINs do not match."
                    else -> {
                        prefs.edit().putString(PREFS_PIN_KEY, newPinState.value).apply()
                        onConfirm()
                    }
                }
            }) {
                Text("Save", color = CfGreen, fontWeight = FontWeight.Bold)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel", color = CfTextSecond)
            }
        }
    )
}

@Composable
private fun PinTextField(label: String, value: String, onValueChange: (String) -> Unit) {
    OutlinedTextField(
        value                = value,
        onValueChange        = onValueChange,
        label                = { Text(label, color = CfTextHint, fontSize = 12.sp) },
        singleLine           = true,
        visualTransformation = PasswordVisualTransformation(),
        keyboardOptions      = KeyboardOptions(keyboardType = KeyboardType.NumberPassword),
        colors               = OutlinedTextFieldDefaults.colors(
            focusedBorderColor   = CfPurple,
            unfocusedBorderColor = CfBorder,
            focusedTextColor     = CfTextPrimary,
            unfocusedTextColor   = CfTextPrimary,
            cursorColor          = CfPurple,
        ),
        modifier = Modifier.fillMaxWidth()
    )
}

// ─── Preset chip ──────────────────────────────────────────────────────────────

@Composable
private fun PresetChip(preset: Preset, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Surface(
        onClick  = onClick,
        shape    = RoundedCornerShape(12.dp),
        color    = CfSurface,
        border   = BorderStroke(1.dp, CfBorder),
        modifier = modifier
    ) {
        Row(
            modifier          = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(preset.label.take(2), fontSize = 16.sp)
            Spacer(Modifier.width(6.dp))
            Text(
                preset.label.drop(2).trim(),
                color      = CfTextPrimary,
                fontSize   = 13.sp,
                fontWeight = FontWeight.Medium,
                maxLines   = 1,
                overflow   = TextOverflow.Ellipsis
            )
        }
    }
}

// ─── Blocked site row ─────────────────────────────────────────────────────────

@Composable
private fun BlockedSiteRow(site: String, onRemove: () -> Unit) {
    Card(
        shape    = RoundedCornerShape(12.dp),
        colors   = CardDefaults.cardColors(containerColor = CfSurface),
        border   = BorderStroke(1.dp, CfBorder),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier          = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                Icons.Default.Block,
                null,
                tint     = CfRed.copy(alpha = 0.7f),
                modifier = Modifier.size(18.dp)
            )
            Spacer(Modifier.width(12.dp))
            Text(
                site,
                color    = CfTextPrimary,
                fontSize = 14.sp,
                modifier = Modifier.weight(1f),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            IconButton(onClick = onRemove, modifier = Modifier.size(32.dp)) {
                Icon(
                    Icons.Default.Close,
                    "Remove $site",
                    tint     = CfTextHint,
                    modifier = Modifier.size(18.dp)
                )
            }
        }
    }
}