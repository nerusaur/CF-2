package com.childfocus.ui

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.childfocus.viewmodel.SafetyViewModel

/**
 * LandingScreen
 *
 * Entry point of the app.
 * - When Safety Mode is OFF: shows the landing/home screen
 * - When Safety Mode is ON: transitions into SafetyModeScreen
 *
 * Uses SafetyViewModel so the toggle state persists across recompositions
 * and is accessible to SafetyModeScreen via the shared ViewModel.
 */
@Composable
fun LandingScreen(viewModel: SafetyViewModel) {
    val safetyModeOn by viewModel.safetyModeOn.collectAsState()

    AnimatedContent(
        targetState = safetyModeOn,
        transitionSpec = { fadeIn() togetherWith fadeOut() },
        label = "safety_mode_transition"
    ) { isOn ->
        if (isOn) {
            SafetyModeScreen(viewModel = viewModel)
        } else {
            HomeContent(
                onEnableSafetyMode = { viewModel.toggleSafetyMode() }
            )
        }
    }
}

@Composable
private fun HomeContent(onEnableSafetyMode: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        // ── Branding ─────────────────────────────────────────────────────────
        Text(
            text       = "ChildFocus",
            fontSize   = 32.sp,
            fontWeight = FontWeight.Bold,
            color      = MaterialTheme.colorScheme.primary,
        )

        Spacer(Modifier.height(8.dp))

        Text(
            text      = "A CHILD'S FOCUS, IN SAFE HANDS.",
            fontSize  = 12.sp,
            color     = Color.Gray,
            textAlign = TextAlign.Center,
        )

        Spacer(Modifier.height(48.dp))

        // ── Headline ─────────────────────────────────────────────────────────
        Text(
            text       = "Protect what matters most.",
            fontSize   = 18.sp,
            fontWeight = FontWeight.Medium,
            textAlign  = TextAlign.Center,
        )

        Spacer(Modifier.height(16.dp))

        Text(
            text      = "AI-powered analysis that detects\noverstimulating video content for children.",
            fontSize  = 14.sp,
            color     = Color.Gray,
            textAlign = TextAlign.Center,
        )

        Spacer(Modifier.height(48.dp))

        // ── Safety Mode Toggle ────────────────────────────────────────────────
        Button(
            onClick  = onEnableSafetyMode,
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = MaterialTheme.colorScheme.primary,
            )
        ) {
            Text(
                text       = "TURN ON SAFETY MODE",
                fontWeight = FontWeight.Bold,
            )
        }

        Spacer(Modifier.height(32.dp))

        // ── Feature chips ─────────────────────────────────────────────────────
        Row(
            modifier              = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly,
        ) {
            FeatureChip("⏱ Screen-time")
            FeatureChip("🌐 Web Blocking")
            FeatureChip("🚫 Content Filter")
        }
    }
}

@Composable
fun FeatureChip(label: String) {
    Surface(
        shape = MaterialTheme.shapes.small,
        color = MaterialTheme.colorScheme.secondaryContainer,
    ) {
        Text(
            text     = label,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
            fontSize = 12.sp,
        )
    }
}