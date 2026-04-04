package com.childfocus.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// ── Palette ──────────────────────────────────────────────────────────────────
private val BgTop        = Color(0xFFF0F4FF)
private val BgBottom     = Color(0xFFE8F5E9)
private val AccentGreen  = Color(0xFF43A047)
private val AccentPurple = Color(0xFF7C4DFF)
private val PillBg       = Color(0xFFFFFFFF)
private val TextPrimary  = Color(0xFF1A237E)
private val TextSecond   = Color(0xFF546E7A)
private val BtnText      = Color(0xFFFFFFFF)
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun LandingScreen(
    isWaiting: Boolean = false,
    onTurnOn: () -> Unit,
    onSettingsClick: () -> Unit = {}          // ← NEW: navigates to SettingsScreen
) {
    val bgGradient = Brush.verticalGradient(
        colors = listOf(BgTop, BgBottom)
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(bgGradient)
            .navigationBarsPadding()
    ) {

        // ── Settings gear icon — top-right corner ─────────────────────────────
        IconButton(
            onClick  = onSettingsClick,
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(top = 8.dp, end = 8.dp)
        ) {
            Icon(
                imageVector        = Icons.Default.Settings,
                contentDescription = "Permissions & Settings",
                tint               = TextSecond
            )
        }

        // ── Main content ──────────────────────────────────────────────────────
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 28.dp, vertical = 32.dp)
                .align(Alignment.Center)
        ) {

            // ── Logo / title ─────────────────────────────────────────────────
            Text(
                text         = "ChildFocus",
                fontSize     = 38.sp,
                fontWeight   = FontWeight.ExtraBold,
                color        = AccentPurple,
                letterSpacing = 1.sp
            )

            Text(
                text          = "A CHILD'S FOCUS, IN SAFE HANDS.",
                fontSize      = 11.sp,
                fontWeight    = FontWeight.Medium,
                color         = TextSecond,
                letterSpacing = 2.5.sp,
                textAlign     = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(8.dp))

            // ── Feature pills ─────────────────────────────────────────────────
            listOf(
                "🎬  AI-Powered Overstimulation Detection",
                "🌐  Website Blocking",
                "⏱️  Screen-Time Control",
                "🔒  Content Restrictions"
            ).forEach { feature ->
                Card(
                    shape     = RoundedCornerShape(50),
                    colors    = CardDefaults.cardColors(containerColor = PillBg),
                    elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
                    modifier  = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text       = feature,
                        color      = TextPrimary,
                        fontSize   = 14.sp,
                        fontWeight = FontWeight.Medium,
                        modifier   = Modifier.padding(horizontal = 20.dp, vertical = 14.dp),
                        textAlign  = TextAlign.Center
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // ── CTA Button ───────────────────────────────────────────────────
            Button(
                onClick   = onTurnOn,
                enabled   = !isWaiting,
                modifier  = Modifier
                    .fillMaxWidth()
                    .height(58.dp),
                shape     = RoundedCornerShape(29.dp),
                colors    = ButtonDefaults.buttonColors(
                    containerColor         = AccentGreen,
                    disabledContainerColor = AccentGreen.copy(alpha = 0.5f)
                ),
                elevation = ButtonDefaults.buttonElevation(defaultElevation = 4.dp)
            ) {
                if (isWaiting) {
                    CircularProgressIndicator(
                        modifier    = Modifier.size(22.dp),
                        color       = BtnText,
                        strokeWidth = 2.5.dp
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Text(
                        text          = "WAITING FOR SERVICE…",
                        fontWeight    = FontWeight.Bold,
                        fontSize      = 15.sp,
                        color         = BtnText,
                        letterSpacing = 0.8.sp
                    )
                } else {
                    Text(
                        text          = "TURN ON SAFETY MODE",
                        fontWeight    = FontWeight.Bold,
                        fontSize      = 16.sp,
                        color         = BtnText,
                        letterSpacing = 0.8.sp
                    )
                }
            }

            Text(
                text      = "We're here to support you\nin protecting children",
                color     = TextSecond,
                fontSize  = 13.sp,
                textAlign = TextAlign.Center
            )
        }
    }
}