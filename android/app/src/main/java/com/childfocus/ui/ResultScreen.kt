package com.childfocus.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
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
import com.childfocus.ui.theme.*

// ── Backend thresholds — MUST stay in sync with classify.py ──────────────────
// classify.py: THRESHOLD_BLOCK = 0.20, THRESHOLD_ALLOW = 0.18
// The score bar color must reflect the same boundaries used by the backend
// to decide the label, so the visual feedback matches what was actually decided.
private const val SCORE_BLOCK = 0.20f   // >= 0.20 → Overstimulating (Red)
private const val SCORE_ALLOW = 0.18f   // <= 0.18 → Educational      (Green)
// 0.18 < score < 0.20 → Neutral (Amber)

@Composable
fun ResultScreen(
    videoId: String,
    label: String,
    score: Float,
    cached: Boolean,
    onBack: () -> Unit
) {
    // State-driven colors using the unified palette
    val (cardBg, accentColor, emoji) = when (label) {
        "Overstimulating" -> Triple(CfRedLight,   CfRed,    "⛔")
        "Educational"     -> Triple(CfGreenLight, CfGreen,  "📚")
        else              -> Triple(CfPurpleLight, CfPurple, "✅")
    }

    val bgGradient = Brush.verticalGradient(colors = listOf(CfBgTop, CfBgBottom))

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(bgGradient)
            .navigationBarsPadding(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(32.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {

            // Result label
            Text(
                text       = "$emoji  $label",
                fontSize   = 26.sp,
                fontWeight = FontWeight.ExtraBold,
                color      = accentColor
            )

            // Detail card
            Card(
                shape    = RoundedCornerShape(16.dp),
                colors   = CardDefaults.cardColors(containerColor = cardBg),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    ResultRow("Video ID", videoId.take(16) + if (videoId.length > 16) "…" else "")
                    ResultRow("OIR Score", "%.4f".format(score))
                    ResultRow("Source", if (cached) "Cache (instant)" else "Live classification")
                }
            }

            // Score bar
            // FIX: bar color thresholds now match backend classify.py exactly.
            // Old values (0.75 / 0.35) were from the original thesis config —
            // the current backend uses 0.20 (block) / 0.18 (allow).
            // A video blocked by the backend (score >= 0.20) now shows RED,
            // not misleadingly green or amber.
            Column(modifier = Modifier.fillMaxWidth()) {
                Text(
                    text     = "Overstimulation Index",
                    color    = CfTextSecond,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.SemiBold
                )
                Spacer(modifier = Modifier.height(6.dp))
                LinearProgressIndicator(
                    progress    = { score.coerceIn(0f, 1f) },
                    modifier    = Modifier.fillMaxWidth().height(10.dp),
                    color       = when {
                        score >= SCORE_BLOCK -> CfRed    // Overstimulating (>= 0.20)
                        score <= SCORE_ALLOW -> CfGreen  // Educational     (<= 0.18)
                        else                 -> CfAmber  // Neutral         (0.18–0.20)
                    },
                    trackColor  = CfBorder
                )
                Spacer(modifier = Modifier.height(4.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("Safe (≤0.18)",    color = CfGreen, fontSize = 11.sp, fontWeight = FontWeight.Medium)
                    Text("Neutral",         color = CfAmber, fontSize = 11.sp, fontWeight = FontWeight.Medium)
                    Text("Block (≥0.20)",   color = CfRed,   fontSize = 11.sp, fontWeight = FontWeight.Medium)
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Button(
                onClick  = onBack,
                modifier = Modifier.fillMaxWidth().height(54.dp),
                shape    = RoundedCornerShape(27.dp),
                colors   = ButtonDefaults.buttonColors(containerColor = CfGreen),
                elevation = ButtonDefaults.buttonElevation(defaultElevation = 4.dp)
            ) {
                Text(
                    text       = "Back",
                    color      = CfTextOnDark,
                    fontWeight = FontWeight.Bold,
                    fontSize   = 16.sp
                )
            }
        }
    }
}

@Composable
private fun ResultRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(label, color = CfTextSecond, fontSize = 13.sp)
        Text(
            value,
            color      = CfTextPrimary,
            fontSize   = 13.sp,
            fontWeight = FontWeight.SemiBold,
            textAlign  = TextAlign.End,
            modifier   = Modifier.weight(1f, fill = false)
        )
    }
}
