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
                        score >= 0.75f -> CfRed
                        score <= 0.35f -> CfGreen
                        else           -> CfAmber
                    },
                    trackColor  = CfBorder
                )
                Spacer(modifier = Modifier.height(4.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("Safe",    color = CfGreen,   fontSize = 11.sp, fontWeight = FontWeight.Medium)
                    Text("Neutral", color = CfAmber,   fontSize = 11.sp, fontWeight = FontWeight.Medium)
                    Text("Block",   color = CfRed,     fontSize = 11.sp, fontWeight = FontWeight.Medium)
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