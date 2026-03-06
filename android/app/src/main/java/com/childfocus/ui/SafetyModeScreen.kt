package com.childfocus.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.childfocus.model.ClassificationResult
import com.childfocus.viewmodel.SafetyViewModel

/**
 * SafetyModeScreen
 *
 * Shown when Safety Mode is ON.
 * Allows parents to enter a YouTube URL and analyze it.
 * Displays OIR result with color-coded label and action.
 */
@Composable
fun SafetyModeScreen(viewModel: SafetyViewModel) {
    val isLoading   by viewModel.isLoading.collectAsState()
    val result      by viewModel.result.collectAsState()
    val error       by viewModel.error.collectAsState()
    val videoUrl    by viewModel.videoUrl.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp)
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        // ── Header ──────────────────────────────────────────────────────────
        Text(
            text       = "🛡️ Safety Mode Active",
            fontSize   = 22.sp,
            fontWeight = FontWeight.Bold,
            color      = Color(0xFF4CAF50),
        )
        Spacer(Modifier.height(4.dp))
        Text(
            text     = "Enter a YouTube URL to analyze",
            fontSize = 13.sp,
            color    = Color.Gray,
        )

        Spacer(Modifier.height(24.dp))

        // ── URL Input ────────────────────────────────────────────────────────
        OutlinedTextField(
            value         = videoUrl,
            onValueChange = { viewModel.setVideoUrl(it) },
            label         = { Text("YouTube URL or Video ID") },
            placeholder   = { Text("https://youtube.com/watch?v=...") },
            modifier      = Modifier.fillMaxWidth(),
            singleLine    = true,
        )

        Spacer(Modifier.height(12.dp))

        // ── Analyze Button ───────────────────────────────────────────────────
        Button(
            onClick  = { viewModel.analyzeVideo(videoUrl) },
            enabled  = !isLoading && videoUrl.isNotBlank(),
            modifier = Modifier
                .fillMaxWidth()
                .height(52.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = MaterialTheme.colorScheme.primary
            )
        ) {
            if (isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.size(20.dp),
                    color    = Color.White,
                    strokeWidth = 2.dp,
                )
                Spacer(Modifier.width(8.dp))
                Text("Analyzing... (may take up to 60s)")
            } else {
                Text("ANALYZE VIDEO", fontWeight = FontWeight.Bold)
            }
        }

        // ── Error ────────────────────────────────────────────────────────────
        error?.let {
            Spacer(Modifier.height(12.dp))
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFEBEE)),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(
                    text     = "⚠ $it",
                    modifier = Modifier.padding(12.dp),
                    color    = Color(0xFFC62828),
                    fontSize = 13.sp,
                )
            }
        }

        // ── Result ───────────────────────────────────────────────────────────
        result?.let { r ->
            Spacer(Modifier.height(24.dp))
            ClassificationResultCard(result = r)
        }
    }
}


@Composable
fun ClassificationResultCard(result: ClassificationResult) {
    val (bgColor, emoji) = when (result.oirLabel) {
        "Overstimulating" -> Color(0xFFFFEBEE) to "🚫"
        "Neutral"         -> Color(0xFFFFF9C4) to "⚠️"
        "Educational"     -> Color(0xFFE8F5E9) to "✅"
        else              -> Color(0xFFF5F5F5) to "ℹ️"
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape    = RoundedCornerShape(12.dp),
        colors   = CardDefaults.cardColors(containerColor = bgColor),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(text = emoji, fontSize = 40.sp)
            Spacer(Modifier.height(8.dp))

            Text(
                text       = result.oirLabel,
                fontSize   = 24.sp,
                fontWeight = FontWeight.Bold,
                color      = when (result.oirLabel) {
                    "Overstimulating" -> Color(0xFFC62828)
                    "Neutral"         -> Color(0xFFF57F17)
                    else              -> Color(0xFF2E7D32)
                }
            )

            Spacer(Modifier.height(4.dp))
            Text(
                text     = "Action: ${result.action.uppercase()}",
                fontSize = 13.sp,
                color    = Color.Gray,
            )

            Spacer(Modifier.height(16.dp))
            Divider()
            Spacer(Modifier.height(12.dp))

            // Score breakdown
            ScoreRow("Final OIR Score",     result.scoreFinal)
            ScoreRow("NB Metadata Score",   result.scoreNb)
            ScoreRow("Heuristic Score",     result.scoreH)

            result.hDetails?.let { h ->
                Spacer(Modifier.height(8.dp))
                Text(
                    text     = "Segments analyzed: ${h.segments.size}",
                    fontSize = 12.sp,
                    color    = Color.Gray,
                )
                Text(
                    text     = "Analysis time: ${result.runtimeSeconds}s",
                    fontSize = 12.sp,
                    color    = Color.Gray,
                )
            }

            if (result.videoTitle.isNotEmpty()) {
                Spacer(Modifier.height(8.dp))
                Text(
                    text      = result.videoTitle,
                    fontSize  = 12.sp,
                    color     = Color.DarkGray,
                    textAlign = TextAlign.Center,
                )
            }
        }
    }
}


@Composable
fun ScoreRow(label: String, score: Float) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(text = label, fontSize = 13.sp, color = Color.DarkGray)
        Text(
            text       = String.format("%.3f", score),
            fontSize   = 13.sp,
            fontWeight = FontWeight.Medium,
        )
    }
}