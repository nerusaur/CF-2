package com.childfocus.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val LightColors = lightColorScheme(
    primary          = CfPurple,
    onPrimary        = CfTextOnDark,
    primaryContainer = CfPurpleLight,
    onPrimaryContainer = CfTextPrimary,

    secondary        = CfGreen,
    onSecondary      = CfTextOnDark,
    secondaryContainer = CfGreenLight,
    onSecondaryContainer = CfAllowedText,

    tertiary         = CfAmber,
    onTertiary       = CfTextOnDark,
    tertiaryContainer = CfAmberLight,
    onTertiaryContainer = CfAnalyzingText,

    background       = CfBgTop,
    onBackground     = CfTextPrimary,

    surface          = CfSurface,
    onSurface        = CfTextPrimary,
    surfaceVariant   = CfSurfaceAlt,
    onSurfaceVariant = CfTextSecond,

    error            = CfRed,
    onError          = CfTextOnDark,
    errorContainer   = CfRedLight,
    onErrorContainer = CfBlockedText,

    outline          = CfBorder,
)

@Composable
fun ChildFocusTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = LightColors,
        typography  = Typography,
        content     = content
    )
}