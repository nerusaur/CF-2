package com.childfocus.model

/**
 * Video segment data model.
 * Represents one analyzed 20-second segment from frame_sampler.py.
 */
data class Segment(
    val segmentId:     String = "",   // S1, S2, S3
    val offsetSeconds: Int    = 0,
    val lengthSeconds: Int    = 0,
    val fcr:           Float  = 0f,   // Frame-Change Rate
    val csv:           Float  = 0f,   // Color Saturation Variance
    val att:           Float  = 0f,   // Audio Tempo Transitions
    val scoreH:        Float  = 0f,   // Heuristic score for this segment
)