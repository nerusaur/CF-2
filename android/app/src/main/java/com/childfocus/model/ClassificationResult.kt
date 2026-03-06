package com.childfocus.model

/**
 * Full classification result from /classify_full endpoint.
 * Maps to hybrid_fusion.py classify_full() response.
 */
data class ClassificationResult(
    val videoId:     String,
    val videoTitle:  String  = "",

    // Individual scores
    val scoreNb:     Float   = 0f,
    val scoreH:      Float   = 0f,
    val scoreFinal:  Float   = 0f,

    // OIR classification
    val oirLabel:    String  = "",   // Educational / Neutral / Overstimulating
    val action:      String  = "",   // block / allow

    // Supporting details
    val nbDetails:   NbDetails?        = null,
    val hDetails:    HeuristicDetails? = null,

    val status:          String = "",
    val runtimeSeconds:  Float  = 0f,
)

data class NbDetails(
    val label:         String              = "",
    val confidence:    Float               = 0f,
    val probabilities: Map<String, Float>  = emptyMap(),
)

data class HeuristicDetails(
    val segments:      List<Segment> = emptyList(),
    val thumbnail:     Float         = 0f,
    val videoDuration: Float         = 0f,
    val runtime:       Float         = 0f,
)

/**
 * Fast classification result from /classify_fast endpoint.
 */
data class FastClassificationResult(
    val videoId:          String = "",
    val scoreNb:          Float  = 0f,
    val nbLabel:          String = "",
    val preliminaryLabel: String = "",
    val action:           String = "",
    val status:           String = "",
    val metadata:         VideoMetadata? = null,
)

data class VideoMetadata(
    val title:        String = "",
    val channel:      String = "",
    val thumbnailUrl: String = "",
    val duration:     String = "",
    val viewCount:    Long   = 0L,
)