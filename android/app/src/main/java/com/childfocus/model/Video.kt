package com.childfocus.model

/**
 * Video domain model (used in UI layer).
 * Maps from VideoEntity (Room) or API response.
 */
data class Video(
    val videoId:     String,
    val label:       String  = "",   // Educational / Neutral / Overstimulating
    val finalScore:  Float   = 0f,
    val lastChecked: Long    = 0L,
    val checkedBy:   String  = "",
)