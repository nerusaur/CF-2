package com.childfocus.data.local

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "videos")
data class VideoEntity(
    @PrimaryKey val videoId: String,
    val label: String = "",           // Educational, Neutral, Overstimulating
    val finalScore: Float = 0f,
    val lastChecked: Long = System.currentTimeMillis(),
    val checkedBy: String = ""
)