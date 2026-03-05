package com.childfocus.data.local

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.PrimaryKey

@Entity(
    tableName = "segments",
    foreignKeys = [ForeignKey(
        entity = VideoEntity::class,
        parentColumns = ["videoId"],
        childColumns = ["videoId"],
        onDelete = ForeignKey.CASCADE
    )]
)
data class SegmentEntity(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val videoId: String,
    val offsetSeconds: Int,
    val lengthSeconds: Int,
    val fcr: Float,     // Frame-Change Rate
    val csv: Float,     // Color Saturation Variance
    val att: Float,     // Audio Tempo Transitions
    val score: Float
)