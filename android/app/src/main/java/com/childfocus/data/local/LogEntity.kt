package com.childfocus.data.local

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "logs")
data class LogEntity(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val videoId: String,
    val userId: String,
    val action: String,   // allowed, blocked, blurred
    val timestamp: Long = System.currentTimeMillis(),
    val reasonDetails: String = ""
)