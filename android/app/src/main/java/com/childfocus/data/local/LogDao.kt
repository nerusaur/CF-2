package com.childfocus.data.local

import androidx.room.*

@Dao
interface LogDao {
    @Insert
    suspend fun insertLog(log: LogEntity)

    @Query("SELECT * FROM logs WHERE videoId = :videoId")
    suspend fun getLogsByVideo(videoId: String): List<LogEntity>

    @Query("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 50")
    suspend fun getRecentLogs(): List<LogEntity>
}