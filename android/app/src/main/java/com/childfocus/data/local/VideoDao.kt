package com.childfocus.data.local

import androidx.room.*

@Dao
interface VideoDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertVideo(video: VideoEntity)

    @Query("SELECT * FROM videos WHERE videoId = :id")
    suspend fun getVideoById(id: String): VideoEntity?

    @Query("SELECT * FROM videos ORDER BY lastChecked DESC")
    suspend fun getAllVideos(): List<VideoEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertSegment(segment: SegmentEntity)

    @Query("SELECT * FROM segments WHERE videoId = :videoId")
    suspend fun getSegmentsByVideo(videoId: String): List<SegmentEntity>
}