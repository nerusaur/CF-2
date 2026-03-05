package com.childfocus.data.local

import androidx.room.Database
import androidx.room.RoomDatabase

@Database(
    entities = [VideoEntity::class, SegmentEntity::class, UserEntity::class, LogEntity::class],
    version = 1,
    exportSchema = false
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun videoDao(): VideoDao
    abstract fun logDao(): LogDao
}