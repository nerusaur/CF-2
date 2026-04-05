package com.childfocus.data

import com.childfocus.data.api.ChildFocusApi
import com.childfocus.data.api.ClassifyResponse
import com.childfocus.data.api.FullAnalysisResponse
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

/**
 * Single source of truth for all backend calls via Retrofit.
 *
 * ⚠ Keep FLASK_BASE_URL in sync with ChildFocusAccessibilityService.BASE_URL.
 *   Both must point to the same host:port.
 *
 *   Emulator  → "http://10.0.2.2:5000/"
 *   Physical  → "http://192.168.1.23:5000/"  ← change to your PC's WiFi IP
 */
class VideoRepository {

    companion object {
        private const val FLASK_BASE_URL = "http://192.168.1.23:5000/"
    }

    private val api: ChildFocusApi by lazy {
        val client = OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)   // classify_full can take ~60-90 s
            .writeTimeout(30, TimeUnit.SECONDS)
            .build()

        Retrofit.Builder()
            .baseUrl(FLASK_BASE_URL)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(ChildFocusApi::class.java)
    }

    /**
     * Quick metadata-only classification (title / tags / description).
     * Returns near-instantly — good for a pre-check before calling classifyFull.
     *
     * Tags are now included in the request body so the backend's build_nb_text()
     * receives the same inputs used during training (title × 3 + tags + description[:300]).
     * This makes Score_NB fully deterministic: same video → same label, every time.
     */
    suspend fun classifyFast(
        title: String,
        tags: List<String> = emptyList(),
        description: String = ""
    ): ClassifyResponse = api.classifyFast(
        mapOf(
            "title"       to title,
            "description" to description,
            "tags"        to tags,          // ← tags now sent to match training formula
        )
    )

    /**
     * Full hybrid classification (NB + heuristic audiovisual analysis).
     * May take 30-120 s depending on video length and backend load.
     */
    suspend fun classifyFull(
        videoUrl: String,
        thumbnailUrl: String = "",
        hintTitle: String    = "",
    ): FullAnalysisResponse = api.classifyFull(
        mapOf(
            "video_url"     to videoUrl,
            "thumbnail_url" to thumbnailUrl,
            "hint_title"    to hintTitle,
        )
    )
}
