package com.childfocus.data.api

import retrofit2.http.Body
import retrofit2.http.POST
import com.google.gson.annotations.SerializedName

// ── Response models ───────────────────────────────────────────────────────────

data class ClassifyResponse(
    val video_id:    String  = "",
    val oir_label:   String  = "Neutral",
    val score_final: Float   = 0f,
    val cached:      Boolean = false,
    val action:      String  = "allow",
    val status:      String  = "success",
)

data class SegmentDetail(
    val segment_id:     String = "",
    val offset_seconds: Int    = 0,
    val length_seconds: Int    = 20,
    val fcr:            Float  = 0f,
    val csv:            Float  = 0f,
    val att:            Float  = 0f,
    val score_h:        Float  = 0f,
)

data class HeuristicDetails(
    val segments:    List<SegmentDetail> = emptyList(),
    val score_h_max: Float               = 0f,
    val thumbnail:   Float               = 0f,
)

data class NbDetails(
    // FIX: backend sends key "predicted", not "predicted_label"
    // Without @SerializedName, Gson looks for "predicted_label" in the JSON
    // and always returns the default empty string.
    @SerializedName("predicted")
    val predicted_label: String             = "",
    val confidence:      Float              = 0f,
    // NOTE: backend classify_full sends score_nb at the top level of the
    // response, not inside nb_details. probabilities is only present in
    // the NB-only fallback path. Both fields are safe to keep as defaults.
    val score_nb:        Float              = 0f,
    val probabilities:   Map<String, Float> = emptyMap(),
)

data class FullAnalysisResponse(
    val video_id:          String            = "",
    val video_title:       String            = "",
    val oir_label:         String            = "Neutral",
    val score_nb:          Float             = 0f,
    val score_h:           Float             = 0f,
    val score_final:       Float             = 0f,
    val cached:            Boolean           = false,
    val action:            String            = "allow",
    val status:            String            = "success",
    val runtime_seconds:   Float             = 0f,
    val nb_details:        NbDetails?        = null,
    val heuristic_details: HeuristicDetails? = null,
)

// ── API interface ─────────────────────────────────────────────────────────────

interface ChildFocusApi {
    // classifyFast body stays Map<String, String> — VideoRepository joins
    // tags as a space-separated string before building the map so the type
    // stays Map<String, String> throughout (see VideoRepository.classifyFast).
    @POST("classify_fast")
    suspend fun classifyFast(@Body body: Map<String, String>): ClassifyResponse

    @POST("classify_full")
    suspend fun classifyFull(@Body body: Map<String, String>): FullAnalysisResponse
}
