interface ChildFocusApi {
    @POST("classify_fast")
    suspend fun classifyFast(@Body body: Map<String, String>): ClassifyResponse

    @POST("classify_full")
    suspend fun classifyFull(@Body body: Map<String, String>): FullAnalysisResponse
}