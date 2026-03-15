package com.childfocus.viewmodel

import android.app.Application
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import androidx.lifecycle.AndroidViewModel
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

sealed class ClassifyState {
    object Idle : ClassifyState()
    data class Analyzing(val videoId: String) : ClassifyState()
    data class Allowed(
        val videoId: String,
        val label:   String,
        val score:   Float,
        val cached:  Boolean = false,
    ) : ClassifyState()
    data class Blocked(
        val videoId: String,
        val label:   String,
        val score:   Float,
        val cached:  Boolean = false,
    ) : ClassifyState()
    data class Error(val videoId: String) : ClassifyState()
}

/**
 * SafetyViewModel
 *
 * FIX: Replaced Context.RECEIVER_NOT_EXPORTED (API 33+ only) with
 * LocalBroadcastManager which works on API 21+.
 * The old code crashed silently on API 30 — receiver was never registered,
 * so the UI never updated regardless of what the service sent.
 */
class SafetyViewModel(application: Application) : AndroidViewModel(application) {

    private val _safetyModeOn  = MutableStateFlow(false)
    val safetyModeOn: StateFlow<Boolean> = _safetyModeOn

    private val _classifyState = MutableStateFlow<ClassifyState>(ClassifyState.Idle)
    val classifyState: StateFlow<ClassifyState> = _classifyState

    private val localBroadcast = LocalBroadcastManager.getInstance(application)

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(ctx: Context?, intent: Intent?) {
            intent ?: return
            val videoId = intent.getStringExtra("video_id")    ?: return
            val label   = intent.getStringExtra("oir_label")   ?: return
            val score   = intent.getFloatExtra("score_final", 0.5f)
            val cached  = intent.getBooleanExtra("cached", false)

            _classifyState.value = when (label) {
                "Analyzing"       -> ClassifyState.Analyzing(videoId)
                "Overstimulating" -> ClassifyState.Blocked(videoId, label, score, cached)
                "Error"           -> ClassifyState.Error(videoId)
                else              -> ClassifyState.Allowed(videoId, label, score, cached)
            }
        }
    }

    init {
        localBroadcast.registerReceiver(
            receiver,
            IntentFilter("com.childfocus.CLASSIFICATION_RESULT")
        )
    }

    fun toggleSafetyMode() {
        _safetyModeOn.value = !_safetyModeOn.value
        if (!_safetyModeOn.value) {
            _classifyState.value = ClassifyState.Idle
        }
    }

    fun dismissBlock() {
        _classifyState.value = ClassifyState.Idle
    }

    override fun onCleared() {
        super.onCleared()
        localBroadcast.unregisterReceiver(receiver)
    }
}
