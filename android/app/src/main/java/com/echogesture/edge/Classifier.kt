package com.echogesture.edge

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.SystemClock
import org.json.JSONObject
import java.nio.FloatBuffer
import kotlin.math.exp

/** Port of echogesture/inference.py Predictor.classify (INT8 ONNX, temperature-scaled softmax). */
class Classifier(context: Context) : AutoCloseable {
    data class Result(
        val prediction: String,
        val confidence: Float,
        val top5: List<Pair<String, Float>>,
        val classifierMs: Double,
    )

    val labels: List<String>
    val temperature: Double
    val threshold: Float
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val meta = JSONObject(context.assets.open("echogesture.int8.json").bufferedReader().use { it.readText() })
        val arr = meta.getJSONArray("labels")
        labels = List(arr.length()) { arr.getString(it) }
        temperature = meta.optDouble("temperature", 1.0)
        threshold = meta.optDouble("confidence_threshold", 0.6).toFloat()
        check(meta.getInt("feature_dim") == Features.FEATURE_DIM && meta.getInt("frames") == Features.FRAMES) {
            "Model/preprocessing schema mismatch"
        }
        check(temperature > 0) { "Temperature must be positive" }
        val bytes = context.assets.open("echogesture.int8.onnx").use { it.readBytes() }
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(1)
            setInterOpNumThreads(1)
        }
        session = env.createSession(bytes, opts)
    }

    fun classify(base: List<FloatArray>): Result {
        val x = Features.encodeSequence(base)
        val start = SystemClock.elapsedRealtimeNanos()
        val logits = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(x), longArrayOf(1, Features.FEATURE_DIM.toLong(), Features.FRAMES.toLong())
        ).use { input ->
            session.run(mapOf("landmarks" to input)).use { out ->
                @Suppress("UNCHECKED_CAST")
                (out[0].value as Array<FloatArray>)[0]
            }
        }
        val ms = (SystemClock.elapsedRealtimeNanos() - start) / 1e6
        val scaled = DoubleArray(logits.size) { logits[it] / temperature }
        val max = scaled.max()
        val e = DoubleArray(scaled.size) { exp(scaled[it] - max) }
        val sum = e.sum()
        val probs = e.map { (it / sum).toFloat() }
        val order = probs.indices.sortedByDescending { probs[it] }.take(5)
        return Result(labels[order[0]], probs[order[0]], order.map { labels[it] to probs[it] }, ms)
    }

    override fun close() {
        session.close()
    }
}
