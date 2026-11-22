package com.echogesture.edge

import kotlin.math.round
import kotlin.math.sqrt

/**
 * Exact port of echogesture/features.py (feature_version body2d-handlocal3d-v1).
 * Pure Kotlin so it can be unit-tested on the JVM against Python fixtures.
 */
object Features {
    val BODY = intArrayOf(0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24)
    const val COORD_DIM = 156   // body xy (26), local hand xyz (126), wrist xy (4)
    const val MASK_DIM = 17     // body (13), hands (2), wrist positions (2)
    const val BASE_DIM = COORD_DIM + MASK_DIM
    const val FEATURE_DIM = COORD_DIM * 2 + MASK_DIM
    const val FRAMES = 64

    /** One detected hand: 21x3 world xyz (flattened) and 21x2 normalized image xy (flattened). */
    class Hand(val world: FloatArray, val image: FloatArray)

    /**
     * poseXy: 33x2 normalized image coordinates (flattened) or null; poseVis: 33 visibilities or null.
     * hands: keyed by MediaPipe handedness label "Left"/"Right" on UNMIRRORED frames.
     * aspect: image width / height.
     */
    fun frameFeatures(poseXy: FloatArray?, poseVis: FloatArray?, hands: Map<String, Hand>, aspect: Double): FloatArray {
        val coords = DoubleArray(COORD_DIM)
        val masks = FloatArray(MASK_DIM)
        var cx = 0.0; var cy = 0.0; var scale = 0.0; var haveCenter = false
        if (poseXy != null && poseVis != null) {
            fun px(i: Int) = poseXy[2 * i] * aspect
            fun py(i: Int) = poseXy[2 * i + 1].toDouble()
            if (poseVis[11] >= .5f && poseVis[12] >= .5f) {
                cx = (px(11) + px(12)) / 2; cy = (py(11) + py(12)) / 2
                val dx = px(11) - px(12); val dy = py(11) - py(12)
                scale = sqrt(dx * dx + dy * dy)
                if (scale > 1e-4) {
                    haveCenter = true
                    for ((j, idx) in BODY.withIndex()) {
                        if (poseVis[idx] >= .5f) {
                            coords[j * 2] = (px(idx) - cx) / scale
                            coords[j * 2 + 1] = (py(idx) - cy) / scale
                            masks[j] = 1f
                        }
                    }
                }
            }
        }
        for ((h, name) in listOf("Left", "Right").withIndex()) {
            val hand = hands[name] ?: continue
            val w = hand.world; val im = hand.image
            require(w.size == 63 && im.size == 42) { "A hand requires 21 world and image landmarks" }
            if (w.any { !it.isFinite() }) continue
            val ddx = w[27] - w[0]; val ddy = w[28] - w[1]; val ddz = w[29] - w[2]
            val palm = sqrt((ddx * ddx + ddy * ddy + ddz * ddz).toDouble())
            if (palm < 1e-6) continue
            val start = 26 + 63 * h
            for (k in 0 until 21) for (c in 0 until 3) {
                coords[start + 3 * k + c] = (w[3 * k + c] - w[c]) / palm
            }
            masks[13 + h] = 1f
            if (haveCenter) {
                coords[152 + 2 * h] = (im[0] * aspect - cx) / scale
                coords[153 + 2 * h] = (im[1] - cy) / scale
                masks[15 + h] = 1f
            }
        }
        val out = FloatArray(BASE_DIM)
        for (i in 0 until COORD_DIM) out[i] = coords[i].toFloat().coerceIn(-10f, 10f)
        for (i in 0 until MASK_DIM) out[COORD_DIM + i] = masks[i]
        return out
    }

    /** Which coordinate channels are valid, from the 17 masks (coord_mask in Python). */
    private fun coordMask(masks: FloatArray, offset: Int): FloatArray {
        val v = FloatArray(COORD_DIM)
        for (j in 0 until 13) { v[2 * j] = masks[offset + j]; v[2 * j + 1] = masks[offset + j] }
        for (h in 0 until 2) for (k in 0 until 63) v[26 + 63 * h + k] = masks[offset + 13 + h]
        for (h in 0 until 2) { v[152 + 2 * h] = masks[offset + 15 + h]; v[153 + 2 * h] = masks[offset + 15 + h] }
        return v
    }

    /**
     * encode_sequence(base) with ablation "none": nearest uniform resampling to 64 frames,
     * masked coordinates, presence-gated velocity, masks. Returns the (329, 64) tensor
     * flattened row-major (feature-major), ready for ONNX input [1, 329, 64].
     */
    fun encodeSequence(base: List<FloatArray>, frames: Int = FRAMES): FloatArray {
        require(base.isNotEmpty() && base.all { it.size == BASE_DIM }) { "Expected a nonempty (T,$BASE_DIM) array" }
        require(base.all { r -> r.all { it.isFinite() } }) { "Landmarks contain NaN or infinity" }
        val n = base.size
        // np.rint(np.linspace(0, n-1, frames)): round half to even, last sample exactly n-1.
        val step = (n - 1).toDouble() / (frames - 1)
        val ids = IntArray(frames) { i -> if (i == frames - 1) n - 1 else round(i * step).toInt() }
        val xyz = Array(frames) { FloatArray(COORD_DIM) }
        val valid = Array(frames) { FloatArray(COORD_DIM) }
        for (t in 0 until frames) {
            val row = base[ids[t]]
            valid[t] = coordMask(row, COORD_DIM)
            for (i in 0 until COORD_DIM) xyz[t][i] = row[i] * valid[t][i]
        }
        val out = FloatArray(FEATURE_DIM * frames)
        for (t in 0 until frames) {
            val row = base[ids[t]]
            for (i in 0 until COORD_DIM) {
                out[i * frames + t] = xyz[t][i]
                val vel = if (t == 0) 0f else (xyz[t][i] - xyz[t - 1][i]) * valid[t][i] * valid[t - 1][i]
                out[(COORD_DIM + i) * frames + t] = vel
            }
            for (m in 0 until MASK_DIM) out[(2 * COORD_DIM + m) * frames + t] = row[COORD_DIM + m]
        }
        return out
    }
}
