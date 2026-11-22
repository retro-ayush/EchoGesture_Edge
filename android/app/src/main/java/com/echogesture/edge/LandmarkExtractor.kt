package com.echogesture.edge

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker

/**
 * Port of echogesture/extract.py Extractor: MediaPipe hand (2 hands) + pose lite landmarkers
 * in VIDEO mode, GPU delegate with CPU fallback. Frames must be upright and UNMIRRORED.
 */
class LandmarkExtractor(context: Context, preferGpu: Boolean = true) : AutoCloseable {
    class Frame(
        val features: FloatArray,
        /** Normalized image points of detected hands, for drawing. */
        val handPoints: List<FloatArray>,
        val poseDetected: Boolean,
        val processMs: Double,
    )

    private lateinit var hand: HandLandmarker
    private lateinit var pose: PoseLandmarker
    var delegate: String = "cpu"
        private set
    private var lastTimestamp = -1L

    init {
        var error: Exception? = null
        for (d in if (preferGpu) listOf(Delegate.GPU, Delegate.CPU) else listOf(Delegate.CPU)) {
            try {
                build(context, d)
                // Some GPUs/drivers create the delegate but fail on the first inference
                // (e.g. unsupported GL calls), so warm up on a blank frame before accepting it.
                val blank = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
                val image = BitmapImageBuilder(blank).build()
                pose.detectForVideo(image, 0)
                hand.detectForVideo(image, 0)
                lastTimestamp = 0
                delegate = if (d == Delegate.GPU) "gpu" else "cpu"
                error = null
                break
            } catch (e: Exception) {
                Log.w(TAG, "MediaPipe $d delegate unavailable", e)
                close()
                error = e
            }
        }
        if (error != null) throw RuntimeException("Could not create MediaPipe landmarkers: ${error.message}", error)
    }

    private fun build(context: Context, d: Delegate) {
        fun base(file: String) = BaseOptions.builder().setModelAssetPath(file).setDelegate(d).build()
        val h = HandLandmarker.createFromOptions(
            context, HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(base("hand_landmarker.task"))
                .setRunningMode(RunningMode.VIDEO)
                .setNumHands(2)
                .build()
        )
        try {
            pose = PoseLandmarker.createFromOptions(
                context, PoseLandmarker.PoseLandmarkerOptions.builder()
                    .setBaseOptions(base("pose_landmarker_lite.task"))
                    .setRunningMode(RunningMode.VIDEO)
                    .setNumPoses(1)
                    .setOutputSegmentationMasks(false)
                    .build()
            )
        } catch (e: Exception) {
            h.close(); throw e
        }
        hand = h
    }

    fun frame(bitmap: Bitmap, timestampMs: Long): Frame {
        val start = System.nanoTime()
        val ts = maxOf(lastTimestamp + 1, timestampMs)
        lastTimestamp = ts
        val image = BitmapImageBuilder(bitmap).build()
        val p = pose.detectForVideo(image, ts)
        val h = hand.detectForVideo(image, ts)

        var poseXy: FloatArray? = null
        var vis: FloatArray? = null
        val poses = p.landmarks()
        if (poses.isNotEmpty()) {
            val lm = poses[0]
            poseXy = FloatArray(lm.size * 2)
            vis = FloatArray(lm.size)
            for ((i, v) in lm.withIndex()) {
                poseXy[2 * i] = v.x(); poseXy[2 * i + 1] = v.y()
                vis[i] = v.visibility().orElse(1f)
            }
        }
        val hands = HashMap<String, Features.Hand>()
        val scores = HashMap<String, Float>()
        val points = ArrayList<FloatArray>()
        val handedness = h.handedness()
        val worlds = h.worldLandmarks()
        val images = h.landmarks()
        for (i in handedness.indices) {
            if (i >= worlds.size || i >= images.size || handedness[i].isEmpty()) continue
            val cat = handedness[i][0]
            val name = cat.categoryName(); val score = cat.score()
            val img = FloatArray(42)
            for ((k, v) in images[i].withIndex()) if (k < 21) { img[2 * k] = v.x(); img[2 * k + 1] = v.y() }
            points.add(img)
            if (score <= (scores[name] ?: -1f)) continue
            scores[name] = score
            val world = FloatArray(63)
            for ((k, v) in worlds[i].withIndex()) if (k < 21) {
                world[3 * k] = v.x(); world[3 * k + 1] = v.y(); world[3 * k + 2] = v.z()
            }
            hands[name] = Features.Hand(world, img)
        }
        val aspect = bitmap.width.toDouble() / bitmap.height
        val features = Features.frameFeatures(poseXy, vis, hands, aspect)
        return Frame(features, points, poseXy != null, (System.nanoTime() - start) / 1e6)
    }

    override fun close() {
        // A graph that failed (e.g. a broken GPU delegate) rethrows its error on close: ignore it.
        if (::hand.isInitialized) runCatching { hand.close() }
        if (::pose.isInitialized) runCatching { pose.close() }
    }

    companion object { private const val TAG = "EchoGesture" }
}
