package com.echogesture.edge

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs

/**
 * Runs real WLASL100 test-clip landmarks through the on-device INT8 classifier and checks the
 * result against Python ONNX Runtime (fixture from tools/make_device_fixture.py). Also checks
 * that both MediaPipe landmarkers load on this device.
 */
@RunWith(AndroidJUnit4::class)
class ClassifierDeviceTest {
    private val instr = InstrumentationRegistry.getInstrumentation()

    @Test
    fun classifierMatchesPython() {
        // The fixture holds landmarks derived from WLASL clips, so it is not committed.
        assumeTrue("Run tools/make_device_fixture.py first", "device_fixture.json" in instr.context.assets.list("").orEmpty())
        val fixture = JSONObject(instr.context.assets.open("device_fixture.json").bufferedReader().use { it.readText() })
        val clips = fixture.getJSONArray("clips")
        Classifier(instr.targetContext).use { c ->
            for (i in 0 until clips.length()) {
                val clip = clips.getJSONObject(i)
                val rows = clip.getJSONArray("base")
                val base = List(rows.length()) { t ->
                    val r = rows.getJSONArray(t); FloatArray(r.length()) { r.getDouble(it).toFloat() }
                }
                val r = c.classify(base)
                val id = clip.getString("clip_id")
                assertEquals("clip $id prediction", clip.getString("prediction"), r.prediction)
                assertTrue(
                    "clip $id confidence ${r.confidence} vs ${clip.getDouble("confidence")}",
                    abs(r.confidence - clip.getDouble("confidence")) < 0.01
                )
                val top5 = clip.getJSONArray("top5")
                assertEquals("clip $id top5", List(top5.length()) { top5.getString(it) }, r.top5.map { it.first })
                android.util.Log.i("EchoGestureTest", "clip $id: ${r.prediction} ${"%.3f".format(r.confidence)} in ${"%.1f".format(r.classifierMs)} ms")
            }
        }
    }

    @Test
    fun landmarkersLoad() {
        LandmarkExtractor(instr.targetContext).use { ex ->
            val bmp = android.graphics.Bitmap.createBitmap(480, 640, android.graphics.Bitmap.Config.ARGB_8888)
            val f = ex.frame(bmp, 0)
            assertEquals(Features.BASE_DIM, f.features.size)
            android.util.Log.i("EchoGestureTest", "MediaPipe delegate ${ex.delegate}, frame ${"%.1f".format(f.processMs)} ms")
        }
    }
}
