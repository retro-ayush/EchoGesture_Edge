package com.echogesture.edge

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.abs

/** Kotlin feature code must match echogesture/features.py (fixture from tools/make_parity_fixture.py). */
class FeaturesParityTest {
    private val fixture = JSONObject(javaClass.classLoader!!.getResource("parity.json")!!.readText())

    private fun JSONArray.floats() = FloatArray(length()) { getDouble(it).toFloat() }

    private fun assertClose(expected: FloatArray, actual: FloatArray, what: String) {
        assertEquals("$what size", expected.size, actual.size)
        for (i in expected.indices) {
            if (abs(expected[i] - actual[i]) > 1e-4f) {
                throw AssertionError("$what[$i]: expected ${expected[i]}, got ${actual[i]}")
            }
        }
    }

    private fun frames(): List<FloatArray> {
        val arr = fixture.getJSONArray("frames")
        return List(arr.length()) { i ->
            val f = arr.getJSONObject(i)
            val pose = if (f.isNull("pose_xy")) null else f.getJSONArray("pose_xy").floats()
            val vis = if (f.isNull("vis")) null else f.getJSONArray("vis").floats()
            val hands = HashMap<String, Features.Hand>()
            val h = f.getJSONObject("hands")
            for (name in h.keys()) {
                val o = h.getJSONObject(name)
                hands[name] = Features.Hand(o.getJSONArray("world").floats(), o.getJSONArray("image").floats())
            }
            val actual = Features.frameFeatures(pose, vis, hands, f.getDouble("aspect"))
            assertClose(f.getJSONArray("expected").floats(), actual, "frame $i")
            actual
        }
    }

    @Test
    fun frameFeaturesMatchPython() {
        assertEquals(120, frames().size)
    }

    @Test
    fun encodeSequenceMatchesPython() {
        val base = frames()
        val seqs = fixture.getJSONArray("sequences")
        for (i in 0 until seqs.length()) {
            val s = seqs.getJSONObject(i)
            val n = s.getInt("length")
            assertClose(s.getJSONArray("expected").floats(), Features.encodeSequence(base.subList(0, n)), "sequence T=$n")
        }
    }
}
