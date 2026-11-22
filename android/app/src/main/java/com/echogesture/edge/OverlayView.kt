package com.echogesture.edge

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

/** Draws hand landmarks over a FILL_CENTER PreviewView. Points are normalized, unmirrored. */
class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {
    private val joint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#7CF2D4") }
    private val bone = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#B37CF2D4"); strokeWidth = 5f
    }
    private var hands: List<FloatArray> = emptyList()
    private var imgW = 1; private var imgH = 1; private var mirror = true

    fun update(points: List<FloatArray>, width: Int, height: Int, mirrored: Boolean) {
        hands = points; imgW = width; imgH = height; mirror = mirrored
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (hands.isEmpty()) return
        val scale = maxOf(width.toFloat() / imgW, height.toFloat() / imgH)
        val dx = (width - imgW * scale) / 2f; val dy = (height - imgH * scale) / 2f
        fun x(p: FloatArray, k: Int): Float {
            val nx = if (mirror) 1f - p[2 * k] else p[2 * k]
            return dx + nx * imgW * scale
        }
        fun y(p: FloatArray, k: Int) = dy + p[2 * k + 1] * imgH * scale
        val r = 6f * resources.displayMetrics.density / 2f
        for (p in hands) {
            for (c in CONNECTIONS) canvas.drawLine(x(p, c[0]), y(p, c[0]), x(p, c[1]), y(p, c[1]), bone)
            for (k in 0 until 21) canvas.drawCircle(x(p, k), y(p, k), r, joint)
        }
    }

    companion object {
        private val CONNECTIONS = arrayOf(
            intArrayOf(0, 1), intArrayOf(1, 2), intArrayOf(2, 3), intArrayOf(3, 4),
            intArrayOf(0, 5), intArrayOf(5, 6), intArrayOf(6, 7), intArrayOf(7, 8),
            intArrayOf(5, 9), intArrayOf(9, 10), intArrayOf(10, 11), intArrayOf(11, 12),
            intArrayOf(9, 13), intArrayOf(13, 14), intArrayOf(14, 15), intArrayOf(15, 16),
            intArrayOf(13, 17), intArrayOf(0, 17), intArrayOf(17, 18), intArrayOf(18, 19), intArrayOf(19, 20),
        )
    }
}
