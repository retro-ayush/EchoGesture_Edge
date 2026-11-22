package com.echogesture.edge

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.ImageButton
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Camera caption HUD (port of demo.py). Tap START, sign one word, tap FINISH.
 * Everything runs on the phone; no frames are stored or sent anywhere.
 */
class MainActivity : ComponentActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var caption: TextView
    private lateinit var detail: TextView
    private lateinit var alternatives: TextView
    private lateinit var status: TextView
    private lateinit var captureButton: Button
    private lateinit var switchButton: ImageButton

    private val worker: ExecutorService = Executors.newSingleThreadExecutor()
    private var extractor: LandmarkExtractor? = null   // worker thread only
    private var classifier: Classifier? = null         // worker thread only
    private var cameraProvider: ProcessCameraProvider? = null
    private var useFront = true

    // Capture state: touched only on the worker thread.
    private val sequence = ArrayList<FloatArray>()
    private var capturing = false
    private var fpsWindowStart = 0L
    private var fpsFrames = 0
    private var fps = 0.0

    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else caption.text = getString(R.string.need_camera)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        previewView = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        caption = findViewById(R.id.caption)
        detail = findViewById(R.id.detail)
        alternatives = findViewById(R.id.alternatives)
        status = findViewById(R.id.status)
        captureButton = findViewById(R.id.capture)
        switchButton = findViewById(R.id.switch_camera)

        val topBar = findViewById<View>(R.id.top_bar)
        val panel = findViewById<View>(R.id.panel)
        val topPad = topBar.paddingTop; val bottomPad = panel.paddingBottom
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.root)) { _, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            topBar.updatePadding(top = topPad + bars.top, left = bars.left, right = bars.right)
            panel.updatePadding(bottom = bottomPad + bars.bottom, left = bars.left, right = bars.right)
            insets
        }

        captureButton.isEnabled = false
        captureButton.setOnClickListener { worker.execute { toggleCapture() } }
        switchButton.setOnClickListener {
            useFront = !useFront
            worker.execute { capturing = false; sequence.clear() }
            setCapturingUi(false)
            bindCamera()
        }

        worker.execute { loadModels() }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            permission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun loadModels() {
        try {
            val t = SystemClock.elapsedRealtime()
            classifier = Classifier(this)
            extractor = LandmarkExtractor(this)
            val ms = SystemClock.elapsedRealtime() - t
            Log.i(TAG, "Models ready in $ms ms, MediaPipe delegate ${extractor?.delegate}")
            runOnUiThread {
                captureButton.isEnabled = true
                caption.text = getString(R.string.ready)
                detail.text = getString(R.string.ready_detail, classifier!!.labels.size)
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Model load failed", e)
            runOnUiThread { caption.text = getString(R.string.load_failed); detail.text = e.message ?: e.toString() }
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            cameraProvider = future.get()
            bindCamera()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCamera() {
        val provider = cameraProvider ?: return
        val selector = if (useFront && provider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA))
            CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
        if (selector == CameraSelector.DEFAULT_BACK_CAMERA) useFront = false
        val resolution = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .build()
        val preview = Preview.Builder().setResolutionSelector(resolution).build()
            .also { it.surfaceProvider = previewView.surfaceProvider }
        // ~640x480 analysis is plenty for MediaPipe (it resizes internally) and keeps latency low.
        val analysisResolution = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(
                androidx.camera.core.resolutionselector.ResolutionStrategy(
                    android.util.Size(640, 480),
                    androidx.camera.core.resolutionselector.ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                )
            ).build()
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(analysisResolution)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        val mirrored = useFront
        analysis.setAnalyzer(worker) { proxy -> analyze(proxy, mirrored) }
        try {
            provider.unbindAll()
            provider.bindToLifecycle(this, selector, preview, analysis)
        } catch (e: Exception) {
            Log.e(TAG, "Camera bind failed", e)
            caption.text = getString(R.string.camera_failed)
            detail.text = e.message ?: ""
        }
    }

    /** Worker thread. */
    private fun analyze(proxy: ImageProxy, mirrored: Boolean) {
        val ex = extractor
        if (ex == null) { proxy.close(); return }
        val bitmap: Bitmap
        try {
            bitmap = uprightBitmap(proxy)
        } finally {
            proxy.close()
        }
        val frame = try {
            ex.frame(bitmap, SystemClock.uptimeMillis())
        } catch (e: Exception) {
            Log.e(TAG, "Landmark extraction failed on ${ex.delegate}", e)
            if (ex.delegate == "gpu") {   // some GPUs fail at runtime: fall back to CPU once
                ex.close()
                extractor = try { LandmarkExtractor(this, preferGpu = false) } catch (e2: Exception) { null }
            }
            return
        }
        val now = SystemClock.elapsedRealtime()
        if (fpsWindowStart == 0L) fpsWindowStart = now
        fpsFrames++
        if (now - fpsWindowStart >= 1000) {
            fps = fpsFrames * 1000.0 / (now - fpsWindowStart); fpsFrames = 0; fpsWindowStart = now
        }
        var tooLong = false
        if (capturing) {
            sequence.add(frame.features)
            if (sequence.size > MAX_FRAMES) { capturing = false; sequence.clear(); tooLong = true }
        }
        val n = sequence.size; val cap = capturing
        val hands = frame.handPoints.size
        val stat = String.format(
            Locale.US, "Hands %d · Body %s · %s · %.0f fps · %.0f ms",
            hands, if (frame.poseDetected) "✓" else "–", ex.delegate.uppercase(Locale.US), fps, frame.processMs
        )
        val w = bitmap.width; val h = bitmap.height
        runOnUiThread {
            overlay.update(frame.handPoints, w, h, mirrored)
            status.text = stat
            if (tooLong) {
                setCapturingUi(false)
                caption.text = getString(R.string.too_long); detail.text = ""; alternatives.text = ""
            } else if (cap) {
                detail.text = getString(R.string.capturing_detail, n)
            }
        }
    }

    private fun uprightBitmap(proxy: ImageProxy): Bitmap {
        val raw = proxy.toBitmap()
        val rot = proxy.imageInfo.rotationDegrees
        if (rot == 0) return raw
        val m = Matrix().apply { postRotate(rot.toFloat()) }
        return Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
    }

    /** Worker thread. */
    private fun toggleCapture() {
        if (!capturing) {
            sequence.clear(); capturing = true
            runOnUiThread {
                setCapturingUi(true)
                caption.text = getString(R.string.capturing)
                detail.text = getString(R.string.capturing_detail, 0)
                alternatives.text = ""
            }
            return
        }
        capturing = false
        val frames = ArrayList(sequence); sequence.clear()
        if (frames.size < MIN_FRAMES) {
            runOnUiThread {
                setCapturingUi(false)
                caption.text = getString(R.string.too_short); detail.text = getString(R.string.too_short_detail)
            }
            return
        }
        val c = classifier ?: return
        val r = try { c.classify(frames) } catch (e: Exception) {
            Log.e(TAG, "Classification failed", e)
            runOnUiThread { setCapturingUi(false); caption.text = getString(R.string.classify_failed); detail.text = e.message ?: "" }
            return
        }
        val accepted = r.confidence >= c.threshold
        val alt = r.top5.joinToString("   ") { (l, p) -> String.format(Locale.US, "%s %.0f%%", l, p * 100) }
        val info = String.format(
            Locale.US, "Score %.2f · classifier %.1f ms · %d frames", r.confidence, r.classifierMs, frames.size
        )
        runOnUiThread {
            setCapturingUi(false)
            caption.text = if (accepted) r.prediction else getString(R.string.uncertain, r.prediction)
            caption.setTextColor(getColor(if (accepted) R.color.caption else R.color.caption_dim))
            detail.text = info
            alternatives.text = alt
        }
    }

    private fun setCapturingUi(active: Boolean) {
        captureButton.text = getString(if (active) R.string.finish else R.string.start)
        captureButton.isActivated = active
        caption.setTextColor(getColor(R.color.caption))
        switchButton.isEnabled = !active
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraProvider?.unbindAll()
        worker.execute {
            extractor?.close(); extractor = null
            classifier?.close(); classifier = null
        }
        worker.shutdown()
    }

    companion object {
        private const val TAG = "EchoGesture"
        private const val MIN_FRAMES = 8
        private const val MAX_FRAMES = 300
    }
}
