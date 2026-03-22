package com.currencyidentifier

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
import android.widget.EditText
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.Job
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.*
import android.text.Editable
import android.text.TextWatcher

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var previewView: PreviewView
    private lateinit var esp32Preview: ImageView
    private lateinit var modeSelector: RadioGroup
    private lateinit var modeSettingsRow: View
    private lateinit var modeInputEditText: EditText
    private var esp32StreamJob: Job? = null
    private var esp32StreamConnection: HttpURLConnection? = null
    private var lastEsp32PreviewUpdateMs: Long = 0L
    private var lastEsp32Frame: Bitmap? = null
    private var lastEsp32StreamErrorMs: Long = 0L
    private var currentEsp32StreamBaseUrl: String? = null
    private var esp32IpDebounceJob: Job? = null
    private var esp32PollingJob: Job? = null

    private lateinit var captureFab: FloatingActionButton
    private lateinit var scanLine: View
    private lateinit var scanOverlay: View
    private lateinit var btnFlash: ImageButton
    private lateinit var btnSwitchCamera: ImageButton
    private lateinit var btnSpeak: ImageButton
    private lateinit var btnHistory: ImageButton

    private lateinit var resultValue: TextView
    private lateinit var resultCurrencyName: TextView
    private lateinit var resultConfidence: TextView
    private lateinit var statusIndicator: View

    private var imageCapture: ImageCapture? = null
    private var tfliteInterpreter: Interpreter? = null
    private var labels: List<String> = listOf()
    private var cameraProvider: ProcessCameraProvider? = null
    private var boundCamera: Camera? = null
    private var tts: TextToSpeech? = null
    private var lastErrorSpokenAtMs: Long = 0L
    
    private enum class Mode { MOBILE, ESP32 }
    private var mode: Mode = Mode.MOBILE
    private var cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
    private var scanAnimatorRunning = false
    private var isProcessing = false

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA, Manifest.permission.INTERNET)
        private const val MODEL_FILE = "best-fp16.tflite"
        private const val LABELS_FILE = "labels.txt"
        private const val ESP32_DEFAULT_BASE_URL = "http://192.168.4.1"
        private const val ESP32_STREAM_URL_SUFFIX = "/stream"
        private const val ESP32_CAPTURE_URL_SUFFIX = "/capture"
        private const val TAG = "CurrencyApp"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Clear history from previous session on every fresh launch
        getSharedPreferences("currency_history", Context.MODE_PRIVATE).edit().clear().apply()
        
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        esp32Preview = findViewById(R.id.esp32Preview)
        modeSelector = findViewById(R.id.modeSelector)
        modeSettingsRow = findViewById(R.id.modeSettingsRow)
        modeInputEditText = findViewById(R.id.modeInput)
        // ESP32 hotspot uses a fixed IP, so keep the field locked and error-free.
        modeInputEditText.setText(ESP32_DEFAULT_BASE_URL)
        modeInputEditText.isEnabled = true
        modeInputEditText.isFocusable = true
        modeInputEditText.isFocusableInTouchMode = true
        modeInputEditText.isClickable = true

        // Editing the ESP32 IP should not trigger network calls automatically.
        // We only connect when the user taps "Capture" to keep ESP32 stable.
        modeInputEditText.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) = Unit
            override fun afterTextChanged(s: Editable?) = Unit
        })

        captureFab = findViewById(R.id.captureFab)
        scanLine = findViewById(R.id.scanLine)
        scanOverlay = findViewById(R.id.scanOverlay)
        btnFlash = findViewById(R.id.btnFlash)
        btnSwitchCamera = findViewById(R.id.btnSwitchCamera)
        btnSpeak = findViewById(R.id.btnSpeak)
        btnHistory = findViewById(R.id.btnHistory)

        resultValue = findViewById(R.id.resultValue)
        resultCurrencyName = findViewById(R.id.resultCurrencyName)
        resultConfidence = findViewById(R.id.resultConfidence)
        statusIndicator = findViewById(R.id.statusIndicator)

        tts = TextToSpeech(this, this)

        loadModel()
        loadLabels()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        modeSelector.setOnCheckedChangeListener { _, checkedId ->
            when (checkedId) {
                R.id.radioMobile -> setMode(Mode.MOBILE)
                R.id.radioEsp32 -> setMode(Mode.ESP32)
            }
        }

        captureFab.setOnClickListener {
            if (isProcessing) return@setOnClickListener

            if (tfliteInterpreter == null) {
                loadModel()
            }
            
            setProcessingState(true)
            when (mode) {
                Mode.ESP32 -> captureFromEsp32()
                Mode.MOBILE -> captureAndInfer()
            }
        }

        btnSpeak.setOnClickListener {
            speak(getSpokenSummary())
        }

        btnFlash.setOnClickListener {
            toggleTorch()
        }

        btnSwitchCamera.setOnClickListener {
            switchCamera()
        }

        btnHistory.setOnClickListener {
            startActivity(Intent(this, HistoryActivity::class.java))
        }

        previewView.setOnTouchListener { _, event -> event.action == MotionEvent.ACTION_MOVE }

        setMode(Mode.MOBILE)
        startScanAnimation()
    }

    private fun setProcessingState(processing: Boolean) {
        isProcessing = processing
        captureFab.alpha = if (processing) 0.5f else 1.0f
        if (processing) {
            setResultUi(value = "--", name = "Processing...", confidence = null)
            speak("Processing. Please wait.")
        }
    }

    private fun setMode(newMode: Mode) {
        mode = newMode
        when (mode) {
            Mode.MOBILE -> {
                modeSettingsRow.visibility = View.GONE
                modeInputEditText.isEnabled = false
                modeInputEditText.isFocusable = false
                modeInputEditText.isFocusableInTouchMode = false
                modeInputEditText.isClickable = false
                esp32Preview.visibility = View.GONE
                previewView.visibility = View.VISIBLE
                scanOverlay.visibility = View.VISIBLE
                scanLine.visibility = View.VISIBLE
                btnFlash.visibility = View.VISIBLE
                btnSwitchCamera.visibility = View.VISIBLE
                speak("Mobile camera mode. Point the camera at the currency and press capture.")
                startCamera()
            }
            Mode.ESP32 -> {
                modeSettingsRow.visibility = View.VISIBLE
                // Allow the user to set the ESP32 IP/URL (default is 192.168.4.1).
                if (modeInputEditText.text?.toString()?.trim().isNullOrEmpty()) {
                    modeInputEditText.setText(ESP32_DEFAULT_BASE_URL)
                }
                modeInputEditText.isEnabled = true
                modeInputEditText.isFocusable = true
                modeInputEditText.isFocusableInTouchMode = true
                modeInputEditText.isClickable = true
                esp32Preview.visibility = View.VISIBLE
                previewView.visibility = View.GONE
                scanOverlay.visibility = View.VISIBLE
                scanLine.visibility = View.VISIBLE
                btnFlash.visibility = View.GONE
                btnSwitchCamera.visibility = View.GONE
                stopCamera()
                speak("ESP32 camera mode. Enter the ESP32 IP address and press capture.")
                fetchEsp32PreviewOnce()
            }
        }
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun getEsp32BaseUrl(): String {
        val raw = modeInputEditText.text?.toString()?.trim().orEmpty()
        var s = if (raw.isEmpty()) ESP32_DEFAULT_BASE_URL else raw

        // Normalize to base URL
        s = s.trimEnd('/')
        s = s.removeSuffix(ESP32_STREAM_URL_SUFFIX)
        s = s.removeSuffix(ESP32_CAPTURE_URL_SUFFIX)

        if (!s.startsWith("http://") && !s.startsWith("https://")) {
            s = "http://$s"
        }
        return s.trimEnd('/')
    }

    private fun captureFromEsp32() {
        val baseUrl = getEsp32BaseUrl()
        val captureUrls = listOf(
            "${baseUrl}$ESP32_CAPTURE_URL_SUFFIX",
            "${baseUrl}/"
        )

        lifecycleScope.launch(Dispatchers.IO) {
            var bitmap: Bitmap? = null
            for (url in captureUrls) {
            try {
                val connection = URL(url).openConnection() as HttpURLConnection
                connection.connectTimeout = 5000
                connection.readTimeout = 5000
                connection.instanceFollowRedirects = true
                connection.doInput = true
                bitmap = connection.inputStream.use { input ->
                    BitmapFactory.decodeStream(input)
                }
                
                if (bitmap != null) break
            } catch (e: Exception) {
                Log.e(TAG, "ESP32 capture URL failed: $url", e)
                // Try next URL (e.g. /capture then /)
            }
            }

            withContext(Dispatchers.Main) {
                if (bitmap != null) {
                    lastEsp32Frame = bitmap
                    esp32Preview.setImageBitmap(bitmap)
                    processImage(bitmap)
                } else {
                    val usedFallback = tryFallbackToLastEsp32FrameOrToast()
                    if (!usedFallback) setProcessingState(false)
                }
            }
        }
    }

    // Preview fetch for ESP32 mode: fetches one JPEG from `/capture` without running inference.
    private fun fetchEsp32PreviewOnce() {
        val url = "${getEsp32BaseUrl()}$ESP32_CAPTURE_URL_SUFFIX"

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val connection = URL(url).openConnection() as HttpURLConnection
                connection.connectTimeout = 5000
                connection.readTimeout = 5000
                connection.instanceFollowRedirects = true
                connection.doInput = true

                val bitmap = connection.inputStream.use { input ->
                    BitmapFactory.decodeStream(input)
                }

                withContext(Dispatchers.Main) {
                    if (bitmap != null) {
                        lastEsp32Frame = bitmap
                        esp32Preview.setImageBitmap(bitmap)
                    } else if (lastEsp32Frame == null) {
                        setResultUi(value = "--", name = "Could not reach ESP32", confidence = null)
                        val now = System.currentTimeMillis()
                        if (now - lastErrorSpokenAtMs > 1500) {
                            lastErrorSpokenAtMs = now
                            speak("Could not reach ESP32. Please check the hotspot and IP address.")
                        }
                        Toast.makeText(this@MainActivity, "Could not reach ESP32", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (_: Exception) {
                withContext(Dispatchers.Main) {
                    if (lastEsp32Frame == null) {
                        setResultUi(value = "--", name = "Could not reach ESP32", confidence = null)
                        val now = System.currentTimeMillis()
                        if (now - lastErrorSpokenAtMs > 1500) {
                            lastErrorSpokenAtMs = now
                            speak("Could not reach ESP32. Please check the hotspot and IP address.")
                        }
                        Toast.makeText(this@MainActivity, "Could not reach ESP32", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }
    }

    private fun tryFallbackToLastEsp32FrameOrToast(): Boolean {
        val fallback = lastEsp32Frame
        if (fallback != null) {
            esp32Preview.setImageBitmap(fallback)
            processImage(fallback)
            return true
        }

        setResultUi(value = "--", name = "Could not reach ESP32", confidence = null)
        val now = System.currentTimeMillis()
        if (now - lastErrorSpokenAtMs > 1500) {
            lastErrorSpokenAtMs = now
            speak("Could not reach ESP32. Please check the hotspot and IP address.")
        }
        Toast.makeText(this, "Could not reach ESP32", Toast.LENGTH_SHORT).show()
        return false
    }

    private fun startEsp32Stream() {
        val baseUrl = getEsp32BaseUrl()
        if (esp32StreamJob?.isActive == true && currentEsp32StreamBaseUrl == baseUrl) return

        stopEsp32Stream()
        currentEsp32StreamBaseUrl = baseUrl
        lastEsp32PreviewUpdateMs = 0L

        esp32StreamJob = lifecycleScope.launch(Dispatchers.IO) {
            val streamUrl = "$baseUrl$ESP32_STREAM_URL_SUFFIX"

            var connection: HttpURLConnection? = null
            try {
                connection = URL(streamUrl).openConnection() as HttpURLConnection
                connection.connectTimeout = 5000
                connection.readTimeout = 0 // long-lived stream
                connection.instanceFollowRedirects = true
                connection.doInput = true
                esp32StreamConnection = connection

                val inputStream = connection.inputStream
                val buffer = ByteArray(4096)
                var prevByte = -1
                var capturingJpeg = false
                val jpegBuffer = ByteArrayOutputStream()
                val maxJpegBytes = 2_500_000 // safety guard to avoid unbounded buffer growth

                inputStream.use { stream ->
                    while (isActive) {
                        val bytesRead = stream.read(buffer)
                    if (bytesRead == -1) break

                        for (i in 0 until bytesRead) {
                            val b = buffer[i].toInt() and 0xFF

                            if (!capturingJpeg) {
                                if (prevByte == 0xFF && b == 0xD8) {
                                    capturingJpeg = true
                                    jpegBuffer.reset()
                                    jpegBuffer.write(0xFF)
                                    jpegBuffer.write(0xD8)
                                }
                            }
                            else {
                                jpegBuffer.write(b)

                                // Safety: if markers are missing, avoid infinite growth.
                                if (jpegBuffer.size() > maxJpegBytes) {
                                    capturingJpeg = false
                                    jpegBuffer.reset()
                                }

                                // End marker: 0xFF 0xD9
                                if (prevByte == 0xFF && b == 0xD9) {
                                    capturingJpeg = false
                                    val frameBytes = jpegBuffer.toByteArray()
                                    jpegBuffer.reset()

                                    val now = System.currentTimeMillis()
                                    if (now - lastEsp32PreviewUpdateMs > 150) {
                                        lastEsp32PreviewUpdateMs = now
                                        val bmp = BitmapFactory.decodeByteArray(frameBytes, 0, frameBytes.size)
                                        if (bmp != null) {
                                            lastEsp32Frame = bmp
                                        // Stream is working; stop polling fallback.
                                        stopEsp32PollingFallback()
                                            withContext(Dispatchers.Main) {
                                                esp32Preview.setImageBitmap(bmp)
                                            }
                                        }
                                    }
                                }
                            }

                            prevByte = b
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "ESP32 stream error", e)
                val now = System.currentTimeMillis()
                if (now - lastEsp32StreamErrorMs > 2000) {
                    lastEsp32StreamErrorMs = now
                    withContext(Dispatchers.Main) {
                        setResultUi(value = "--", name = "ESP32 stream not reachable", confidence = null)
                    }
                }
            } finally {
                try {
                    connection?.disconnect()
                } catch (_: Exception) {
                }
                esp32StreamConnection = null
            }
        }
    }

    private fun stopEsp32Stream() {
        esp32StreamJob?.cancel()
        esp32StreamJob = null
        try {
            esp32StreamConnection?.disconnect()
        } catch (_: Exception) {
        }
        esp32StreamConnection = null
    }

    private fun stopEsp32PollingFallback() {
        esp32PollingJob?.cancel()
        esp32PollingJob = null
    }

    private fun startEsp32PollingFallbackIfNeeded() {
        // If we already have streamed frames, don't poll.
        if (lastEsp32Frame != null) return
        if (esp32PollingJob?.isActive == true) return

        // Wait a bit for /stream to deliver frames first.
        esp32PollingJob = lifecycleScope.launch(Dispatchers.IO) {
            delay(6000)
            if (mode != Mode.ESP32) return@launch
            if (lastEsp32Frame != null) return@launch

            // Poll /capture periodically for preview frames.
            while (isActive && mode == Mode.ESP32 && lastEsp32Frame == null) {
                val bmp = fetchEsp32CaptureBitmap()
                if (bmp != null) {
                    lastEsp32Frame = bmp
                    withContext(Dispatchers.Main) {
                        esp32Preview.setImageBitmap(bmp)
                    }
                }
                delay(1000)
            }
        }
    }

    private fun fetchEsp32CaptureBitmap(): Bitmap? {
        val url = "${getEsp32BaseUrl()}$ESP32_CAPTURE_URL_SUFFIX"
        return try {
            val connection = URL(url).openConnection() as HttpURLConnection
            connection.connectTimeout = 5000
            connection.readTimeout = 5000
            connection.instanceFollowRedirects = true
            connection.doInput = true
            connection.inputStream.use { input ->
                BitmapFactory.decodeStream(input)
            }
        } catch (e: Exception) {
            null
        }
    }

    private fun processImage(bitmap: Bitmap) {
        lifecycleScope.launch(Dispatchers.Default) {
            val result = runInference(bitmap)
            withContext(Dispatchers.Main) {
                applyInferenceResultToUi(result)
                speak(getSpokenSummary())
                setProcessingState(false)
            }
        }
    }

    private fun captureAndInfer() {
        val imageCapture = imageCapture ?: run {
            Toast.makeText(this, "Camera not ready", Toast.LENGTH_SHORT).show()
            setProcessingState(false)
            return
        }

        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    val rotation = image.imageInfo.rotationDegrees
                    lifecycleScope.launch(Dispatchers.Default) {
                        try {
                            var bitmap = CameraUtils.imageProxyToBitmap(image)
                            bitmap = rotateBitmap(bitmap, rotation)
                            
                            val result = runInference(bitmap)
                            
                            withContext(Dispatchers.Main) {
                                applyInferenceResultToUi(result)
                                speak(getSpokenSummary())
                                setProcessingState(false)
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Processing failed", e)
                            withContext(Dispatchers.Main) { setProcessingState(false) }
                        } finally {
                            image.close()
                        }
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e(TAG, "Capture failed", exception)
                    setProcessingState(false)
                }
            }
        )
    }

    private fun runInference(bitmap: Bitmap): String {
        val interpreter = tfliteInterpreter ?: return "Model not loaded"
        
        return try {
            val inputTensor = interpreter.getInputTensor(0)
            val inputShape = inputTensor.shape()
            val modelSize = inputShape[1] 
            val inputBuffer = CameraUtils.prepareInputBuffer(bitmap, modelSize, inputTensor)

            val outputTensor = interpreter.getOutputTensor(0)
            val outputShape = outputTensor.shape()

            val isYoloOutput = (outputShape.size == 3 && outputShape[0] == 1 && outputShape[1] > 1000 && outputShape[2] >= 6)
            if (isYoloOutput) {
                val numPred = outputShape[1]
                val numVals = outputShape[2]
                val yoloOutput = Array(1) { Array(numPred) { FloatArray(numVals) } }
                interpreter.run(inputBuffer, yoloOutput)

                val detections = CameraUtils.yoloV5Postprocess(
                    output = yoloOutput[0],
                    labels = labels,
                    inputSize = modelSize,
                    confThreshold = 0.25f,
                    iouThreshold = 0.45f,
                    maxDetections = 5,
                )

                if (detections.isEmpty()) {
                    return "No currency detected\nTry better lighting."
                }

                val best = detections[0]
                val confPercent = (best.confidence * 100).toInt()
                "Currency: ${best.label}\nConfidence: $confPercent%"
            } else {
                val numClasses = outputShape[outputShape.size - 1]
                val probabilities = if (outputTensor.dataType() == DataType.UINT8 || outputTensor.dataType() == DataType.INT8) {
                    val outputBuffer = ByteArray(numClasses)
                    interpreter.run(inputBuffer, outputBuffer)
                    outputBuffer.map { (it.toInt() and 0xFF).toFloat() / 255.0f }.toFloatArray()
                } else {
                    val outputBuffer = Array(1) { FloatArray(numClasses) }
                    interpreter.run(inputBuffer, outputBuffer)
                    outputBuffer[0]
                }

                val topPredictions = CameraUtils.getTopPredictions(probabilities, labels, topK = 1)
                if (topPredictions.isEmpty()) return "Inference failed"

                val (label, confidence) = topPredictions[0]
                val confPercent = (confidence * 100).toInt()
                if (confidence > 0.6f) {
                    "Currency: $label\nConfidence: $confPercent%"
                } else {
                    "Low confidence ($confPercent%)\nTry again."
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Inference error", e)
            "Error: ${e.message}"
        }
    }

    private fun loadModel() {
        try {
            val assetFileDescriptor = assets.openFd(MODEL_FILE)
            val inputStream = assetFileDescriptor.createInputStream()
            val fileChannel = inputStream.channel
            val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, assetFileDescriptor.startOffset, assetFileDescriptor.declaredLength)
            tfliteInterpreter = Interpreter(modelBuffer)
        } catch (e: Exception) {
            Log.e(TAG, "Model load failed", e)
        }
    }

    private fun loadLabels() {
        try {
            labels = assets.open(LABELS_FILE).bufferedReader().readLines().filter { it.isNotBlank() }
        } catch (e: Exception) {
            Log.e(TAG, "Labels load failed", e)
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
                imageCapture = ImageCapture.Builder().build()
                cameraProvider?.unbindAll()
                boundCamera = cameraProvider?.bindToLifecycle(this, cameraSelector, preview, imageCapture)
                updateFlashUi()
            } catch (e: Exception) { Log.e(TAG, "Camera bind failed", e) }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        cameraProvider?.unbindAll()
        boundCamera = null
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) tts?.language = Locale.US
    }

    private fun speak(text: String) {
        val cleanText = text.replace("\n", " ")
        tts?.speak(cleanText, TextToSpeech.QUEUE_FLUSH, null, null)
    }

    private fun setResultUi(value: String, name: String, confidence: Int?) {
        resultValue.text = value
        resultCurrencyName.text = name
        resultConfidence.text = confidence?.let { "Confidence: $it%" } ?: "Waiting..."

        val c = when {
            confidence == null -> ContextCompat.getColor(this, R.color.text_secondary)
            confidence >= 70 -> ContextCompat.getColor(this, R.color.success_color)
            confidence >= 40 -> ContextCompat.getColor(this, R.color.accent_color)
            else -> ContextCompat.getColor(this, R.color.error_color)
        }
        statusIndicator.background = GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            setColor(c)
        }

        // Add to history if valid
        if (value != "--" && value.isNotEmpty()) {
            addToHistory(value)
        }
    }

    private fun addToHistory(amount: String) {
        val prefs = getSharedPreferences("currency_history", Context.MODE_PRIVATE)
        val jsonString = prefs.getString("history_data", "[]") ?: "[]"
        val jsonArray = JSONArray(jsonString)

        val timestamp = SimpleDateFormat("MMM dd, yyyy HH:mm", Locale.getDefault()).format(Date())
        val newItem = JSONObject()
        newItem.put("id", System.currentTimeMillis())
        newItem.put("amount", amount)
        newItem.put("date", timestamp)

        jsonArray.put(newItem)
        prefs.edit().putString("history_data", jsonArray.toString()).apply()
    }

    private fun applyInferenceResultToUi(result: String) {
        val lines = result.split("\n")
        val currencyLine = lines.firstOrNull { it.startsWith("Currency:", ignoreCase = true) }
        val confLine = lines.firstOrNull { it.contains("Confidence:", ignoreCase = true) }

        val confPercent = confLine
            ?.substringAfter("Confidence:", "")
            ?.replace("%", "")
            ?.trim()
            ?.toIntOrNull()

        if (currencyLine != null) {
            val label = currencyLine.substringAfter("Currency:", "").trim()
            val valueText = if (label.startsWith("₹")) label else "₹$label"
            setResultUi(value = valueText, name = "Indian Rupee", confidence = confPercent)
        } else {
            setResultUi(value = "--", name = lines.firstOrNull()?.take(40) ?: "Ready", confidence = confPercent)
        }
    }

    private fun getSpokenSummary(): String {
        val v = resultValue.text?.toString().orEmpty()
        val n = resultCurrencyName.text?.toString().orEmpty()
        if (v == "--") return n

        val confText = resultConfidence.text?.toString().orEmpty()
        val confPercent = confText
            .substringAfter("Confidence:", confText)
            .replace("%", "")
            .trim()
            .toIntOrNull()

        val rupees = v.replace("₹", "rupees").trim()
        return if (confPercent != null) {
            "$n. Amount is $rupees. Confidence is $confPercent percent."
        } else {
            "$n. Amount is $rupees."
        }
    }

    private fun toggleTorch() {
        val camera = boundCamera ?: return
        if (!camera.cameraInfo.hasFlashUnit()) return
        val enabled = camera.cameraInfo.torchState.value == TorchState.ON
        camera.cameraControl.enableTorch(!enabled)
        updateFlashUi()
    }

    private fun updateFlashUi() {
        val camera = boundCamera
        val hasFlash = camera?.cameraInfo?.hasFlashUnit() == true
        btnFlash.isEnabled = hasFlash
        btnFlash.alpha = if (hasFlash) 1f else 0.4f
    }

    private fun switchCamera() {
        cameraSelector = if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) {
            CameraSelector.DEFAULT_FRONT_CAMERA
        } else {
            CameraSelector.DEFAULT_BACK_CAMERA
        }
        startCamera()
    }

    private fun startScanAnimation() {
        if (scanAnimatorRunning) return
        scanAnimatorRunning = true

        scanOverlay.post {
            val h = scanOverlay.height
            if (h <= 0) {
                scanAnimatorRunning = false
                return@post
            }
            val top = (h * 0.28f)
            val bottom = (h * 0.72f)
            scanLine.translationY = top
            scanLine.animate()
                .translationY(bottom)
                .setDuration(1600)
                .setInterpolator(AccelerateDecelerateInterpolator())
                .withEndAction {
                    scanLine.animate()
                        .translationY(top)
                        .setDuration(1600)
                        .setInterpolator(AccelerateDecelerateInterpolator())
                        .withEndAction {
                            scanAnimatorRunning = false
                            startScanAnimation()
                        }
                        .start()
                }
                .start()
        }
    }

    override fun onDestroy() {
        stopEsp32Stream()
        super.onDestroy()
        tfliteInterpreter?.close()
        tts?.shutdown()
        
        // Also clear history here as a secondary measure
        getSharedPreferences("currency_history", Context.MODE_PRIVATE).edit().clear().apply()
    }
}
