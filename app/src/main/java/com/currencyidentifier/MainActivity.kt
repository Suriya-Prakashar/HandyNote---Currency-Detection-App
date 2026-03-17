package com.currencyidentifier

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.net.HttpURLConnection
import java.net.URL
import java.nio.channels.FileChannel
import java.util.*

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var previewView: PreviewView
    private lateinit var esp32Preview: ImageView
    private lateinit var captureButton: Button
    private lateinit var resultTextView: TextView
    private lateinit var modeSelector: RadioGroup
    private lateinit var esp32Settings: View
    private lateinit var esp32IpEditText: EditText
    
    private var imageCapture: ImageCapture? = null
    private var tfliteInterpreter: Interpreter? = null
    private var labels: List<String> = listOf()
    private var cameraProvider: ProcessCameraProvider? = null
    private var tts: TextToSpeech? = null
    
    private var isEsp32Mode = false

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA, Manifest.permission.INTERNET)
        // Use the FP16 (float16-weight) export for better accuracy than INT8 on most devices.
        private const val MODEL_FILE = "best-fp16.tflite"
        private const val LABELS_FILE = "labels.txt"
        private const val TAG = "CurrencyApp"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        esp32Preview = findViewById(R.id.esp32Preview)
        captureButton = findViewById(R.id.captureButton)
        resultTextView = findViewById(R.id.resultTextView)
        modeSelector = findViewById(R.id.modeSelector)
        esp32Settings = findViewById(R.id.esp32Settings)
        esp32IpEditText = findViewById(R.id.esp32Ip)

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
                R.id.radioRear -> {
                    isEsp32Mode = false
                    esp32Settings.visibility = View.GONE
                    esp32Preview.visibility = View.GONE
                    previewView.visibility = View.VISIBLE
                    startCamera()
                }
                R.id.radioEsp32 -> {
                    isEsp32Mode = true
                    esp32Settings.visibility = View.VISIBLE
                    esp32Preview.visibility = View.VISIBLE
                    previewView.visibility = View.GONE
                    stopCamera()
                }
            }
        }

        captureButton.setOnClickListener {
            if (tfliteInterpreter == null) {
                loadModel()
            }
            
            if (isEsp32Mode) {
                captureFromEsp32()
            } else {
                captureAndInfer()
            }
        }
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun captureFromEsp32() {
        val ip = esp32IpEditText.text.toString().trim()
        if (ip.isEmpty()) {
            Toast.makeText(this, "Please enter ESP32 IP address", Toast.LENGTH_SHORT).show()
            return
        }

        val url = if (ip.startsWith("http")) ip else "http://$ip/capture"
        
        resultTextView.text = "Fetching from ESP32..."
        
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val connection = URL(url).openConnection() as HttpURLConnection
                connection.connectTimeout = 5000
                connection.readTimeout = 5000
                val bitmap = BitmapFactory.decodeStream(connection.inputStream)
                
                withContext(Dispatchers.Main) {
                    if (bitmap != null) {
                        esp32Preview.setImageBitmap(bitmap)
                        processImage(bitmap)
                    } else {
                        resultTextView.text = "Failed to fetch image"
                        Toast.makeText(this@MainActivity, "Could not reach ESP32", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Log.e(TAG, "ESP32 Error", e)
                    resultTextView.text = "Error: ${e.message}"
                }
            }
        }
    }

    private fun processImage(bitmap: Bitmap) {
        lifecycleScope.launch(Dispatchers.Default) {
            val result = runInference(bitmap)
            withContext(Dispatchers.Main) {
                resultTextView.text = result
                speak(result)
            }
        }
    }

    private fun captureAndInfer() {
        val imageCapture = imageCapture ?: run {
            Toast.makeText(this, "Camera not ready", Toast.LENGTH_SHORT).show()
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
                            
                            lifecycleScope.launch(Dispatchers.Main) {
                                resultTextView.text = result
                                speak(result)
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Processing failed", e)
                        } finally {
                            image.close()
                        }
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e(TAG, "Capture failed", exception)
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
            
            // Use ByteBuffer to handle both FLOAT32 and INT8 models correctly
            val inputBuffer = CameraUtils.prepareInputBuffer(bitmap, modelSize, inputTensor)

            val outputTensor = interpreter.getOutputTensor(0)
            val outputShape = outputTensor.shape()

            // YOLOv5 TFLite export outputs [1, 25200, 5 + numClasses] (e.g. [1,25200,14])
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
                    return "No currency detected\nTry better lighting and move closer."
                }

                val best = detections[0]
                val confPercent = (best.confidence * 100).toInt()

                if (best.label.equals("None", ignoreCase = true)) {
                    "No currency detected\nConfidence: $confPercent%"
                } else {
                    "Currency: ${best.label}\nConfidence: $confPercent%"
                }
            } else {
                // Fallback for classification-style models: output [1, numClasses]
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
                if (topPredictions.isEmpty()) return "Inference failed: No results"

                val (label, confidence) = topPredictions[0]
                val confPercent = (confidence * 100).toInt()
                if (confidence > 0.6f) {
                    "Currency: $label\nConfidence: $confPercent%"
                } else if (confidence > 0.3f) {
                    "Likely $label ($confPercent%)\nMove closer for better result."
                } else {
                    "Detection low ($confPercent%)\nEnsure good light and focus."
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
            Log.d(TAG, "Model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Model load failed", e)
            runOnUiThread {
                resultTextView.text = "Model Load Error: ${e.message}"
            }
        }
    }

    private fun loadLabels() {
        try {
            labels = assets.open(LABELS_FILE).bufferedReader().readLines().filter { it.isNotBlank() }
            Log.d(TAG, "Labels loaded: ${labels.size} classes")
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
                cameraProvider?.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
            } catch (e: Exception) { Log.e(TAG, "Camera bind failed", e) }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        cameraProvider?.unbindAll()
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) tts?.language = Locale.US
    }

    private fun speak(text: String) {
        val cleanText = text.replace("\n", " ")
        tts?.speak(cleanText, TextToSpeech.QUEUE_FLUSH, null, null)
    }

    override fun onDestroy() {
        super.onDestroy()
        tfliteInterpreter?.close()
        tts?.shutdown()
    }
}
