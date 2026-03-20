package com.currencyidentifier

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Tensor
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

object CameraUtils {

    private const val TAG = "CameraUtils"

    fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        return image.toBitmap()
    }

    fun centerCrop(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val newSize = if (width < height) width else height
        val left = (width - newSize) / 2
        val top = (height - newSize) / 2
        return Bitmap.createBitmap(bitmap, left, top, newSize, newSize)
    }

    fun resizeBitmap(bitmap: Bitmap, width: Int, height: Int): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, width, height, true)
    }

    /**
     * Prepares a ByteBuffer for TFLite inference, handling both FLOAT32 and INT8 models.
     */
    fun prepareInputBuffer(bitmap: Bitmap, size: Int, inputTensor: Tensor): ByteBuffer {
        val cropped = centerCrop(bitmap)
        val resized = resizeBitmap(cropped, size, size)

        val dataType = inputTensor.dataType()
        val q = inputTensor.quantizationParams() // scale/zeroPoint for quantized models

        val bytePerChannel = if (dataType == DataType.FLOAT32) 4 else 1
        val inputBuffer = ByteBuffer.allocateDirect(1 * size * size * 3 * bytePerChannel)
        inputBuffer.order(ByteOrder.nativeOrder())
        inputBuffer.rewind()

        val pixels = IntArray(size * size)
        resized.getPixels(pixels, 0, size, 0, 0, size, size)

        for (pixel in pixels) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF

            if (dataType == DataType.FLOAT32) {
                inputBuffer.putFloat(r / 255.0f)
                inputBuffer.putFloat(g / 255.0f)
                inputBuffer.putFloat(b / 255.0f)
            } else {
                // For INT8/UINT8 models, apply tensor quantization params instead of raw bytes.
                // real_value = scale * (quantized_value - zeroPoint)
                // => quantized_value = real_value / scale + zeroPoint
                val rf = r / 255.0f
                val gf = g / 255.0f
                val bf = b / 255.0f

                val qr = (rf / q.scale + q.zeroPoint).toInt()
                val qg = (gf / q.scale + q.zeroPoint).toInt()
                val qb = (bf / q.scale + q.zeroPoint).toInt()

                if (dataType == DataType.UINT8) {
                    inputBuffer.put(qr.coerceIn(0, 255).toByte())
                    inputBuffer.put(qg.coerceIn(0, 255).toByte())
                    inputBuffer.put(qb.coerceIn(0, 255).toByte())
                } else {
                    inputBuffer.put(qr.coerceIn(-128, 127).toByte())
                    inputBuffer.put(qg.coerceIn(-128, 127).toByte())
                    inputBuffer.put(qb.coerceIn(-128, 127).toByte())
                }
            }
        }
        return inputBuffer
    }

    fun normalizeBitmap(bitmap: Bitmap, width: Int = 224, height: Int = 224): Array<Array<Array<FloatArray>>> {
        val cropped = centerCrop(bitmap)
        val resized = resizeBitmap(cropped, width, height)
        val pixels = IntArray(width * height)
        resized.getPixels(pixels, 0, width, 0, 0, width, height)

        val input = Array(1) { Array(height) { Array(width) { FloatArray(3) } } }

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val row = i / width
            val col = i % width

            input[0][row][col][0] = ((pixel shr 16) and 0xFF) / 255.0f
            input[0][row][col][1] = ((pixel shr 8) and 0xFF) / 255.0f
            input[0][row][col][2] = (pixel and 0xFF) / 255.0f
        }

        return input
    }

    fun getTopPredictions(predictions: FloatArray, labels: List<String>, topK: Int = 1): List<Pair<String, Float>> {
        val results = mutableListOf<Pair<String, Float>>()

        predictions.forEachIndexed { index, confidence ->
            val label = when {
                index < labels.size -> labels[index]
                index == 6 -> "unrecognized"
                else -> "Unknown Class $index"
            }
            results.add(Pair(label, confidence))
        }

        return results.sortedByDescending { it.second }.take(topK)
    }

    data class Detection(
        val label: String,
        val confidence: Float,
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val classId: Int,
    )

    /**
     * Post-process YOLOv5 TFLite output of shape [25200, 5 + numClasses].
     * Expected row format: [x, y, w, h, obj, cls0..clsN-1]
     */
    fun yoloV5Postprocess(
        output: Array<FloatArray>,
        labels: List<String>,
        inputSize: Int,
        confThreshold: Float = 0.25f,
        iouThreshold: Float = 0.45f,
        maxDetections: Int = 10,
    ): List<Detection> {
        if (output.isEmpty()) return emptyList()
        val numValues = output[0].size
        val numClasses = max(0, numValues - 5)
        if (numClasses <= 0) return emptyList()

        val candidates = ArrayList<Detection>(512)
        for (i in output.indices) {
            val row = output[i]
            if (row.size < 6) continue

            var x = row[0]
            var y = row[1]
            var w = row[2]
            var h = row[3]
            val obj = row[4]

            // Heuristic: some exports output normalized coords (0..1), others output pixels.
            val looksNormalized = (x <= 1.5f && y <= 1.5f && w <= 1.5f && h <= 1.5f)
            if (looksNormalized) {
                x *= inputSize
                y *= inputSize
                w *= inputSize
                h *= inputSize
            }

            // Find best class
            var bestClass = -1
            var bestClassProb = 0f
            for (c in 0 until numClasses) {
                val p = row[5 + c]
                if (p > bestClassProb) {
                    bestClassProb = p
                    bestClass = c
                }
            }
            if (bestClass < 0) continue

            val conf = obj * bestClassProb
            if (conf < confThreshold) continue

            val x1 = x - w / 2f
            val y1 = y - h / 2f
            val x2 = x + w / 2f
            val y2 = y + h / 2f

            val label = when {
                bestClass < labels.size -> labels[bestClass]
                bestClass == 6 -> "unrecognized"
                else -> "Unknown Class $bestClass"
            }
            candidates.add(
                Detection(
                    label = label,
                    confidence = conf,
                    x1 = x1,
                    y1 = y1,
                    x2 = x2,
                    y2 = y2,
                    classId = bestClass,
                )
            )
        }

        if (candidates.isEmpty()) return emptyList()
        candidates.sortByDescending { it.confidence }

        // NMS
        val selected = ArrayList<Detection>(min(maxDetections, candidates.size))
        for (det in candidates) {
            var keep = true
            for (sel in selected) {
                if (iou(det, sel) > iouThreshold) {
                    keep = false
                    break
                }
            }
            if (keep) {
                selected.add(det)
                if (selected.size >= maxDetections) break
            }
        }
        return selected
    }

    private fun iou(a: Detection, b: Detection): Float {
        val ax1 = min(a.x1, a.x2)
        val ay1 = min(a.y1, a.y2)
        val ax2 = max(a.x1, a.x2)
        val ay2 = max(a.y1, a.y2)

        val bx1 = min(b.x1, b.x2)
        val by1 = min(b.y1, b.y2)
        val bx2 = max(b.x1, b.x2)
        val by2 = max(b.y1, b.y2)

        val interX1 = max(ax1, bx1)
        val interY1 = max(ay1, by1)
        val interX2 = min(ax2, bx2)
        val interY2 = min(ay2, by2)

        val interW = max(0f, interX2 - interX1)
        val interH = max(0f, interY2 - interY1)
        val interArea = interW * interH

        val areaA = max(0f, ax2 - ax1) * max(0f, ay2 - ay1)
        val areaB = max(0f, bx2 - bx1) * max(0f, by2 - by1)
        val union = areaA + areaB - interArea

        return if (union <= 0f) 0f else interArea / union
    }
}

fun ImageProxy.toBitmap(): Bitmap {
    val planes = this.planes
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    yBuffer.rewind()
    uBuffer.rewind()
    vBuffer.rewind()

    val ySize = yBuffer.remaining()
    val u11Size = uBuffer.remaining()
    val v11Size = vBuffer.remaining()

    val nv21 = ByteArray(ySize + u11Size + v11Size)

    yBuffer.get(nv21, 0, ySize)
    val pixelStride = planes[1].pixelStride
    if (pixelStride == 1) {
        uBuffer.get(nv21, ySize, u11Size)
        vBuffer.get(nv21, ySize + u11Size, v11Size)
    } else {
        if (u11Size > 1) {
            val uvBuffer = ByteArray(u11Size + v11Size)
            uBuffer.get(uvBuffer, 0, u11Size)
            vBuffer.get(uvBuffer, u11Size, v11Size)
            for (i in 0 until u11Size) {
                nv21[ySize + i * 2] = uvBuffer[i]
                nv21[ySize + i * 2 + 1] = uvBuffer[u11Size + i]
            }
        }
    }

    val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, this.width, this.height), 100, out)
    val imageBytes = out.toByteArray()
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}
