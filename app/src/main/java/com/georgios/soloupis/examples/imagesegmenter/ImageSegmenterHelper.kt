/*
 * Copyright 2024 Georgios Soloupis. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.georgios.soloupis.examples.imagesegmenter

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import com.georgios.soloupis.examples.imagesegmenter.fragments.GalleryFragment
import java.nio.ByteBuffer


class ImageSegmenterHelper(
    var currentDelegate: Int = DELEGATE_CPU,
    var runningMode: RunningMode = RunningMode.IMAGE,
    var currentModel: Int = MODEL_FASTSAM,
    val context: Context,
    var imageSegmenterListener: SegmenterListener? = null
) {
    private val ortEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null
    private var ortOptions: OrtSession.SessionOptions? = null
    private var inferenceTime: Long = 0L

    init {
        loadModel(FAST_SAM_MODEL)
    }

    private fun loadModel(model: String) {
        ortOptions = OrtSession.SessionOptions()
        //ortOptions?.addCUDA()

        ortSession =
            ortEnvironment.createSession(context.assets.open(model).readBytes(), ortOptions)
    }

    fun clearImageSegmenter() {
        // interpreterFastSam?.close()
    }

    fun setListener(listener: SegmenterListener) {
        imageSegmenterListener = listener
    }

    fun clearListener() {
        imageSegmenterListener = null
    }

    // Return running status of image segmenter helper
    fun isClosed(): Boolean {
        return ortSession == null
    }

    fun segmentLiveStreamFrame(imageProxy: ImageProxy, isFrontCamera: Boolean) {
        if (runningMode != RunningMode.LIVE_STREAM) {
            throw IllegalArgumentException(
                "Attempting to call segmentLiveStreamFrame" + " while not using RunningMode.LIVE_STREAM"
            )
        }

        val time = System.currentTimeMillis()

        val bitmapBuffer = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)

        imageProxy.use {
            bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer)
        }

        imageProxy.close()

        val matrix = Matrix().apply {
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
            if (isFrontCamera) {
                postScale(-1f, 1f, imageProxy.width.toFloat(), imageProxy.height.toFloat())
            }
        }

        val rotatedBitmap = Bitmap.createBitmap(
            bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true
        )

        // val rotatedBitmap = Utils.rotateBitmapUsingOpenCV(bitmapBuffer, imageProxy.imageInfo.rotationDegrees.toDouble())

        val imagePixels = Utils.bitmapToFloatBufferOnnx(
            rotatedBitmap,
            GalleryFragment.INPUT_ONNX_DIMENSIONS,
            GalleryFragment.INPUT_ONNX_DIMENSIONS
        )
        val inputTensor =
            OnnxTensor.createTensor(
                ortEnvironment,
                imagePixels,
                // 1, 3, 640, 640
                longArrayOf(
                    1,
                    3,
                    GalleryFragment.INPUT_ONNX_DIMENSIONS.toLong(),
                    GalleryFragment.INPUT_ONNX_DIMENSIONS.toLong()
                )
            )
        val inputOnnxName = ortSession?.inputNames?.iterator()?.next()
        val outputs = ortSession?.run(mapOf(inputOnnxName to inputTensor))


        val outputTensor0 = outputs?.get(0) as OnnxTensor
        val flatfloatarray0Onnx = Utils.byteBufferToFloatArray(outputTensor0.byteBuffer)
        val boxesArray = Array(1) { Array(37) { FloatArray(8400) } }
        for (i in 0 until 37) {
            for (j in 0 until 8400) {
                boxesArray[0][i][j] = flatfloatarray0Onnx[i * 8400 + j]
            }
        }


        val outputTensor1 = outputs.get(1) as OnnxTensor
        val flatfloatarray1Onnx = Utils.byteBufferToFloatArray(outputTensor1.byteBuffer)
        val masksArray = Array(1) { Array(32) { Array(160) { FloatArray(160) } } }
        var index = 0
        for (c in 0 until 32) {
            for (h in 0 until 160) {
                for (w in 0 until 160) {
                    masksArray[0][c][h][w] = flatfloatarray1Onnx[index++]
                }
            }
        }

        val masks = Utils.postProcess(boxesArray, masksArray)

        val bitmap = Utils.createCombinedBitmapFromFloatArray(masks)

        inferenceTime = System.currentTimeMillis() - time

        imageSegmenterListener?.onResultFastSAM(bitmap, inferenceTime)
    }

    // Wraps results from inference, the time it takes for inference to be
    // performed.
    data class ResultBundle(
        val results: ByteBuffer,
        val width: Int,
        val height: Int,
        val inferenceTime: Long,
    )

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val OTHER_ERROR = 0
        const val GPU_ERROR = 1

        const val MODEL_FASTSAM = 0
        const val FAST_SAM_MODEL = "FastSAM-s.onnx"
        private const val TAG = "ImageSegmenterHelper"

        enum class RunningMode {
            LIVE_STREAM,
            IMAGE,
            VIDEO
        }
    }

    interface SegmenterListener {
        fun onError(error: String, errorCode: Int = OTHER_ERROR)
        fun onResults(resultBundle: ResultBundle)
        fun onResultFastSAM(bitmap: Bitmap, time: Long)
    }
}
