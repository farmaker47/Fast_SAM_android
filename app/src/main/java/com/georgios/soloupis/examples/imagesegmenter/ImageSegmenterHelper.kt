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

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ImageSegmenterHelper(
    var currentDelegate: Int = DELEGATE_CPU,
    var runningMode: RunningMode = RunningMode.IMAGE,
    var currentModel: Int = MODEL_FASTSAM,
    val context: Context,
    var imageSegmenterListener: SegmenterListener? = null
) {
    private var interpreterFastSam: Interpreter? = null
    private var inferenceTime: Long = 0L

    init {
        loadModel(FAST_SAM_MODEL)
    }

    @Throws(IOException::class)
    private fun loadModelFile(modelFile: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelFile)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        val retFile = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        fileDescriptor.close()
        return retFile
    }

    private fun loadModel(model: String) {

        val tfliteOptions = Interpreter.Options()
        /*if (true) {
            // Use with Tensorflow 2.8.0
            //tfliteOptions.addDelegate(GpuDelegate())

            //val delegate = GpuDelegate(GpuDelegate.Options().setQuantizedModelsAllowed(true))
        }*/
        tfliteOptions.setNumThreads(4)

        interpreterFastSam = Interpreter(loadModelFile(model), tfliteOptions)
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
        return interpreterFastSam == null
    }

    fun segmentLiveStreamFrame(imageProxy: ImageProxy, isFrontCamera: Boolean) {
        if (runningMode != RunningMode.LIVE_STREAM) {
            throw IllegalArgumentException(
                "Attempting to call segmentLiveStreamFrame" + " while not using RunningMode.LIVE_STREAM"
            )
        }

        val bitmapBuffer = Bitmap.createBitmap(
            imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888
        )

        imageProxy.use {
            bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer)
        }

        // Used for rotating the frame image so it matches our models
        val matrix = Matrix().apply {
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())

            if(isFrontCamera) {
                postScale(
                    -1f,
                    1f,
                    imageProxy.width.toFloat(),
                    imageProxy.height.toFloat()
                )
            }
        }

        imageProxy.close()

        val rotatedBitmap = Bitmap.createBitmap(
            bitmapBuffer,
            0,
            0,
            bitmapBuffer.width,
            bitmapBuffer.height,
            matrix,
            true
        )

        val time = System.currentTimeMillis()
        // Inputs
        val imageProcessor =
            ImageProcessor.Builder()
                .add(ResizeOp(Utils.MODEL_INPUTS_SIZE, Utils.MODEL_INPUTS_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0.0f, 255.0f))
                .build()
        var tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(rotatedBitmap)
        tensorImage = imageProcessor.process(tensorImage)
        val inputTensorBuffer = tensorImage.buffer
        val inputArray = arrayOf(inputTensorBuffer)

        // Outputs
        val probabilityBuffer1 = TensorBuffer.createFixedSize(
            intArrayOf(1, 37, 8400),
            DataType.FLOAT32
        )
        val probabilityBuffer2 = TensorBuffer.createFixedSize(
            intArrayOf(1, 160, 160, 32),
            DataType.FLOAT32
        )
        val outputMap = HashMap<Int, Any>()
        outputMap[0] = probabilityBuffer1.buffer
        outputMap[1] = probabilityBuffer2.buffer

        interpreterFastSam?.runForMultipleInputsOutputs(inputArray, outputMap)

        // Convert to a float array with the desired shape
        val flatFloatArray = probabilityBuffer1.floatArray
        // Step 2: Convert the flat float array into a 3D float array of shape [1, 37, 8400]
        val boxesArray = Array(1) { Array(37) { FloatArray(8400) } }
        for (i in 0 until 37) {
            for (j in 0 until 8400) {
                boxesArray[0][i][j] = flatFloatArray[i * 8400 + j]
            }
        }

        val flatFloatArray2 = probabilityBuffer2.floatArray

        val masksArray = Array(1) { Array(32) { Array(160) { FloatArray(160) } } }
        for (i in 0 until 160) {
            for (j in 0 until 160) {
                for (k in 0 until 32) {
                    // Calculate the index in flatFloatArray based on original shape [1, 160, 160, 32]
                    val originalIndex = i * 160 * 32 + j * 32 + k

                    // Place the value in the new transposed position [1, 32, 160, 160]
                    masksArray[0][k][i][j] = flatFloatArray2[originalIndex]
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
        const val FAST_SAM_MODEL = "FastSAM-s_float16_final.tflite"
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
