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
package com.georgios.soloupis.examples.imagesegmenter.fragments

import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.lifecycleScope
import com.georgios.soloupis.examples.imagesegmenter.ImageSegmenterHelper
import com.georgios.soloupis.examples.imagesegmenter.ImageSegmenterHelper.Companion.FAST_SAM_MODEL
import com.georgios.soloupis.examples.imagesegmenter.MainViewModel
import com.georgios.soloupis.examples.imagesegmenter.Utils
import com.georgios.soloupis.examples.imagesegmenter.databinding.FragmentGalleryBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.Timer


class GalleryFragment : Fragment(), ImageSegmenterHelper.SegmenterListener {
    enum class MediaType {
        IMAGE, VIDEO, UNKNOWN
    }

    private var _fragmentGalleryBinding: FragmentGalleryBinding? = null
    private val fragmentGalleryBinding
        get() = _fragmentGalleryBinding!!
    private val viewModel: MainViewModel by activityViewModels()
    private var imageSegmenterHelper: ImageSegmenterHelper? = null
    private var backgroundScope: CoroutineScope? = null
    private var fixedRateTimer: Timer? = null
    private var interpreterFastSam: Interpreter? = null
    private var inferenceTime: Long = 0L

    private val getContent =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
            // Handle the returned Uri
            uri?.let { mediaUri ->
                when (val mediaType = loadMediaType(mediaUri)) {
                    MediaType.IMAGE -> runSegmentationOnImage(mediaUri)
                    else -> {
                        updateDisplayView(mediaType)
                        Toast.makeText(
                            requireContext(),
                            "Unsupported data type.",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }
            }
        }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentGalleryBinding =
            FragmentGalleryBinding.inflate(inflater, container, false)

        loadModel(FAST_SAM_MODEL)
        return fragmentGalleryBinding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        fragmentGalleryBinding.fabGetContent.setOnClickListener {
            stopAllTasks()
            getContent.launch(arrayOf("image/*", "video/*"))
            updateDisplayView(MediaType.UNKNOWN)
        }
        initBottomSheetControls()
    }

    override fun onPause() {
        stopAllTasks()
        super.onPause()
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        stopAllTasks()
        setUiEnabled(true)
    }

    @Throws(IOException::class)
    private fun loadModelFile(modelFile: String): MappedByteBuffer {
        val fileDescriptor = requireActivity().assets.openFd(modelFile)
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

    private fun initBottomSheetControls() {

        fragmentGalleryBinding.bottomSheetLayout.spinnerDelegate.setSelection(
            viewModel.currentDelegate, false
        )
        fragmentGalleryBinding.bottomSheetLayout.spinnerDelegate.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long
                ) {

                    viewModel.setDelegate(p2)
                    stopAllTasks()
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {/* no op */
                }
            }

        fragmentGalleryBinding.bottomSheetLayout.spinnerModel.setSelection(
            viewModel.currentModel, false
        )

        fragmentGalleryBinding.bottomSheetLayout.spinnerModel.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    parent: AdapterView<*>?,
                    view: View?,
                    position: Int,
                    id: Long
                ) {
                    viewModel.setModel(position)
                    stopAllTasks()
                }

                override fun onNothingSelected(parent: AdapterView<*>?) {
                    /* no op */
                }
            }
    }

    private fun stopAllTasks() {
        // Cancel all jobs
        fixedRateTimer?.cancel()
        fixedRateTimer = null
        backgroundScope?.cancel()
        backgroundScope = null

        // Clear Image Segmenter
        imageSegmenterHelper?.clearListener()
        imageSegmenterHelper?.clearImageSegmenter()
        imageSegmenterHelper = null

        with(fragmentGalleryBinding) {
            videoView.stopPlayback()
            videoView.setVideoURI(null)

            // clear overlay view
            overlayView.clear()
            progress.visibility = View.GONE
        }
        updateDisplayView(MediaType.UNKNOWN)
    }

    private fun loadBitmapFromAssets(fileName: String): Bitmap? {
        return try {
            // Open the input stream to the asset
            val inputStream: InputStream = requireActivity().assets.open(fileName)
            // Decode the stream into a Bitmap
            BitmapFactory.decodeStream(inputStream)
        } catch (e: IOException) {
            e.printStackTrace()
            null
        }
    }

    private fun loadBitmapFromAssetsScale(fileName: String): Bitmap? {
        return try {
            val options = BitmapFactory.Options()
            options.inPreferredConfig = Bitmap.Config.ARGB_8888
            options.inSampleSize = 2
            options.inScaled = false
            val inputStream: InputStream = requireActivity().assets.open(fileName)
            BitmapFactory.decodeStream(inputStream, null, options)
        } catch (e: IOException) {
            e.printStackTrace()
            null
        }
    }

    private fun bitmapToByteBuffer(
        bitmapIn: Bitmap,
        width: Int,
        height: Int,
        mean: Float = 0.0f,
        std: Float = 255.0f
    ): ByteBuffer {
        //val bitmap = scaleBitmapAndKeepRatio(bitmapIn, width, height)
        val inputImage = ByteBuffer.allocateDirect(1 * width * height * 3 * 4)
        inputImage.order(ByteOrder.nativeOrder())
        inputImage.rewind()

        val intValues = IntArray(width * height)
        bitmapIn.getPixels(intValues, 0, width, 0, 0, width, height)
        var pixel = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                val value = intValues[pixel++]

                inputImage.putFloat(((value shr 16 and 0xFF) - mean) / std)
                inputImage.putFloat(((value shr 8 and 0xFF) - mean) / std)
                inputImage.putFloat(((value and 0xFF) - mean) / std)
            }
        }

        inputImage.rewind()
        return inputImage
    }

    // Load and display the image.
    private fun runSegmentationOnImage(uri: Uri) {
        fragmentGalleryBinding.overlayView.setRunningMode(ImageSegmenterHelper.Companion.RunningMode.IMAGE)
        setUiEnabled(false)
        updateDisplayView(MediaType.IMAGE)
        val inputImage = uri.toBitmap()
        // inputImage = inputImage.scaleDown(INPUT_IMAGE_MAX_WIDTH)

        // val inputImage = loadBitmapFromAssets("resized_image.png")

        // display image on UI
        fragmentGalleryBinding.imageResult.setImageBitmap(inputImage)

        imageSegmenterHelper = ImageSegmenterHelper(
            context = requireContext(),
            runningMode = ImageSegmenterHelper.Companion.RunningMode.IMAGE,
            currentDelegate = viewModel.currentDelegate,
            imageSegmenterListener = this
        )

        lifecycleScope.launch(Dispatchers.Default) {

            val time = System.currentTimeMillis()
            // Inputs
            val imageProcessor =
                ImageProcessor.Builder()
                    .add(ResizeOp(Utils.MODEL_INPUTS_SIZE, Utils.MODEL_INPUTS_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                    .add(NormalizeOp(0.0f, 255.0f))
                    .build()
            var tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(inputImage)
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

            updateOverlayWithFastSAM(bitmap, inputImage.width, inputImage.height)

        }
    }

    private fun updateDisplayView(mediaType: MediaType) {
        fragmentGalleryBinding.imageResult.visibility =
            if (mediaType == MediaType.IMAGE) View.VISIBLE else View.GONE
        fragmentGalleryBinding.videoView.visibility =
            if (mediaType == MediaType.VIDEO) View.VISIBLE else View.GONE
        fragmentGalleryBinding.tvPlaceholder.visibility =
            if (mediaType == MediaType.UNKNOWN) View.VISIBLE else View.GONE
    }

    // Check the type of media that user selected.
    private fun loadMediaType(uri: Uri): MediaType {
        val mimeType = context?.contentResolver?.getType(uri)
        mimeType?.let {
            if (mimeType.startsWith("image")) return MediaType.IMAGE
            if (mimeType.startsWith("video")) return MediaType.VIDEO
        }

        return MediaType.UNKNOWN
    }

    private fun setUiEnabled(enabled: Boolean) {
        fragmentGalleryBinding.fabGetContent.isEnabled = enabled
        fragmentGalleryBinding.bottomSheetLayout.spinnerModel.isEnabled = enabled
        fragmentGalleryBinding.bottomSheetLayout.spinnerDelegate.isEnabled =
            enabled
    }

    private fun updateOverlayWithFastSAM(bitmap: Bitmap, width: Int, height: Int) {
        if (_fragmentGalleryBinding != null) {
            runBlocking {
                withContext(Dispatchers.Main) {
                    setUiEnabled(true)
                    fragmentGalleryBinding.bottomSheetLayout.inferenceTimeVal.text = String.format("%d ms", inferenceTime)
                    fragmentGalleryBinding.overlayView.setResultsImageFastSAM(
                        bitmap, width, height
                    )
                }
            }
        }
    }

    private fun segmentationError() {
        stopAllTasks()
        setUiEnabled(true)
    }

    // Convert Uri to bitmap image.
    private fun Uri.toBitmap(): Bitmap {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(
                requireActivity().contentResolver, this
            )
            ImageDecoder.decodeBitmap(source)
        } else {
            MediaStore.Images.Media.getBitmap(
                requireActivity().contentResolver, this
            )
        }.copy(Bitmap.Config.ARGB_8888, true)
    }

    /**
     * Scales down the given bitmap to the specified target width while maintaining aspect ratio.
     * If the original image is already smaller than the target width, the original image is returned.
     */
    private fun Bitmap.scaleDown(targetWidth: Float): Bitmap {
        // if this image smaller than widthSize, return original image
        if (targetWidth >= width) return this
        val scaleFactor = targetWidth / width
        return Bitmap.createScaledBitmap(
            this,
            (width * scaleFactor).toInt(),
            (height * scaleFactor).toInt(),
            false
        )
    }

    override fun onError(error: String, errorCode: Int) {
        backgroundScope?.launch {
            withContext(Dispatchers.Main) {
                segmentationError()
                Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT)
                    .show()
                if (errorCode == ImageSegmenterHelper.GPU_ERROR) {
                    fragmentGalleryBinding.bottomSheetLayout.spinnerDelegate.setSelection(
                        ImageSegmenterHelper.DELEGATE_CPU, false
                    )
                }
            }
        }
    }

    override fun onResults(resultBundle: ImageSegmenterHelper.ResultBundle) { }

    override fun onResultFastSAM(bitmap: Bitmap, time: Long) {}

    companion object {
        private const val TAG = "GalleryFragment"
    }
}
