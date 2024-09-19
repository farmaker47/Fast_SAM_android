package com.georgios.soloupis.examples.imagesegmenter

import android.graphics.Bitmap
import android.graphics.Color
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random

object Utils {

    // Get array shape
    private fun getArrayShape(array: Any?): List<Int> {
        when (array) {
            is FloatArray -> {
                return listOf(array.size)
            }

            is Array<*> -> {
                if (array.isNotEmpty()) {
                    return listOf(array.size) + getArrayShape(array[0])
                }
                return listOf(array.size)
            }

            else -> {
                return emptyList()
            }
        }
    }

    fun printArrayShape(array: Any?) {
        val shape = getArrayShape(array)
        println("Shape: (${shape.joinToString(", ")})")
    }

    // Function to convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format
    // where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    private fun xywh2xyxy(x: Array<FloatArray>): Array<FloatArray> {
        // Create a copy of the input array
        val y = Array(x.size) { FloatArray(x[0].size) }

        for (i in x.indices) {
            for (j in x[i].indices) {
                y[i][j] = x[i][j]
            }
        }

        // Modify the copied array to convert from (x, y, width, height) to (x1, y1, x2, y2)
        for (i in x.indices) {
            y[i][0] = x[i][0] - x[i][2] / 2 // top left x
            y[i][1] = x[i][1] - x[i][3] / 2 // top left y
            y[i][2] = x[i][0] + x[i][2] / 2 // bottom right x
            y[i][3] = x[i][1] + x[i][3] / 2 // bottom right y
        }

        return y
    }

    // Non-maximum suppression
    private fun nms(boxes: Array<FloatArray>, scores: FloatArray, iouThreshold: Float): IntArray {
        // If no bounding boxes, return an empty array
        if (boxes.isEmpty()) {
            return intArrayOf()
        }

        // Get the coordinates of the bounding boxes
        val x1 = boxes.map { it[0] }.toFloatArray()
        val y1 = boxes.map { it[1] }.toFloatArray()
        val x2 = boxes.map { it[2] }.toFloatArray()
        val y2 = boxes.map { it[3] }.toFloatArray()

        // Calculate the areas of the bounding boxes
        val areas = FloatArray(boxes.size) { i -> (x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1) }

        // Sort the bounding boxes by the scores in descending order
        val order = scores.indices.sortedByDescending { scores[it] }.toIntArray()

        // Initialize a list to store the indices of the selected bounding boxes
        val keep = mutableListOf<Int>()

        // While there are still bounding boxes to process
        var orderList = order.toMutableList()
        while (orderList.isNotEmpty()) {
            // Get the index of the bounding box with the highest score
            val i = orderList[0]

            // Add the index to the list of selected bounding boxes
            keep.add(i)

            // Calculate the coordinates of the intersection between the current bounding box and the remaining bounding boxes
            val xx1 = FloatArray(orderList.size - 1) { j -> maxOf(x1[i], x1[orderList[j + 1]]) }
            val yy1 = FloatArray(orderList.size - 1) { j -> maxOf(y1[i], y1[orderList[j + 1]]) }
            val xx2 = FloatArray(orderList.size - 1) { j -> minOf(x2[i], x2[orderList[j + 1]]) }
            val yy2 = FloatArray(orderList.size - 1) { j -> minOf(y2[i], y2[orderList[j + 1]]) }

            // Calculate the width and height of the intersection
            val w = FloatArray(xx1.size) { j -> maxOf(0.0f, xx2[j] - xx1[j] + 1) }
            val h = FloatArray(yy1.size) { j -> maxOf(0.0f, yy2[j] - yy1[j] + 1) }

            // Calculate the area of the intersection
            val inter = FloatArray(w.size) { j -> w[j] * h[j] }

            // Calculate the IoU between the current bounding box and the remaining bounding boxes
            val iou = FloatArray(inter.size) { j -> inter[j] / (areas[i] + areas[orderList[j + 1]] - inter[j]) }

            // Get the indices of the bounding boxes with IoU less than the threshold
            val inds = iou.withIndex().filter { it.value <= iouThreshold }.map { it.index }

            // Update the order array to only include the remaining bounding boxes
            orderList = inds.map { orderList[it + 1] }.toMutableList()
        }

        // Convert the list of selected bounding box indices to an IntArray
        return keep.toIntArray()
    }

    // Perform non-max suppression.
    private fun nonMaxSuppression(
        prediction: Array<Array<FloatArray>>, // Assume input is a 3D array for predictions
        confThres: Float = 0.25f,
        iouThres: Float = 0.9f,
        classesS: List<Int>? = null,
        agnostic: Boolean = false,
        multiLabel: Boolean = false,
        labels: List<List<FloatArray>> = listOf(),
        maxDetections: Int = 32,
        nc: Int = 0,  // number of classes (optional)
        maxTimeImg: Float = 0.05f,
        maxNms: Int = 30000,
        maxWh: Int = 7680
    ): List<Array<FloatArray>> {
        val bs = prediction.size // batch size
        val numClasses = if (nc > 0) nc else prediction[0].size - 4
        val nm = prediction[0].size - numClasses - 4
        // val mi = 4 + numClasses // mask start index

        val xc = BooleanArray(8400) { i -> prediction[0][4][i] > confThres }

        // Initialize output list
        val output = MutableList(bs) { Array(0) { FloatArray(6 + nm) } }
        var x = mutableListOf<FloatArray>()

        for (xi in 0 until bs) {

            val transposedX = Array(8400) { FloatArray(37) }
            for (i in 0 until 8400) {
                for (j in 0 until 37) {
                    transposedX[i][j] = prediction[xi][j][i]
                }
            }

            // Filter the transposed array using the boolean array xc[xi]
            for (i in 0 until 8400) {
                if (xc[i]) {
                    x.add(transposedX[i])
                }
            }

            x = x.toTypedArray().toMutableList()

            // If no candidates remain, continue to next image
            // if (x.isEmpty()) continue

            // Convert boxes to (x1, y1, x2, y2) format
            val boxes = x.map { row -> row.sliceArray(0..3) }.toTypedArray()
            val classes = x.map { row -> row.sliceArray(4 until 4 + numClasses) }.toTypedArray()
            val masks = x.map { row -> row.sliceArray(4 + numClasses until 4 + numClasses + nm) }.toTypedArray()

            val boxesConverted = xywh2xyxy(boxes)

            // Find the class with the highest confidence
            val conf = classes.map { it.maxOrNull() ?: 0f }.toFloatArray()
            //val j = classes.map { it.indexOf(it.maxOrNull() ?: 0f).toFloat() }.toFloatArray()
            val j = classes.map { classArray -> classArray.indexOfFirst { it == (classArray.maxOrNull() ?: 0f) }.toFloat() }.toFloatArray()

            // Concatenate (x1, y1, x2, y2, conf, class, mask...)
            val xConcat = Array(x.size) { i ->
                boxesConverted[i] + floatArrayOf(conf[i], j[i]) + masks[i]
            }

            // Filter by confidence
            val xFiltered = xConcat.filter { it[4] > confThres }.toTypedArray()

            // Check if any boxes remain after filtering
            if (xFiltered.isEmpty()) continue

            // Sort by confidence score in descending order
            val sortedIndices = xFiltered.indices.sortedByDescending { xFiltered[it][4] }.take(maxNms)
            val xSorted = sortedIndices.map { xFiltered[it] }.toTypedArray()

            // Prepare for NMS
            val c = xSorted.map { it[5] * maxWh }.toFloatArray()
            val boxesNms = Array(xSorted.size) { i ->
                xSorted[i].sliceArray(0..3).mapIndexed { idx, v -> v + c[i] * (if (idx < 2) 1f else -1f) }.toFloatArray()
            }
            val scoresNms = xSorted.map { it[4] }.toFloatArray()

            // Perform NMS
            val nmsIndices = nms(boxesNms, scoresNms, iouThres).take(maxDetections).toIntArray()

            // Collect final detections for this image
            output[xi] = nmsIndices.map { xSorted[it] }.toTypedArray()
        }

        return output
    }

    // Crop mask
    private fun cropMask(masks: Array<Array<FloatArray>>, boxes: Array<FloatArray>): Array<Array<FloatArray>> {
        val n = masks.size
        val h = masks[0].size
        val w = masks[0][0].size

        // Initialize the output array with zeros (optional as Kotlin arrays are initialized with zeros)
        val croppedMasks = Array(n) { Array(h) { FloatArray(w) } }

        // Iterate through each mask and corresponding bounding box
        for (i in 0 until n) {
            // Get bounding box coordinates, and clamp them to ensure they are within image boundaries
            val x1 = boxes[i][0].toInt().coerceAtLeast(0)
            val y1 = boxes[i][1].toInt().coerceAtLeast(0)
            val x2 = boxes[i][2].toInt().coerceAtMost(w)
            val y2 = boxes[i][3].toInt().coerceAtMost(h)

            // Copy only the pixels within the bounding box directly
            for (y in y1 until y2) {
                System.arraycopy(masks[i][y], x1, croppedMasks[i][y], x1, x2 - x1)
            }
        }

        return croppedMasks
    }


    private fun processMaskNative(
        protos: Array<Array<FloatArray>>, // [c, mh, mw]
        masksIn: Array<FloatArray>, // [n, mask_dim]
        bboxes: Array<FloatArray>, // [n, 4]
        shape: Pair<Int, Int> // (h, w)
    ): Array<Array<FloatArray>> { // Output masks: [h, w, n]

        val c = protos.size
        val mh = protos[0].size
        val mw = protos[0][0].size
        val time = System.currentTimeMillis()

        // Flatten protos
        //val protosFlattened = flattenProtosJNI(protos)
        // Kotlin equivalent
        val protosFlattened = Array(32) { FloatArray(25600) }
        for (i in 0 until 32) {
            var k = 0  // Index for the flattened dimension
            for (j in 0 until 160) {
                for (l in 0 until 160) {
                    protosFlattened[i][k] = protos[i][j][l]
                    k++
                }
            }
        }

        // Matrix multiplication masks_in * protos_flattened
        val masks = Array(32) { FloatArray(25600) }
        // computeMasks(masksIn, protosFlattened, masks)
        // Populate masksIn and protosFlattened as needed
        // Kotlin equivalent
        for (i in masksIn.indices) {
            for (j in 0 until c) {
                for (k in 0 until mh * mw) {
                    masks[i][k] += masksIn[i][j] * protosFlattened[j][k]
                }
            }
        }

        // Reshape to [n, mh, mw] aka [32, 160, 160]
        val masksReshaped = Array(masks.size) { Array(mh) { FloatArray(mw) } }
        // reshapeMasks(masks, masksReshaped, mh, mw)
        // Kotlin equivalent
        for (i in masks.indices) {
            for (j in 0 until mh) {
                for (k in 0 until mw) {
                    masksReshaped[i][j][k] = masks[i][j * mw + k]
                }
            }
        }


        val gain = minOf(mh / shape.first.toFloat(), mw / shape.second.toFloat()) // gain = old / new
        val padW = (mw - shape.second * gain) / 2
        val padH = (mh - shape.first * gain) / 2
        val top = padH.toInt()
        val left = padW.toInt()
        val bottom = (mh - padH).toInt()
        val right = (mw - padW).toInt()

        // Crop masks
        val masksCropped = Array(masksReshaped.size) { Array(bottom - top) { FloatArray(right - left) } }
        for (i in masksReshaped.indices) {
            for (j in top until bottom) {
                for (k in left until right) {
                    masksCropped[i][j - top][k - left] = masksReshaped[i][j][k]
                }
            }
        }

        // Resize masks using OpenCV
        val resizedMasks = Array(masksCropped.size) { Array(shape.first) { FloatArray(shape.second) } }
        for (i in masksCropped.indices) {
            // Manually flatten the 2D array into a single FloatArray
            val maskFlatArray = FloatArray(masksCropped[i].size * masksCropped[i][0].size)
            var index = 0
            for (y in masksCropped[i].indices) {
                for (x in masksCropped[i][y].indices) {
                    maskFlatArray[index++] = masksCropped[i][y][x]
                }
            }

            // Create a Mat directly from the FloatArray, specifying rows and cols
            val mat = Mat(masksCropped[i].size, masksCropped[i][0].size, CvType.CV_32F)
            mat.put(0, 0, maskFlatArray)

            // Resize the Mat
            val resizedMat = Mat()
            Imgproc.resize(mat, resizedMat, Size(shape.second.toDouble(), shape.first.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)

            // Retrieve the resized Mat as a single FloatArray
            val resizedFlatArray = FloatArray(shape.first * shape.second)
            resizedMat.get(0, 0, resizedFlatArray)

            // Convert the flattened array back to 2D and store it in resizedMasks
            for (y in 0 until shape.first) {
                System.arraycopy(resizedFlatArray, y * shape.second, resizedMasks[i][y], 0, shape.second)
            }
        }

        // Crop masks based on bounding boxes
        val finalMasks = cropMask(resizedMasks, bboxes)

        // Log.v("utils_", (System.currentTimeMillis() - time).toString())

        // Convert masks to binary
        return Array(finalMasks.size) { i ->
            Array(finalMasks[i].size) { j ->
                FloatArray(finalMasks[i][j].size) { k ->
                    // The python implementation creates False/True mask
                    // Here we convert to 0/1
                    if (finalMasks[i][j][k] > 0) 1f else 0f
                }
            }
        }
    }

    fun postProcess(preds0: Array<Array<FloatArray>>, preds1: Array<Array<Array<FloatArray>>>): Array<Array<FloatArray>> {
        // Perform Non-Maximum Suppression (NMS)
        val pred = nonMaxSuppression(
            preds0,
            confThres = 0.4f,
            iouThres = 0.9f,
            agnostic = false,
            maxDetections = 32,
            nc = 1,
            classesS = null
        )

        // Size 32x32 as per python code.
        val sliceMasksIn = Array(32) { FloatArray(32) }
        for (i in 0 until pred[0].size) {
            for (j in 6 until 38) {
                sliceMasksIn[i][j - 6] = pred[0][i][j]
            }
        }

        // Size 32x4 as per python code.
        val sliceBoxes = Array(32) { FloatArray(4) }
        for (i in 0 until pred[0].size) {
            for (j in 0 until 4) {
                sliceBoxes[i][j] = pred[0][i][j]
            }
        }

        // Process masks using native function
        val masks = processMaskNative(
            protos = preds1[0],
            masksIn = sliceMasksIn,
            bboxes = sliceBoxes,
            shape = Pair(MODEL_INPUTS_SIZE, MODEL_INPUTS_SIZE)
        )

        return masks
    }

    /////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////
    private fun generateRandomHexColors(): List<Int> {
        // Generate 32 random colors
        return List(32) {
            // Create a random color with full opacity
            Color.argb(
                255, // Alpha (fully opaque)
                Random.nextInt(256), // Red
                Random.nextInt(256), // Green
                Random.nextInt(256) // Blue
            )
        }
    }

    private external fun generateRandomHexColorsJNI(): IntArray
    private external fun flattenProtosJNI(protos: Array<Array<FloatArray>>): Array<FloatArray>
    private external fun computeMasks(masksIn: Array<FloatArray>, protosFlattened: Array<FloatArray>, masks: Array<FloatArray>)
    private external fun reshapeMasks(masks: Array<FloatArray>, masksReshaped: Array<Array<FloatArray>>, mh: Int, mw: Int)


    fun createCombinedBitmapFromFloatArray(array: Array<Array<FloatArray>>): Bitmap {
        val bitmap = Bitmap.createBitmap(MODEL_INPUTS_SIZE, MODEL_INPUTS_SIZE, Bitmap.Config.ARGB_8888)

        val colors = generateRandomHexColorsJNI() // generate random colors

        // Initialize an IntArray to hold pixel data for faster manipulation
        val pixels = IntArray(MODEL_INPUTS_SIZE * MODEL_INPUTS_SIZE) { Color.TRANSPARENT }

        for (i in array.indices) {
            val element = array[i]
            val color = colors[i]

            for (y in 0 until MODEL_INPUTS_SIZE) {
                for (x in 0 until MODEL_INPUTS_SIZE) {
                    val value = element[y][x]

                    if (value == 1f) {
                        // Set the pixel in the corresponding position
                        pixels[y * MODEL_INPUTS_SIZE + x] = color
                    }
                }
            }
        }

        // Set all the pixels at once, this is much faster than setting them individually
        bitmap.setPixels(pixels, 0, MODEL_INPUTS_SIZE, 0, 0, MODEL_INPUTS_SIZE, MODEL_INPUTS_SIZE)

        return bitmap
    }

    fun bitmapToByteBuffer(
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

    const val MODEL_INPUTS_SIZE = 640

}
