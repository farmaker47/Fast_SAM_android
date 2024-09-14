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
import android.graphics.Canvas
import android.util.AttributeSet
import android.view.View
import kotlin.math.max
import kotlin.math.min

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    private var scaleBitmap: Bitmap? = null
    private var runningMode: ImageSegmenterHelper.Companion.RunningMode = ImageSegmenterHelper.Companion.RunningMode.IMAGE

    fun clear() {
        scaleBitmap = null
        invalidate()
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        scaleBitmap?.let {
            canvas.drawBitmap(it, 0f, 0f, null)
        }
    }

    fun setRunningMode(runningMode: ImageSegmenterHelper.Companion.RunningMode) {
        this.runningMode = runningMode
    }

    /*
       Function for live streaming.
     */
    fun setResultsFastSAM(bitmap: Bitmap, outputWidth: Int, outputHeight: Int) {
        // Calculate scale factor based on the largest dimension
        val scaleFactor = max(width * 1f / outputWidth, height * 1f / outputHeight)
        // Precompute scaled dimensions
        val scaledWidth = (outputWidth * scaleFactor).toInt()
        val scaledHeight = (outputHeight * scaleFactor).toInt()
        val finalWidth = scaledWidth - (scaledWidth / 4)
        // Create the scaled bitmap
        if (bitmap.width != finalWidth || bitmap.height != scaledHeight) {
            scaleBitmap = Bitmap.createScaledBitmap(bitmap, finalWidth, scaledHeight, false)
            invalidate()  // Only invalidate if the bitmap has actually changed
        }
    }

    /*
       Function for gallery's image segmentation.
     */
    fun setResultsImageFastSAM(bitmap: Bitmap, outputWidth: Int, outputHeight: Int) {
        val scaleFactor = when (runningMode) {
            ImageSegmenterHelper.Companion.RunningMode.IMAGE,
            ImageSegmenterHelper.Companion.RunningMode.VIDEO -> {
                min(width * 1f / outputWidth, height * 1f / outputHeight)
            }
            ImageSegmenterHelper.Companion.RunningMode.LIVE_STREAM -> {
                max(width * 1f / outputWidth, height * 1f / outputHeight)
            }
        }

        val scaleWidth = (outputWidth * scaleFactor).toInt()
        val scaleHeight = (outputHeight * scaleFactor).toInt()

        scaleBitmap = Bitmap.createScaledBitmap(
            bitmap, scaleWidth, scaleHeight, false
        )
        invalidate()
    }
}
