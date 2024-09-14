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

import androidx.lifecycle.ViewModel

/**
 *  This ViewModel is used to store image segmenter helper settings
 */
class MainViewModel : ViewModel() {

    private var _delegate: Int = ImageSegmenterHelper.DELEGATE_CPU
    private var _model: Int = ImageSegmenterHelper.MODEL_FASTSAM

    val currentDelegate: Int get() = _delegate
    val currentModel: Int get() = _model

    fun setDelegate(delegate: Int) {
        _delegate = delegate
    }

    fun setModel(model: Int) {
        _model = model
    }
}
