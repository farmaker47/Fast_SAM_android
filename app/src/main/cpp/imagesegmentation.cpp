// Write C++ code here.
#include <jni.h>
#include <cstdlib>  // for rand
#include <ctime>    // for time
#include <vector>

// Constants
const int NUM_COLORS = 32;

// Generate random colors in ARGB format (as 32-bit integers)
std::vector<uint32_t> generateRandomHexColors() {
    std::vector<uint32_t> colors(NUM_COLORS);

    for (int i = 0; i < NUM_COLORS; ++i) {
        uint8_t red = std::rand() % 256;
        uint8_t green = std::rand() % 256;
        uint8_t blue = std::rand() % 256;
        uint8_t alpha = 200; // Fully (255) opaque or less (200) opaque

        // Combine into a single ARGB value (32-bit integer)
        colors[i] = (alpha << 24) | (red << 16) | (green << 8) | blue;
    }

    return colors;
}

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_georgios_soloupis_examples_imagesegmenter_Utils_generateRandomHexColorsJNI(JNIEnv *env, jobject thiz) {
    // Call the C++ function to generate random colors
    std::vector<uint32_t> colors = generateRandomHexColors();

    // Create a jintArray (Java int array) to return the colors
    jintArray result = env->NewIntArray(NUM_COLORS);
    env->SetIntArrayRegion(result, 0, NUM_COLORS, reinterpret_cast<jint *>(colors.data()));

    return result;
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_georgios_soloupis_examples_imagesegmenter_Utils_flattenProtosJNI(JNIEnv *env, jobject thiz, jobjectArray protos) {
    // Get the outer array length (32)
    int outerLength = env->GetArrayLength(protos);

    // Prepare Java class for 1D float arrays
    jclass floatArrayClass = env->FindClass("[F");

    // Create the result 2D array (32x25600)
    jobjectArray result = env->NewObjectArray(outerLength, floatArrayClass, NULL);

    // Iterate over the outer array (protos array)
    for (int i = 0; i < outerLength; ++i) {
        // Get the inner 160x160 array
        jobjectArray innerArray160 = (jobjectArray) env->GetObjectArrayElement(protos, i);

        // Create a 1D float array to store the flattened array (160*160 = 25600)
        jfloatArray flattenedRow = env->NewFloatArray(25600);
        jfloat* flattenedRowElements = env->GetFloatArrayElements(flattenedRow, 0);

        int k = 0;
        for (int j = 0; j < 160; ++j) {
            jfloatArray rowArray160 = (jfloatArray) env->GetObjectArrayElement(innerArray160, j);
            jfloat *rowElements = env->GetFloatArrayElements(rowArray160, 0);

            // Use a single loop to flatten the row
            memcpy(&flattenedRowElements[k], rowElements, 160 * sizeof(jfloat));
            k += 160;

            // Release the row elements and local references to avoid memory leaks
            env->ReleaseFloatArrayElements(rowArray160, rowElements, JNI_ABORT);
            env->DeleteLocalRef(rowArray160);
        }

        // Release the inner 160x160 array
        env->DeleteLocalRef(innerArray160);

        // Commit the flattened array to the result array
        env->ReleaseFloatArrayElements(flattenedRow, flattenedRowElements, 0);
        env->SetObjectArrayElement(result, i, flattenedRow);

        // Clean up the local reference for the flattened array
        env->DeleteLocalRef(flattenedRow);
    }

    return result;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_georgios_soloupis_examples_imagesegmenter_Utils_computeMasks(JNIEnv* env, jobject /* this */,
                                                                      jobjectArray masksInArray,
                                                                      jobjectArray protosFlattenedArray,
                                                                      jobjectArray masksArray) {

    size_t numMasks = 32;
    size_t numProtos = 32;
    size_t numFeatures = 25600;

    // Prepare JNI arrays without converting to C++ vectors
    std::vector<jfloatArray> masksInJni(numMasks);
    std::vector<jfloatArray> protosFlattenedJni(numProtos);
    std::vector<jfloatArray> masksJni(numMasks);

    // Pre-fetch the masksIn JNI arrays
    for (size_t i = 0; i < numMasks; ++i) {
        masksInJni[i] = (jfloatArray) env->GetObjectArrayElement(masksInArray, i);
    }

    // Pre-fetch the protosFlattened JNI arrays
    for (size_t i = 0; i < numProtos; ++i) {
        protosFlattenedJni[i] = (jfloatArray) env->GetObjectArrayElement(protosFlattenedArray, i);
    }

    // Compute the masks directly from JNI arrays
    for (size_t i = 0; i < numMasks; ++i) {
        jfloat* masksInElements = env->GetFloatArrayElements(masksInJni[i], nullptr);

        // Create new float array for result
        jfloatArray rowArray = env->NewFloatArray(numFeatures);
        jfloat* masksOutElements = env->GetFloatArrayElements(rowArray, nullptr);

        std::fill(masksOutElements, masksOutElements + numFeatures, 0.0f);

        for (size_t j = 0; j < numProtos; ++j) {
            jfloat* protosElements = env->GetFloatArrayElements(protosFlattenedJni[j], nullptr);

            // Perform the computation for each feature
            for (size_t k = 0; k < numFeatures; ++k) {
                masksOutElements[k] += masksInElements[j] * protosElements[k];
            }

            env->ReleaseFloatArrayElements(protosFlattenedJni[j], protosElements, JNI_ABORT);
        }

        // Commit the results to the masks array
        env->ReleaseFloatArrayElements(rowArray, masksOutElements, 0);
        env->SetObjectArrayElement(masksArray, i, rowArray);
        env->DeleteLocalRef(rowArray);

        env->ReleaseFloatArrayElements(masksInJni[i], masksInElements, JNI_ABORT);
    }

    // Clean up local references
    for (size_t i = 0; i < numMasks; ++i) {
        env->DeleteLocalRef(masksInJni[i]);
    }
    for (size_t i = 0; i < numProtos; ++i) {
        env->DeleteLocalRef(protosFlattenedJni[i]);
    }
}


extern "C"
JNIEXPORT void JNICALL
Java_com_georgios_soloupis_examples_imagesegmenter_Utils_reshapeMasks(JNIEnv* env, jobject /* this */,
                                                                      jobjectArray masksArray,
                                                                      jobjectArray masksReshapedArray,
                                                                      jint mh, jint mw) {

    size_t numMasks = env->GetArrayLength(masksArray);
    size_t totalElements = mh * mw;  // total elements in each mask

    // Directly reshape and set the data into masksReshapedArray
    for (size_t i = 0; i < numMasks; ++i) {
        // Get 1D array for the current mask (already flattened as mh * mw)
        jfloatArray maskArray1D = (jfloatArray) env->GetObjectArrayElement(masksArray, i);
        jfloat* maskElements = env->GetFloatArrayElements(maskArray1D, nullptr);

        // Get the 2D Java array where the reshaped mask will be stored
        jobjectArray reshapedMask2D = (jobjectArray) env->GetObjectArrayElement(masksReshapedArray, i);

        // Process each row (mh rows, mw columns per row)
        for (size_t row = 0; row < mh; ++row) {
            // Create a new 1D float array for this row (mw columns)
            jfloatArray rowArray1D = env->NewFloatArray(mw);

            // Copy the relevant segment of the original mask to this row
            env->SetFloatArrayRegion(rowArray1D, 0, mw, &maskElements[row * mw]);

            // Set this row into the reshaped 2D array
            env->SetObjectArrayElement(reshapedMask2D, row, rowArray1D);

            // Clean up the local reference for the row
            env->DeleteLocalRef(rowArray1D);
        }

        // Clean up local references and release JNI arrays
        env->ReleaseFloatArrayElements(maskArray1D, maskElements, JNI_ABORT);
        env->DeleteLocalRef(maskArray1D);
        env->DeleteLocalRef(reshapedMask2D);
    }
}
