package com.example.final_test

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.core.content.ContextCompat
import com.example.final_test.ml.ConvFloatModel
import kotlinx.coroutines.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

// =============================== Activity ===============================

class MainActivity : ComponentActivity() {

    private companion object {
        const val SAMPLE_RATE = 16_000
        const val WINDOW_SAMPLES = SAMPLE_RATE            // 1 s analysis window
        const val HOP_MS = 100
        const val HOP_SAMPLES = SAMPLE_RATE * HOP_MS / 1000
        const val FEATURE_SIZE = 1960                     // 49 frames x 40 channels
        const val SNORE_PROB_WINDOW = 5                   // decisions smoothed over last N probs
        const val SNORE_THRESHOLD = 0.6f
        const val NUM_CLASSES = 2

        // Compile-time switch: false ships the float model, true the INT8
        // export from the source repo (~4x smaller, 90.7% reported accuracy).
        const val USE_INT8_MODEL = false
        const val INT8_MODEL_ASSET = "conv_int8_model.tflite"
    }

    /** Uniform inference API so the float and INT8 models are interchangeable. */
    private interface SnoreClassifier {
        /** Feeds 1,960 micro-frontend features; returns [snoring, not_snoring]. */
        fun classify(features: FloatArray): FloatArray
        fun close()
    }

    private class FloatModelClassifier(context: Context) : SnoreClassifier {
        private val model = ConvFloatModel.newInstance(context)
        private val input =
            TensorBuffer.createFixedSize(intArrayOf(1, FEATURE_SIZE), DataType.FLOAT32)

        override fun classify(features: FloatArray): FloatArray {
            input.loadArray(features)
            return model.process(input).outputFeature0AsTensorBuffer.floatArray
        }

        override fun close() = model.close()
    }

    private class QuantizedModelClassifier(context: Context) : SnoreClassifier {
        private val interpreter = Interpreter(FileUtil.loadMappedFile(context, INT8_MODEL_ASSET))
        private val inputTensor = interpreter.getInputTensor(0)
        private val outputTensor = interpreter.getOutputTensor(0)
        private val inputBuffer =
            ByteBuffer.allocateDirect(FEATURE_SIZE).order(ByteOrder.nativeOrder())
        private val outputBuffer =
            ByteBuffer.allocateDirect(NUM_CLASSES).order(ByteOrder.nativeOrder())
        private val probabilities = FloatArray(NUM_CLASSES)

        init {
            val type = inputTensor.dataType()
            require(type == DataType.UINT8 || type == DataType.INT8) {
                "Expected a quantized input tensor, got $type"
            }
        }

        override fun classify(features: FloatArray): FloatArray {
            val inParams = inputTensor.quantizationParams()
            val signedInput = inputTensor.dataType() == DataType.INT8
            val qMin = if (signedInput) -128 else 0
            val qMax = if (signedInput) 127 else 255
            inputBuffer.rewind()
            for (f in features) {
                val q = (Math.round(f / inParams.scale) + inParams.zeroPoint).coerceIn(qMin, qMax)
                inputBuffer.put(q.toByte())
            }
            inputBuffer.rewind()
            outputBuffer.rewind()
            interpreter.run(inputBuffer, outputBuffer)

            val outParams = outputTensor.quantizationParams()
            val signedOutput = outputTensor.dataType() == DataType.INT8
            outputBuffer.rewind()
            for (i in 0 until NUM_CLASSES) {
                val raw = outputBuffer.get().toInt().let { if (signedOutput) it else it and 0xFF }
                probabilities[i] = (raw - outParams.zeroPoint) * outParams.scale
            }
            return probabilities
        }

        override fun close() = interpreter.close()
    }

    private val requestMic =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startListening()
        }

    private var listenJob: Job? = null
    private var audioRecord: AudioRecord? = null
    private var classifier: SnoreClassifier? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        classifier =
            if (USE_INT8_MODEL) QuantizedModelClassifier(this) else FloatModelClassifier(this)

        setContent {
            var listening by remember { mutableStateOf(false) }
            var bg by remember { mutableStateOf(Color.Gray) }

            LaunchedEffect(Unit) { BgBus.onUpdate = { c -> bg = c } }

            Scaffold(modifier = Modifier.fillMaxSize()) { _ ->
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Box(Modifier.fillMaxSize()) { /* background */ }
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .background(bg)
                        )
                        Button(onClick = {
                            if (!listening) {
                                if (ContextCompat.checkSelfPermission(
                                        this@MainActivity, Manifest.permission.RECORD_AUDIO
                                    ) == PackageManager.PERMISSION_GRANTED
                                ) { startListening(); listening = true }
                                else requestMic.launch(Manifest.permission.RECORD_AUDIO)
                            } else {
                                stopListening(); listening = false; bg = Color.Gray
                            }
                        }) { Text(if (listening) "Stop" else "Snore Detect") }
                    }
                }
            }
        }
    }

    private fun startListening() {
        stopListening()

        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC, SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuf, WINDOW_SAMPLES * 2)
        ).apply { startRecording() }

        val rec = audioRecord!!
        val m = classifier ?: return

        listenJob = CoroutineScope(Dispatchers.Default).launch {
            val hopPcm = ShortArray(HOP_SAMPLES)
            val window = FloatArray(WINDOW_SAMPLES)
            val probHistory = FloatArray(SNORE_PROB_WINDOW)
            val sortedProbs = FloatArray(SNORE_PROB_WINDOW)
            var probIndex = 0
            var probCount = 0

            while (isActive) {
                var read = 0
                while (read < HOP_SAMPLES) {
                    val n = rec.read(hopPcm, read, HOP_SAMPLES - read)
                    if (n <= 0) break
                    read += n
                }

                // Slide the 1 s window left by one hop, append the new 100 ms
                // (PCM 16-bit -> float [-1,1], zero-pad if the read came up short).
                System.arraycopy(window, HOP_SAMPLES, window, 0, WINDOW_SAMPLES - HOP_SAMPLES)
                val tail = WINDOW_SAMPLES - HOP_SAMPLES
                for (i in 0 until read) window[tail + i] = hopPcm[i] / 32768f
                for (i in read until HOP_SAMPLES) window[tail + i] = 0f

                // ===== MICRO FRONTEND: 49×40, scaled like TF op =====
                // MicroFrontend keeps stateful noise/PCAN estimators across calls.
                // With overlapping windows they now update once per 100 ms hop —
                // 10x more often than the old one-shot-per-second loop. The
                // smoothing constants are intentionally left unchanged.
                val features: FloatArray = MicroFrontend.compute(window) // 1960 floats

                val out = m.classify(features)

                // Model output is [1,2] softmax: out[0]=snoring, out[1]=not_snoring.
                val snoreProb = out[0]
                probHistory[probIndex] = snoreProb
                probIndex = (probIndex + 1) % SNORE_PROB_WINDOW
                if (probCount < SNORE_PROB_WINDOW) probCount++

                // Median of the last SNORE_PROB_WINDOW probabilities (fewer while
                // the history warms up) smooths out single-window flickers.
                System.arraycopy(probHistory, 0, sortedProbs, 0, probCount)
                java.util.Arrays.sort(sortedProbs, 0, probCount)
                val medianProb = sortedProbs[probCount / 2]

                val snore = medianProb > SNORE_THRESHOLD
                BgBus.post(if (snore) Color.Red else Color.Green)
            }
        }
    }

    private fun stopListening() {
        listenJob?.cancel()
        listenJob = null
        audioRecord?.apply { stop(); release() }
        audioRecord = null
    }

    override fun onDestroy() {
        super.onDestroy()
        stopListening()
        classifier?.close()
        classifier = null
    }
}

/** Tiny event bus to update Compose from background thread. */
object BgBus {
    @Volatile var onUpdate: ((Color) -> Unit)? = null
    fun post(c: Color) { onUpdate?.invoke(c) }
}

// =========================== Micro Frontend (pure Kotlin) ===========================
//
// Matches TF Micro Frontend stages used when PREPROCESS='micro':
// STFT → Mel power → Noise Reduction → PCAN → log → clip [-80,0] → u8 → ×(10/256).
//
// Output: frames-major flatten of [49 frames][40 channels] → float[1960].
//

object MicroFrontend {
    // ---- Pipeline constants (chosen to mirror tflite-micro defaults) ----
    private const val SR = 16_000
    private const val NFFT = 512
    private const val WIN_SAMPLES = 480           // 30 ms
    private const val HOP_SAMPLES = 320           // 20 ms
    private const val NMELS = 40
    private const val TARGET_FRAMES = 49
    private const val DB_FLOOR = -80.0
    private const val EPS = 1e-12

    // Noise reduction smoothing (even/odd) & residual floor
    // These approximate the micro op behavior.
    private const val EVEN_SMOOTH = 0.5   // higher = faster update
    private const val ODD_SMOOTH  = 0.5
    private const val MIN_SIGNAL_REMAINING = 0.05

    // PCAN auto-gain constants
    private const val PCAN_ENABLE = true
    private const val PCAN_STRENGTH = 0.95
    private const val PCAN_OFFSET = 80.0
    private const val PCAN_GAIN_SMOOTH = 0.995

    // Final scaling: u8 (0..255) * (10/256)
    private const val U8_SCALE = 255.0 / 80.0
    private const val FINAL_SCALE = 10.0 / 256.0

    private val spec = AudioSpectrogram(SR, WIN_SAMPLES, HOP_SAMPLES, NFFT, useHann = true, preEmphasis = 0f)
    private val melBank = MelFilterBank.make(
        sampleRate = SR, nFft = NFFT, nMels = NMELS,
        fMin = 0.0, fMax = SR / 2.0, htk = false, slaneyNorm = true
    )

    // Stateful estimators (per channel)
    private var noiseEstimate = DoubleArray(NMELS) { 0.0 }
    private var pcanMean = DoubleArray(NMELS) { PCAN_OFFSET }

    // Set to true to dump a quick min/max once every few buffers
    private var debugCounter = 0

    fun compute(wave: FloatArray): FloatArray {
        // 1) Power spectrogram
        val powerSpec = spec.compute(wave) // [frames][nfft/2+1]
        val frames = powerSpec.size

        val out = FloatArray(TARGET_FRAMES * NMELS)
        var idx = 0

        for (f in 0 until TARGET_FRAMES) {
            // 2) Mel power
            val melPow = if (f < frames) melBank.apply(powerSpec[f]) else DoubleArray(NMELS) { EPS }

            // 3) Noise reduction (temporal smoothing + spectral subtraction)
            val smooth = if ((f and 1) == 0) EVEN_SMOOTH else ODD_SMOOTH
            for (m in 0 until NMELS) {
                // Update noise estimate (EMA)
                noiseEstimate[m] = (1.0 - smooth) * noiseEstimate[m] + smooth * melPow[m]
                // Subtract some of the noise, leave min residual
                val clean = max(melPow[m] - noiseEstimate[m], melPow[m] * MIN_SIGNAL_REMAINING)
                melPow[m] = max(clean, EPS)
            }

            // 4) PCAN auto-gain (normalize by slow-moving per-channel mean)
            if (PCAN_ENABLE) {
                for (m in 0 until NMELS) {
                    // Update per-channel mean energy
                    pcanMean[m] = PCAN_GAIN_SMOOTH * pcanMean[m] + (1.0 - PCAN_GAIN_SMOOTH) * melPow[m]
                    val denom = pcanMean[m] + PCAN_OFFSET
                    val norm = melPow[m] / max(denom, EPS)
                    // Apply strength (raise toward 1)
                    melPow[m] = norm.pow(PCAN_STRENGTH)
                }
            }

            // 5) Log-power dB
            val db = DoubleArray(NMELS)
            for (m in 0 until NMELS) {
                db[m] = 10.0 * ln(melPow[m]) / ln(10.0) // log10
            }

            // 6) Clip to [-80, 0], map to u8, then 7) scale by (10/256)
            for (m in 0 until NMELS) {
                var d = db[m]
                if (d < DB_FLOOR) d = DB_FLOOR
                if (d > 0.0) d = 0.0
                val u8 = (d - DB_FLOOR) * U8_SCALE // (db+80)*255/80
                out[idx++] = (u8 * FINAL_SCALE).toFloat()
            }
        }

        if ((debugCounter++ % 50) == 0) {
            var mn = Float.POSITIVE_INFINITY
            var mx = Float.NEGATIVE_INFINITY
            for (v in out) { if (v < mn) mn = v; if (v > mx) mx = v }
            android.util.Log.d("MicroFrontend", "feat[0..]: size=${out.size}, min=$mn, max=$mx")
        }

        return out // frames-major: 49×40 = 1960
    }
}

// --------------------------- Spectrogram (STFT) ---------------------------

class AudioSpectrogram(
    val sampleRate: Int,
    private val winSize: Int,      // 480
    private val hopSize: Int,      // 320
    val nFft: Int,                 // 512
    private val useHann: Boolean = true,
    private val preEmphasis: Float = 0f
) {
    private val window: FloatArray = if (useHann) hann(winSize) else hamming(winSize)

    fun compute(waveIn: FloatArray): Array<DoubleArray> {
        // Pre-emphasis (off by default here)
        val x = FloatArray(waveIn.size)
        if (x.isNotEmpty()) {
            x[0] = waveIn[0]
            if (preEmphasis == 0f) {
                for (i in 1 until x.size) x[i] = waveIn[i]
            } else {
                for (i in 1 until x.size) x[i] = waveIn[i] - preEmphasis * waveIn[i - 1]
            }
        }

        val nFrames = if (x.size < winSize) 1 else 1 + floor((x.size - winSize).toDouble() / hopSize).toInt()
        val out = Array(nFrames) { DoubleArray(nFft / 2 + 1) }

        val buf = DoubleArray(nFft)
        for (f in 0 until nFrames) {
            java.util.Arrays.fill(buf, 0.0)
            val start = f * hopSize
            val end = min(start + winSize, x.size)
            var j = 0
            var i = start
            while (i < end) {
                buf[j] = (x[i] * window[j]).toDouble()
                j++; i++
            }
            val (re, im) = Fft.fftReal(buf)
            val half = nFft / 2
            val row = out[f]
            for (k in 0..half) {
                val r = re[k]; val m = im[k]
                row[k] = r * r + m * m
            }
        }
        return out
    }

    private fun hamming(n: Int): FloatArray {
        val w = FloatArray(n)
        for (i in 0 until n) w[i] = (0.54 - 0.46 * cos(2.0 * Math.PI * i / (n - 1))).toFloat()
        return w
    }
    private fun hann(n: Int): FloatArray {
        val w = FloatArray(n)
        for (i in 0 until n) w[i] = (0.5 - 0.5 * cos(2.0 * Math.PI * i / (n - 1))).toFloat()
        return w
    }
}

// ------------------------------- FFT -------------------------------

object Fft {
    fun fftReal(realInput: DoubleArray): Pair<DoubleArray, DoubleArray> {
        val n = realInput.size
        require(n > 0 && (n and (n - 1)) == 0) { "FFT size must be power of two" }
        val re = realInput.copyOf()
        val im = DoubleArray(n)

        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) {
                val tr = re[i]; re[i] = re[j]; re[j] = tr
                val ti = im[i]; im[i] = im[j]; im[j] = ti
            }
        }

        var len = 2
        while (len <= n) {
            val ang = -2.0 * Math.PI / len
            val wLenCos = cos(ang)
            val wLenSin = sin(ang)
            var i = 0
            while (i < n) {
                var wCos = 1.0
                var wSin = 0.0
                for (k in 0 until len / 2) {
                    val uRe = re[i + k]; val uIm = im[i + k]
                    val vRe0 = re[i + k + len / 2]; val vIm0 = im[i + k + len / 2]
                    val vRe = vRe0 * wCos - vIm0 * wSin
                    val vIm = vRe0 * wSin + vIm0 * wCos
                    re[i + k] = uRe + vRe; im[i + k] = uIm + vIm
                    re[i + k + len / 2] = uRe - vRe; im[i + k + len / 2] = uIm - vIm
                    val nextCos = wCos * wLenCos - wSin * wLenSin
                    val nextSin = wCos * wLenSin + wSin * wLenCos
                    wCos = nextCos; wSin = nextSin
                }
                i += len
            }
            len = len shl 1
        }
        return Pair(re, im)
    }
}

// --------------------------- Mel Filter Bank ---------------------------

class MelFilterBank private constructor(
    private val weights: Array<DoubleArray>, // [nMels][nBins]
    val nMels: Int
) {
    fun apply(powerSpectrum: DoubleArray): DoubleArray {
        val out = DoubleArray(nMels)
        for (m in 0 until nMels) {
            var acc = 0.0
            val w = weights[m]
            for (k in powerSpectrum.indices) acc += w[k] * powerSpectrum[k]
            out[m] = acc
        }
        return out
    }

    companion object {
        fun make(
            sampleRate: Int,
            nFft: Int,
            nMels: Int,
            fMin: Double,
            fMax: Double,
            htk: Boolean,
            slaneyNorm: Boolean
        ): MelFilterBank {
            val nBins = nFft / 2 + 1

            fun hz2mel(hz: Double): Double =
                if (htk) 2595.0 * log10(1.0 + hz / 700.0)
                else 2595.0 * ln(1.0 + hz / 700.0)

            fun mel2hz(mel: Double): Double =
                if (htk) 700.0 * (10.0.pow(mel / 2595.0) - 1.0)
                else 700.0 * (exp(mel / 2595.0) - 1.0)

            val melMin = hz2mel(fMin)
            val melMax = hz2mel(fMax)
            val melPoints = DoubleArray(nMels + 2) { i ->
                melMin + (melMax - melMin) * i / (nMels + 1)
            }
            val hzPoints = DoubleArray(melPoints.size) { i -> mel2hz(melPoints[i]) }

            val binPoints = IntArray(hzPoints.size) { i ->
                floor((nFft + 1) * hzPoints[i] / sampleRate).toInt().coerceIn(0, nBins - 1)
            }

            val filters = Array(nMels) { DoubleArray(nBins) }
            for (m in 1..nMels) {
                val fL = binPoints[m - 1]
                val fC = binPoints[m]
                val fR = binPoints[m + 1]

                for (k in fL until fC) {
                    filters[m - 1][k] = (k - fL).toDouble() / max(1, fC - fL).toDouble()
                }
                for (k in fC..fR) {
                    filters[m - 1][k] = (fR - k).toDouble() / max(1, fR - fC).toDouble()
                }
            }

            // Slaney-style area normalization (matches micro frontend)
            if (slaneyNorm) {
                for (m in 0 until nMels) {
                    var area = 0.0
                    for (k in filters[m].indices) area += filters[m][k]
                    if (area > 0) {
                        val s = 2.0 / area
                        for (k in filters[m].indices) filters[m][k] *= s
                    }
                }
            }

            return MelFilterBank(filters, nMels)
        }
    }
}