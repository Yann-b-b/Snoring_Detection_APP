package com.example.final_test

import java.util.Random

/**
 * Shared fixtures for the MicroFrontend regression tests.
 *
 * The synthetic clip is fully deterministic: a 120 Hz sawtooth plus Gaussian
 * noise from a fixed-seed [java.util.Random] (whose algorithm is specified by
 * the JDK, so the samples are identical on every platform).
 *
 * MicroFrontend is a stateful singleton (noise and PCAN estimators persist
 * across compute calls), so tests must call [resetMicroFrontendState] before
 * computing to reproduce the fresh-process output the reference was
 * generated from. The reset mirrors the object's initial values:
 * noiseEstimate = 0, pcanMean = PCAN_OFFSET (80.0).
 */
object MicroFrontendTestSupport {
    const val SAMPLE_COUNT = 16_000
    const val FEATURE_SIZE = 1960
    const val REFERENCE_RESOURCE = "/microfrontend_reference_features.txt"

    private const val SAWTOOTH_HZ = 120.0
    private const val SAWTOOTH_AMPLITUDE = 0.5
    private const val NOISE_AMPLITUDE = 0.01
    private const val NOISE_SEED = 42L
    private const val NMELS = 40
    private const val PCAN_OFFSET = 80.0

    fun syntheticClip(): FloatArray {
        val rng = Random(NOISE_SEED)
        return FloatArray(SAMPLE_COUNT) { i ->
            val phase = (i * SAWTOOTH_HZ / SAMPLE_COUNT) % 1.0
            val sawtooth = (2.0 * phase - 1.0) * SAWTOOTH_AMPLITUDE
            (sawtooth + rng.nextGaussian() * NOISE_AMPLITUDE).toFloat()
        }
    }

    fun resetMicroFrontendState() {
        val cls = MicroFrontend::class.java
        cls.getDeclaredField("noiseEstimate")
            .apply { isAccessible = true }
            .set(MicroFrontend, DoubleArray(NMELS))
        cls.getDeclaredField("pcanMean")
            .apply { isAccessible = true }
            .set(MicroFrontend, DoubleArray(NMELS) { PCAN_OFFSET })
    }
}
