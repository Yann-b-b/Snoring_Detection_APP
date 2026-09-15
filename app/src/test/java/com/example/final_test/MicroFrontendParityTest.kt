package com.example.final_test

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Regression test pinning the CURRENT numeric behavior of the pure-Kotlin
 * MicroFrontend.
 *
 * The checked-in reference vector was produced by running this same
 * implementation once on the deterministic synthetic clip (see
 * [MicroFrontendReferenceGenerator]). It is NOT ground truth from the
 * TensorFlow micro_frontend op — it only guards against unintended numeric
 * drift in future refactors.
 */
class MicroFrontendParityTest {

    private companion object {
        const val TOLERANCE = 1e-3f
    }

    @Test
    fun computeMatchesCheckedInReference() {
        val stream = javaClass.getResourceAsStream(MicroFrontendTestSupport.REFERENCE_RESOURCE)
            ?: error(
                "Missing ${MicroFrontendTestSupport.REFERENCE_RESOURCE} — regenerate it " +
                    "with MicroFrontendReferenceGenerator (see that class's docs)"
            )
        val reference = stream.bufferedReader().useLines { lines ->
            lines.filter { it.isNotBlank() }.map { it.toFloat() }.toList()
        }
        assertEquals(MicroFrontendTestSupport.FEATURE_SIZE, reference.size)

        MicroFrontendTestSupport.resetMicroFrontendState()
        val features = MicroFrontend.compute(MicroFrontendTestSupport.syntheticClip())

        assertEquals(MicroFrontendTestSupport.FEATURE_SIZE, features.size)
        for (i in features.indices) {
            assertEquals("feature[$i]", reference[i], features[i], TOLERANCE)
        }
    }
}
