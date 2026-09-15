package com.example.final_test

import java.io.File
import org.junit.Test

/**
 * Regenerates the reference feature vector for [MicroFrontendParityTest].
 * No-op unless the MICROFRONTEND_REFERENCE_OUT environment variable is set,
 * so it never interferes with normal test runs.
 *
 * Run it alone (it mutates MicroFrontend's singleton state):
 *
 * MICROFRONTEND_REFERENCE_OUT=$PWD/app/src/test/resources/microfrontend_reference_features.txt \
 *   ./gradlew :app:testDebugUnitTest \
 *   --tests com.example.final_test.MicroFrontendReferenceGenerator --rerun-tasks
 *
 * Only regenerate after an INTENDED numeric change to MicroFrontend; the
 * checked-in file otherwise pins current behavior.
 */
class MicroFrontendReferenceGenerator {

    @Test
    fun regenerateReferenceWhenRequested() {
        val outPath = System.getenv("MICROFRONTEND_REFERENCE_OUT") ?: return

        MicroFrontendTestSupport.resetMicroFrontendState()
        val features = MicroFrontend.compute(MicroFrontendTestSupport.syntheticClip())

        File(outPath).printWriter().use { writer ->
            features.forEach { writer.println(it) }
        }
        println("Wrote ${features.size} reference features to $outPath")
    }
}
