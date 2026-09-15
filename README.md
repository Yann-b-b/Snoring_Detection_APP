# Snoring Detection App 😴

This Android app detects snoring sounds in real time using a TensorFlow Lite model.

## Features
- Real-time audio capture via microphone
- TensorFlow micro-frontend feature extraction reimplemented in pure Kotlin
  (no external DSP libs): log-mel spectrogram with noise reduction and PCAN
  auto-gain, producing 49 frames × 40 channels per second of audio
- CNN-based snore classification (`conv_float_model.tflite`)
- Median-smoothed decisions with dynamic UI color feedback
  (red = snore, green = no snore)

## Model
The model is the TensorFlow `speech_commands` "conv" architecture: input
`[1, 1960]` (49 frames × 40 micro-frontend channels), output `[1, 2]` softmax
`[snoring, not_snoring]`. It was trained by
[adrianagaler/Snoring-Detection](https://github.com/adrianagaler/Snoring-Detection)
on the Khan snoring dataset (1,000 one-second clips). That repo's INT8 export
(~244 KB) reports 90.7% accuracy; this app ships the **float** export
(`conv_float_model.tflite`).

## How it works
1. Audio is captured at 16 kHz and analyzed over a sliding 1-second window
   with a 100 ms hop (~10 model decisions per second).
2. Each window is converted to micro-frontend features (49 × 40 log-mel with
   noise reduction and PCAN) and fed to the CNN.
3. The app keeps the last 5 snoring probabilities and reports "snoring" when
   their median exceeds 0.6, which suppresses single-window flickers.
4. Everything runs on-device; no audio ever leaves the phone.

## Requirements
- Android 8.0 (API 26) or higher
- Microphone permission

## How to Run
1. Clone the repo
   ```bash
   git clone https://github.com/Yann-b-b/Snoring_Detection_APP.git
   ```
2. Open the project in Android Studio and let Gradle sync.
3. Run the `app` configuration on a device (the emulator has no real mic),
   grant microphone permission, and tap **Snore Detect**.
