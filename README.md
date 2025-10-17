# Snoring Detection App 😴

This Android app detects snoring sounds in real time using a TensorFlow Lite model.

## Features
- Real-time audio capture via microphone
- MFCC preprocessing in pure Kotlin (no external libs)
- CNN-based snore classification (`conv_float_model.tflite`)
- Dynamic UI color feedback (red = snore, green = no snore)

## Requirements
- Android 8.0 (API 26) or higher
- Microphone permission

## How to Run
1. Clone the repo
   ```bash
   git clone https://github.com/Yann-b-b/Snoring_Detection_APP.git