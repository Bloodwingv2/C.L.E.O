# Cognitive Linguistic Engagement Operator (C.L.E.O.)

## Overview
Cognitive Linguistic Engagement Operator (C.L.E.O.) is a custom AI voice assistant featuring a personalized voice and offline TTS integration. It is designed for seamless and intelligent voice interactions with enhanced speech recognition and text-to-speech capabilities. Additional functionality, such as application control and a custom TTS engine, will be added in future updates.

## Features
- **Updated Speech Timer**: Fixed crashing issues caused by STT mismatches.
- **Increased Token Limit**: Allows for longer conversational memory.
- **Constant Listening Mode**: Prevents the assistant from reading the entire prompt unintentionally.
- **Speech-to-Text (STT) Integration**: Utilizes `pyaudio` and the `speech_recognition` library for continuous speech detection via a while loop.
- **Basic Text-to-Speech (TTS) Engine**: Enables fine-tuning of temperature and vocal characteristics for improved speech synthesis.
- **Optimized AI Model**: Uses `Qwen/Qwen2-1.5B-Instruct` for faster processing on local machines compared to other models.

## Installation

```sh
# Clone the repository
git clone https://github.com/yourusername/CLEO.git
cd CLEO

# Install dependencies
pip install -r requirements.txt

# Run the assistant
python cleo.py
