# Diphthong Analysis Software

## Overview
This project is a **speech-processing tool** that transcribes and analyzes spoken language to detect **diphthongs** and **syllabic structures**. Built using **Python**, it utilizes **wav2vec2** for speech-to-text conversion and integrates the **CMU Pronouncing Dictionary** for phonetic analysis.

## Features
- **Speech-to-Text Processing**: Uses `wav2vec2` to convert audio into transcriptions.
- **Diphthong & Syllable Detection**: Analyzes phonemes from transcribed text to identify diphthongs.
- **Audio Preprocessing**: Utilizes `torchaudio` for waveform processing and resampling.
- **User-Friendly Interface**: Allows users to select audio files for analysis.

## Technologies Used
- **Python**
- **Wav2Vec2** (Facebook’s `wav2vec2-base-960h` model)
- **Torchaudio**
- **CMU Pronouncing Dictionary**
- **PyTorch**
- **Tkinter** (for file selection)

## How to Run
1. **Clone the repository**  
   git clone https://github.com/your-username/Diphthong-Analysis.git
   cd Diphthong-Analysis
2. **Install dependencies**
pip install torch torchaudio transformers pronouncing
3. **Run the script**
python diphthong_analysis.py
4. Select an audio file when prompted.

## Team Members
This project was developed by:

Samiran Maulick – Lead Developer

Anupam Hui – Speech Processing & Model Integration

Abhisikta Mondal – Phonetics & Linguistic Analysis
