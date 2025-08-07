# 😴 DontSleep – Driver Drowsiness Detection System

A real-time fatigue detection system that monitors drivers' eye and mouth movements to detect signs of drowsiness or yawning. Provides immediate audio-visual alerts to prevent road accidents.

## ⚙️ Features

- 🧠 Real-time facial landmark detection using MediaPipe
- 👁️ Detects Eye Aspect Ratio (EAR) for eye closure
- 👄 Detects Mouth Aspect Ratio (MAR) for yawning
- 🔊 pyttsx3 for speech-based alerts
- 🔗 Firebase integration for storing detection logs and analytics
- 📊 UI shows live camera feed with status (Awake, Yawning, Asleep)

## 🧠 How It Works

1. Captures webcam feed
2. Extracts 3D face landmarks using MediaPipe
3. Calculates EAR & MAR thresholds
4. Triggers alerts when driver is drowsy
5. Logs event details to Firebase

## 🧪 Tech Stack

- **Languages:** Python
- **Libraries:** MediaPipe, OpenCV, NumPy, pyttsx3, Firebase Admin SDK
- **Database:** Firebase Realtime Database

## 🖥️ UI Screens

- ✅ Green: Driver Awake
- ⚠️ Yellow: Yawning Detected
- 🛑 Red: Driver Asleep – Alert Triggered

## 🔐 System Requirements

- Python 3.x
- Webcam
- Firebase project & `serviceAccountKey.json`

