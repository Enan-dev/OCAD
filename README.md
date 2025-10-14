# 🎯 Student Attention Tracker using MediaPipe & OpenCV

A **real-time multi-face attention tracking system** built with **MediaPipe**, **OpenCV**, and **Python**.  
It detects multiple students simultaneously, estimates **gaze direction**, **facial expression**, and **head pose**,  
and logs detailed attention metrics to a CSV file.

---

## 🧠 Overview

This project monitors **student attentiveness** in real time — ideal for online classes, study monitoring, or human behavior research.  
It uses **face landmarks**, **iris tracking**, and **head pose estimation** to determine whether each student is attentive, distracted, or looking away.  
Facial expressions are also analyzed to infer **emotional engagement** (e.g., smiling = engaged, neutral = bored).

---

## ⚙️ Features

✅ Tracks **up to 50 faces** simultaneously  
✅ Estimates **gaze direction** (left, right, away, center)  
✅ Estimates **head pose** with 3D direction line  
✅ Detects **facial expressions** (smile, sad, angry, neutral)  
✅ Calculates **class attention percentage**  
✅ Generates **CSV log reports** with all metrics  
✅ Displays **distraction warning alerts** in real time  

---

<details>
<summary>📦 <b>Installation Instructions</b> (click to expand)</summary>

### 🧰 Requirements

- Python 3.8 or newer  
- A working webcam  
- Libraries:  
  ```bash
  pip install opencv-python mediapipe numpy
<details> <summary>🗂️ <b>Project Structure</b></summary>
📁 student-attention-tracker/
│
├── attention_tracker.py        # Main source code
├── attention_log.csv            # Auto-generated logs
└── README.md                    # Project documentation

</details>
▶️ Running the Project

Run the Python script:

python attention_tracker.py

💻 What Happens:

Webcam opens and starts detecting faces.

Each student’s gaze, expression, and head pose are analyzed.

Real-time bounding boxes and status text are displayed.

Data is logged into attention_log.csv.

Press ESC to exit the program.

📊 CSV Log Format
Timestamp	Student_ID	Gaze_Status	Attention	Expression	Engagement	Final_Status
2025-10-15 10:42:30	1	Attentive	attentive	Smiling	engaged	Focused
2025-10-15 10:42:31	2	Looking Left	distracted	Neutral	bored	Distracted

🗒️ The file attention_log.csv is automatically created in the project directory.

🎨 Color & Status Legend
Status	Color	Meaning
🟢 Attentive	Green	Focused on screen
🟠 Looking Away	Orange	Temporarily distracted
🔴 Distracted	Red	Lost attention for >3 seconds
🟡 Engaged	Yellow	Smiling / participating
🔵 Confused	Blue	Unsure / thinking
🟣 Bored	Purple	Neutral / passive
<details> <summary>🧮 <b>How It Works (Technical Details)</b></summary>
1️⃣ Face & Iris Tracking

Uses MediaPipe FaceMesh (468 landmarks) to locate eyes, iris centers, mouth, eyebrows, etc.

2️⃣ Gaze Estimation

Compares iris center position with eye corner landmarks to determine direction:

Left / Right / Away / Attentive

3️⃣ Expression Analysis

Analyzes ratios of mouth openness, eyebrow height, and mouth slope:

Smile → engaged

Sad → confused

Angry → distracted

Neutral → bored

4️⃣ Head Pose Estimation

cv2.solvePnP() computes 3D rotation and draws a line projecting from the nose tip to show head orientation.

5️⃣ Attention Logic

If gaze ≠ center or head pose deviates → mark as distracted

If time_since_last_attention > 3s → trigger warning alert

Aggregates per-frame data to compute Class Attention %

</details>
⚠️ Notes & Recommendations

Ensure good lighting and frontal face visibility.

Adjust thresholds (e.g. SMILE_THRESH, attention_threshold) for your use case.

Works best in 720p mode for faster inference.

For multi-camera setups, modify cv2.VideoCapture(index).

<details> <summary>🚀 <b>Future Enhancements</b></summary>

 Add emotion recognition using DeepFace or FER

 Implement blink/drowsiness detection

 Integrate Gaze360 or ETH-XGaze datasets for improved gaze accuracy

 Build Streamlit/Flask dashboard for analytics visualization

 Store logs in a database (SQLite, Firebase)

 Add audio-based attention cues or voice detection

</details>
👨‍💻 Author

ENAN
🔬 Real-Time Computer Vision & Deep Learning Enthusiast
📧 [Optional: Add your GitHub or email link here]

📜 License

This project is open-source under the MIT License.
You may freely use, modify, and distribute with attribution.

⭐ If you find this useful, please star the repository and share it!
Let's make classrooms smarter through AI 👁️‍🧠
