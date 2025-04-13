# Face-Recognition-based-Attendance-System-using-Open-CV
This project introduces an automated attendance system using face recognition to replace manual tracking in institutions. It uses OpenCV and a Logitech webcam to capture student images and match them with a pre-trained dataset, ensuring accurate, efficient attendance and better time management.
This project implements an automated attendance system using face recognition technology to replace traditional manual methods in educational institutions. By leveraging OpenCV, Flask, and machine learning algorithms, the system enables real-time attendance tracking through facial identification, improving efficiency, accuracy, and security.

🎯 Objectives
Detect and recognize student faces from a webcam feed.

Automatically record attendance in an organized format.

Eliminate fraudulent attendance and reduce manual work.

🧠 Technologies Used
OpenCV – For face detection and image processing.

Flask – Lightweight Python web framework for backend logic and UI.

Sklearn – For training and predicting face embeddings using KNN.

Joblib – To serialize the trained model.

HTML/CSS + Bootstrap – For frontend user interfaces.

📦 Features
📷 Real-time face detection and recognition using webcam.

📝 Attendance is auto-logged into CSV files.

👨‍🏫 Admin dashboard for user management and analytics.

🔒 Secure login system with role-based access (Admin/User).

📊 Visual statistics and Excel export of attendance data.

🏗️ System Modules
Face Detection Module – Detects faces using HaarCascade.

Face Recognition Module – Recognizes users using trained embeddings.

User Registration Module – Captures and stores user face data.

Model Training Module – Trains KNN classifier with user images.

Webcam Module – Captures and processes live video feed.

Statistics Module – Displays attendance analytics and graphs.

📁 Project Structure
plaintext
Copy
Edit
.
├── static/
│   └── faces/               # Stored face data for each user
├── Attendance/              # Daily attendance logs (CSV)
├── templates/               # HTML files (Flask templates)
├── app.py                   # Main Flask application
├── users.csv                # User database
└── README.md                # Project overview and setup
🚀 Getting Started
🔧 Prerequisites
Python 3.x

Webcam

Required libraries: Install via pip

bash
Copy
Edit
pip install opencv-python flask pandas scikit-learn joblib python-dotenv
▶️ Run the Application
bash
Copy
Edit
python app.py
Open a browser and visit: http://127.0.0.1:5000

👩‍💼 Roles
Admin can:

Add/delete users

View reports

Train models

Users can:

Log in and view their attendance stats

🧪 Performance
Accuracy depends on lighting, distance from the camera, and face orientation.

Best results achieved within 1–2 feet under stable lighting.

Confidence threshold set to 70% for reliable recognition.

📌 Future Improvements
Mask detection integration.

Attendance alerts via email/SMS.

Face recognition via mobile camera.

📚 References
OpenCV Documentation

Flask Documentation

Scikit-learn User Guide

