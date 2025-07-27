# Blood-Detection-Project

# 🩸 Blood Group Detection Using Fingerprint

A deep learning-based project to **predict a person’s blood group using their fingerprint image**. The model leverages **EfficientNet** for classification and is deployed via a lightweight **Flask web application**.

---

## 🚀 Project Overview

Traditional blood group detection requires biochemical tests, which can be invasive and time-consuming.  
This project demonstrates an **innovative, non-invasive approach** by analyzing fingerprint patterns to determine blood group types.

---

## 🧠 Model Summary

- **Architecture:** EfficientNet (Keras)
- **Input Size:** 224 × 224 RGB images
- **Classes:** `A+`, `A−`, `AB+`, `AB−`, `B+`, `B−`, `O+`, `O−`
- **Training Data:** Grayscale fingerprint images converted to RGB
- **Preprocessing:** Standard EfficientNet preprocessing
- **Output:** Predicted blood group with confidence score

---

## 🖥️ Features

- Upload fingerprint images in **JPG, JPEG, PNG, or BMP** formats
- Get **predicted blood group** with **confidence percentage**
- Simple and lightweight **Flask-based UI**

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS (Flask Templates)  
- **Backend:** Flask (Python)  
- **Model:** Keras (EfficientNet)  
- **Others:** NumPy, OpenCV, TensorFlow

---

## 📂 Project Structure
Blood-Group-Detection/
│
├── model/ # Saved EfficientNet model
├── static/ # CSS, JS, and image files
├── templates/ # HTML templates for Flask
├── app.py # Main Flask application
├── requirements.txt # Dependencies
└── README.md # Project documentation
