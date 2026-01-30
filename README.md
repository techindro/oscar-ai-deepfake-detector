# OSCAR AI - Advanced Deepfake Detection System
<div align="center">
<img src =" https://img.shields.io/badge/OSCAR_AI-Deepfake_Detection-blue?style=for-the-badge&logo=ai&logoColor=white target ="blank">
 </div>
 
# Overview
OSCAR AI is a comprehensive, production-ready deepfake detection system that combines state-of-the-art machine learning models with natural language processing to identify manipulated media. Featuring a Labrador-themed interface, the system provides real-time analysis, detailed forensic reports, and an interactive AI chatbot for explaining detection results.a deepfake detection system that uses machine learning and NLP to detect manipulated media. It has a web interface for uploading files and getting analysis results.

# Project Structure
backend/ - Python Flask server with ML models
frontend/ - HTML/CSS/JS web interface

# installation
Backend Setup
1. Install Python 3.9 or higher
2. Navigate to backend folder:
3. Create Virtual environment:
   python -m venv venv

4. Activate virtual environment:
Windows: venv\Scripts\activate
Mac/Linux: source venv/bin/activate

5. Install dependencies:
   pip install -r requirements.txt

# Run the server:
python app.py

# Features
Upload images and videos for analysis
Multiple detection methods (Deepfake, Metadata, Forensic, Comparison)
Real-time progress updates
Chat with AI assistant about results
History of past analyses

# Usage
Upload file using drag & drop or click
Select analysis type and subject
Click "Analyze with OSCAR AI"
View results and chat with AI

# API Endpoints
POST /upload - Upload file
POST /analyze - Start analysis
POST /chat - Chat with AI
GET /history - Get analysis history
GET /stats - Get system statistics

# Requirements
Backend requires:
Python 3.9+
Flask
TensorFlow
OpenCV
MediaPipe

# License
MIT License

