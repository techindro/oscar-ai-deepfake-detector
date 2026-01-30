# OSCAR AI - Advanced Deepfake Detection System

A full-stack deepfake detection system combining machine learning, NLP, and real-time analysis capabilitie

## Features

- **Multi-format Support**: Analyze images (JPG, PNG) and videos (MP4, MOV, AVI)
- **Advanced Detection**: Multiple detection methods (ELA, facial analysis, metadata forensics)
- **Real-time Analysis**: WebSocket-based progress updates
- **NLP Chatbot**: Intelligent responses about deepfake technology
- **Comprehensive Reports**: Detailed analysis with confidence scores
- **Visualizations**: Heatmaps and forensic visualizations
- **History Tracking**: Save and review past analyses

## Installation

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend

# Create a virtual environment
python -m venv venv:
source venv/bin/activate  
On Windows: venv\Scripts\activate

#Install dependencies:
pip install -r requirements.txt

#Set up OpenAI API key:
"export OPENAI_API_KEY="your-api-key-here"
#On Windows: set OPENAI_API_KEY=your-api-key-here

# Run the backend server:
"python app.py
Server will start at: http://localhost:5000




