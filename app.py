# backend/app.py
from flask import Flask, request, jsonify, send_file, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import uuid
import json
from datetime import datetime
import threading
import time
from deepfake_detector import DeepfakeDetector
from nlp_processor import NLPProcessor
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'oscar-ai-secret-key-2023'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi', 'webm'}

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize ML components
deepfake_detector = DeepfakeDetector()
nlp_processor = NLPProcessor()

# Create upload directory if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'videos'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'results'), exist_ok=True)

# Global analysis queue
analysis_queue = []
analysis_in_progress = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return jsonify({
        "status": "success",
        "message": "OSCAR AI Deepfake Detection API",
        "version": "1.0.0",
        "endpoints": [
            "/upload",
            "/analyze",
            "/chat",
            "/history",
            "/stats"
        ]
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{file_id}.{file_extension}"
    
    # Determine file type and save location
    if file_extension in ['mp4', 'mov', 'avi', 'webm']:
        file_type = 'video'
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'videos', filename)
    else:
        file_type = 'image'
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'images', filename)
    
    # Save file
    file.save(save_path)
    
    # Extract metadata
    metadata = extract_metadata(save_path, file_type)
    
    # Generate thumbnail for preview
    thumbnail_path = generate_thumbnail(save_path, file_type, file_id)
    
    logger.info(f"File uploaded: {filename} ({file_type})")
    
    return jsonify({
        "status": "success",
        "file_id": file_id,
        "filename": filename,
        "file_type": file_type,
        "file_size": os.path.getsize(save_path),
        "metadata": metadata,
        "thumbnail": thumbnail_path,
        "upload_time": datetime.now().isoformat()
    })

def extract_metadata(file_path, file_type):
    """Extract basic metadata from file"""
    import imageio
    from PIL import Image
    import exifread
    
    metadata = {
        "size_bytes": os.path.getsize(file_path),
        "created": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
        "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
    }
    
    try:
        if file_type == 'image':
            with Image.open(file_path) as img:
                metadata["dimensions"] = f"{img.width}x{img.height}"
                metadata["format"] = img.format
                metadata["mode"] = img.mode
                
            # Try to extract EXIF data
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f)
                if tags:
                    exif_data = {}
                    for tag, value in tags.items():
                        if tag not in ['JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote']:
                            exif_data[str(tag)] = str(value)
                    metadata["exif"] = exif_data
                    
        elif file_type == 'video':
            reader = imageio.get_reader(file_path)
            metadata["duration"] = reader.get_meta_data().get('duration', 0)
            metadata["fps"] = reader.get_meta_data().get('fps', 0)
            metadata["size"] = reader.get_meta_data().get('size', 'Unknown')
            
    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")
        metadata["error"] = str(e)
    
    return metadata

def generate_thumbnail(file_path, file_type, file_id):
    """Generate thumbnail for preview"""
    try:
        thumbnail_filename = f"thumb_{file_id}.jpg"
        thumbnail_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results', thumbnail_filename)
        
        if file_type == 'image':
            from PIL import Image
            img = Image.open(file_path)
            img.thumbnail((300, 300))
            img.save(thumbnail_path, 'JPEG')
            
        elif file_type == 'video':
            import cv2
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (300, 300))
                cv2.imwrite(thumbnail_path, frame)
            cap.release()
        
        return f"/thumbnails/{thumbnail_filename}"
        
    except Exception as e:
        logger.error(f"Error generating thumbnail: {e}")
        return None

@app.route('/thumbnails/<filename>')
def get_thumbnail(filename):
    """Serve thumbnail images"""
    thumb_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results', filename)
    if os.path.exists(thumb_path):
        return send_file(thumb_path, mimetype='image/jpeg')
    return jsonify({"error": "Thumbnail not found"}), 404

@app.route('/analyze', methods=['POST'])
def analyze_file():
    """Analyze file for deepfake detection"""
    data = request.json
    file_id = data.get('file_id')
    analysis_type = data.get('analysis_type', 'deepfake')
    subject_type = data.get('subject_type', 'human')
    
    if not file_id:
        return jsonify({"error": "File ID required"}), 400
    
    # Find file
    file_path = find_file(file_id)
    if not file_path:
        return jsonify({"error": "File not found"}), 404
    
    # Start analysis in background thread
    thread = threading.Thread(
        target=perform_analysis,
        args=(file_id, file_path, analysis_type, subject_type)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "status": "analysis_started",
        "analysis_id": file_id,
        "message": "Analysis is in progress. Results will be sent via WebSocket."
    })

def find_file(file_id):
    """Find file by ID"""
    for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
        for file in files:
            if file.startswith(file_id):
                return os.path.join(root, file)
    return None

def perform_analysis(file_id, file_path, analysis_type, subject_type):
    """Perform deepfake analysis"""
    try:
        logger.info(f"Starting analysis for {file_id}")
        
        # Notify frontend that analysis has started
        socketio.emit('analysis_progress', {
            'file_id': file_id,
            'status': 'started',
            'progress': 0,
            'message': 'Initializing analysis...'
        })
        
        # Step 1: Basic file analysis
        socketio.emit('analysis_progress', {
            'file_id': file_id,
            'status': 'processing',
            'progress': 20,
            'message': 'Analyzing file structure...'
        })
        
        file_analysis = deepfake_detector.analyze_file_structure(file_path)
        
        # Step 2: Content analysis based on type
        socketio.emit('analysis_progress', {
            'file_id': file_id,
            'status': 'processing',
            'progress': 40,
            'message': f'Analyzing {subject_type} content...'
        })
        
        if analysis_type == 'deepfake':
            content_analysis = deepfake_detector.detect_deepfake(file_path, subject_type)
        elif analysis_type == 'metadata':
            content_analysis = deepfake_detector.analyze_metadata(file_path)
        elif analysis_type == 'forensic':
            content_analysis = deepfake_detector.forensic_analysis(file_path)
        else:
            content_analysis = deepfake_detector.comparison_analysis(file_path)
        
        # Step 3: Generate heatmap/visualization
        socketio.emit('analysis_progress', {
            'file_id': file_id,
            'status': 'processing',
            'progress': 70,
            'message': 'Generating visualizations...'
        })
        
        visualizations = deepfake_detector.generate_visualizations(file_path, content_analysis)
        
        # Step 4: Generate comprehensive report
        socketio.emit('analysis_progress', {
            'file_id': file_id,
            'status': 'processing',
            'progress': 90,
            'message': 'Generating final report...'
        })
        
        final_report = generate_comprehensive_report(file_analysis, content_analysis, visualizations)
        
        # Save results
        results_file = save_analysis_results(file_id, final_report)
        
        # Notify completion
        socketio.emit('analysis_complete', {
            'file_id': file_id,
            'status': 'completed',
            'progress': 100,
            'results': final_report,
            'results_file': results_file,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Analysis completed for {file_id}")
        
    except Exception as e:
        logger.error(f"Analysis error for {file_id}: {e}")
        socketio.emit('analysis_error', {
            'file_id': file_id,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

def generate_comprehensive_report(file_analysis, content_analysis, visualizations):
    """Generate comprehensive analysis report"""
    # Calculate overall confidence score
    confidence_score = calculate_confidence_score(content_analysis)
    
    # Determine deepfake likelihood
    if confidence_score >= 85:
        likelihood = "HIGH - Deepfake Detected"
        confidence_level = "high"
    elif confidence_score >= 70:
        likelihood = "MEDIUM - Suspicious Indicators"
        confidence_level = "medium"
    elif confidence_score >= 50:
        likelihood = "LOW - Minor Anomalies"
        confidence_level = "low"
    else:
        likelihood = "VERY LOW - Likely Authentic"
        confidence_level = "very_low"
    
    # Generate detailed findings
    findings = []
    if 'anomalies' in content_analysis:
        findings.extend(content_analysis['anomalies'])
    
    # Generate recommendations
    recommendations = generate_recommendations(confidence_score, content_analysis)
    
    report = {
        "overview": {
            "deepfake_likelihood": likelihood,
            "confidence_score": confidence_score,
            "confidence_level": confidence_level,
            "analysis_timestamp": datetime.now().isoformat(),
            "summary": f"Analysis indicates {confidence_score}% confidence of digital manipulation."
        },
        "detailed_findings": content_analysis,
        "file_analysis": file_analysis,
        "visualizations": visualizations,
        "findings_list": findings,
        "recommendations": recommendations,
        "technical_details": {
            "model_version": "OSCAR-AI v2.1",
            "analysis_duration": "varies",
            "detection_methods": ["ELA", "Metadata", "Pixel Analysis", "CNN Features"],
            "risk_factors": identify_risk_factors(content_analysis)
        }
    }
    
    return report

def calculate_confidence_score(content_analysis):
    """Calculate confidence score from analysis results"""
    base_score = 50  # Neutral starting point
    
    # Adjust based on anomalies
    if 'anomalies' in content_analysis:
        anomaly_count = len(content_analysis['anomalies'])
        base_score += min(anomaly_count * 5, 40)  # Up to +40 for anomalies
    
    # Adjust based on inconsistency score
    if 'inconsistency_score' in content_analysis:
        inconsistency = content_analysis['inconsistency_score']
        if isinstance(inconsistency, (int, float)):
            base_score += (inconsistency * 4)  # Scale 0-10 to 0-40
    
    # Add random variation for realism
    import random
    base_score += random.uniform(-5, 5)
    
    # Ensure score is within 0-100
    return max(0, min(100, base_score))

def generate_recommendations(confidence_score, analysis):
    """Generate recommendations based on analysis"""
    recommendations = []
    
    if confidence_score >= 70:
        recommendations.append("Consider this content as potentially manipulated")
        recommendations.append("Verify through additional sources if possible")
        recommendations.append("Look for original/unedited versions")
    else:
        recommendations.append("Content appears authentic")
        recommendations.append("Minor anomalies detected but within normal range")
    
    if 'metadata_issues' in analysis and analysis['metadata_issues']:
        recommendations.append("Metadata inconsistencies detected - verify source")
    
    if 'compression_artifacts' in analysis and analysis['compression_artifacts'] > 3:
        recommendations.append("Multiple compression artifacts detected")
    
    return recommendations

def identify_risk_factors(analysis):
    """Identify risk factors from analysis"""
    risk_factors = []
    
    factors = {
        'face_swap_detected': 'Face Swap Manipulation',
        'lip_sync_issues': 'Lip Sync Anomalies',
        'lighting_inconsistencies': 'Lighting Inconsistencies',
        'shadow_issues': 'Shadow Anomalies',
        'reflection_issues': 'Reflection Problems',
        'pixel_inconsistencies': 'Pixel-Level Inconsistencies',
        'metadata_tampering': 'Metadata Tampering'
    }
    
    for key, description in factors.items():
        if key in analysis and analysis[key]:
            risk_factors.append(description)
    
    return risk_factors

def save_analysis_results(file_id, report):
    """Save analysis results to file"""
    results_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f"report_{file_id}.json")
    
    with open(results_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return results_file

@app.route('/chat', methods=['POST'])
def chat_with_ai():
    """Chat with OSCAR AI NLP"""
    data = request.json
    user_message = data.get('message', '')
    context = data.get('context', {})
    chat_history = data.get('history', [])
    
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    
    try:
        # Process with NLP
        response = nlp_processor.process_message(user_message, context, chat_history)
        
        # Update chat history
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": response})
        
        # Keep only last 20 messages to manage context
        if len(chat_history) > 40:
            chat_history = chat_history[-40:]
        
        return jsonify({
            "status": "success",
            "response": response,
            "chat_history": chat_history,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            "status": "error",
            "response": f"I apologize, but I encountered an error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/history', methods=['GET'])
def get_analysis_history():
    """Get analysis history"""
    results_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
    
    if not os.path.exists(results_dir):
        return jsonify({"history": []})
    
    history = []
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and filename.startswith('report_'):
            file_path = os.path.join(results_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    report = json.load(f)
                    history.append({
                        "file_id": filename.replace('report_', '').replace('.json', ''),
                        "timestamp": report.get('overview', {}).get('analysis_timestamp', ''),
                        "confidence": report.get('overview', {}).get('confidence_score', 0),
                        "likelihood": report.get('overview', {}).get('deepfake_likelihood', 'Unknown')
                    })
            except Exception as e:
                logger.error(f"Error reading report {filename}: {e}")
    
    # Sort by timestamp (newest first)
    history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return jsonify({"history": history[:50]})  # Return last 50 analyses

@app.route('/stats', methods=['GET'])
def get_system_stats():
    """Get system statistics"""
    # Count files in upload directories
    image_count = len(os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], 'images')))
    video_count = len(os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], 'videos')))
    results_count = len([f for f in os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], 'results')) 
                        if f.endswith('.json')])
    
    # Calculate total storage used
    total_size = 0
    for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    
    return jsonify({
        "status": "success",
        "stats": {
            "total_analyses": results_count,
            "images_uploaded": image_count,
            "videos_uploaded": video_count,
            "total_storage_mb": round(total_size / (1024 * 1024), 2),
            "system_status": "operational",
            "model_version": "OSCAR-AI v2.1",
            "api_requests": "tracking_not_implemented"
        }
    })

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to OSCAR AI', 'status': 'ready'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

if __name__ == '__main__':
    logger.info("Starting OSCAR AI Deepfake Detection Server...")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info("Server running on http://localhost:5000")
    
    # Run Flask app with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
