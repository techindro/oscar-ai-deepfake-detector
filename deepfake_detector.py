# backend/deepfake_detector.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import mediapipe as mp
from PIL import Image, ImageFilter
import imageio
from scipy import signal
import hashlib
import exifread
from datetime import datetime
import json
import random

class DeepfakeDetector:
    def __init__(self):
        self.model = self.load_model()
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.5
        )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
    def load_model(self):
        """Load or create deepfake detection model"""
        # In production, you would load a pre-trained model
        # For this example, we create a simple model structure
        try:
            # Try to load pre-trained weights
            model = tf.keras.models.load_model('models/deepfake_model.h5')
            print("Loaded pre-trained model")
        except:
            # Create a simple CNN model for demonstration
            print("Creating new model for demonstration")
            input_shape = (256, 256, 3)
            inputs = Input(shape=input_shape)
            
            # CNN layers
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            x = MaxPooling2D((2, 2))(x)
            x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2))(x)
            x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2))(x)
            x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2))(x)
            
            # Dense layers
            x = Flatten()(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.5)(x)
            
            # Output layer
            outputs = Dense(1, activation='sigmoid')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', 
                         loss='binary_crossentropy', 
                         metrics=['accuracy'])
        
        return model
    
    def detect_deepfake(self, file_path, subject_type='human'):
        """Main deepfake detection method"""
        print(f"Analyzing {file_path} for {subject_type} deepfake...")
        
        # Initialize results dictionary
        results = {
            'anomalies': [],
            'inconsistency_score': 0,
            'confidence': 0,
            'detection_methods': []
        }
        
        try:
            # Check file type
            if file_path.lower().endswith(('.mp4', '.mov', '.avi', '.webm')):
                return self.analyze_video(file_path, subject_type)
            else:
                return self.analyze_image(file_path, subject_type)
                
        except Exception as e:
            print(f"Error in deepfake detection: {e}")
            results['error'] = str(e)
            return results
    
    def analyze_image(self, image_path, subject_type):
        """Analyze single image for deepfake indicators"""
        results = {
            'subject_type': subject_type,
            'analysis_type': 'image',
            'anomalies': [],
            'inconsistency_score': 0,
            'confidence': 0,
            'detection_methods': []
        }
        
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not load image")
            
            # Convert to RGB for processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 1. Error Level Analysis (ELA)
            ela_score = self.perform_ela_analysis(image_path)
            if ela_score > 0.7:
                results['anomalies'].append('High ELA score indicates potential manipulation')
                results['inconsistency_score'] += 2
            
            # 2. Face detection and analysis (if human)
            if subject_type == 'human':
                face_analysis = self.analyze_faces(img_rgb)
                results.update(face_analysis)
                results['detection_methods'].append('face_analysis')
            
            # 3. Metadata analysis
            metadata_analysis = self.analyze_metadata(image_path)
            if metadata_analysis.get('suspicious', False):
                results['anomalies'].append('Metadata inconsistencies detected')
                results['inconsistency_score'] += 1
            
            # 4. Noise pattern analysis
            noise_analysis = self.analyze_noise_patterns(img)
            results.update(noise_analysis)
            
            # 5. Color histogram analysis
            color_analysis = self.analyze_color_histograms(img)
            results.update(color_analysis)
            
            # 6. Double JPEG compression detection
            jpeg_analysis = self.detect_double_compression(image_path)
            if jpeg_analysis.get('double_compressed', False):
                results['anomalies'].append('Double JPEG compression detected')
                results['inconsistency_score'] += 1.5
            
            # 7. CNN-based deepfake detection
            cnn_prediction = self.cnn_deepfake_detection(img)
            results['cnn_confidence'] = cnn_prediction
            
            # Calculate overall inconsistency score (0-10)
            results['inconsistency_score'] = min(10, results['inconsistency_score'] + 
                                                cnn_prediction * 5 + 
                                                noise_analysis.get('noise_inconsistency', 0))
            
            # Calculate confidence (0-100%)
            results['confidence'] = min(100, results['inconsistency_score'] * 10 + 
                                       random.uniform(-5, 5))
            
            # Add timestamp
            results['analysis_timestamp'] = datetime.now().isoformat()
            
            return results
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            results['error'] = str(e)
            return results
    
    def analyze_video(self, video_path, subject_type):
        """Analyze video for deepfake indicators"""
        results = {
            'subject_type': subject_type,
            'analysis_type': 'video',
            'anomalies': [],
            'inconsistency_score': 0,
            'confidence': 0,
            'detection_methods': [],
            'frame_analysis': []
        }
        
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            results['video_properties'] = {
                'fps': fps,
                'frame_count': frame_count,
                'duration_seconds': duration,
                'resolution': f"{int(cap.get(3))}x{int(cap.get(4))}"
            }
            
            # Sample frames for analysis
            sample_rate = max(1, frame_count // 30)  # Analyze ~30 frames
            frame_idx = 0
            analyzed_frames = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_rate == 0:
                    # Analyze this frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Face analysis for human subjects
                    if subject_type == 'human':
                        face_results = self.analyze_faces(frame_rgb)
                        
                        # Track face consistency across frames
                        if face_results.get('faces_detected', 0) > 0:
                            results['frame_analysis'].append({
                                'frame': frame_idx,
                                'face_detected': True,
                                'landmark_consistency': face_results.get('landmark_consistency', 0),
                                'blink_detected': face_results.get('blink_detected', False)
                            })
                    
                    analyzed_frames += 1
                
                frame_idx += 1
                
                # Limit analysis for performance
                if analyzed_frames >= 30:
                    break
            
            cap.release()
            
            # Analyze temporal consistency
            if results['frame_analysis']:
                temporal_analysis = self.analyze_temporal_consistency(results['frame_analysis'])
                results.update(temporal_analysis)
            
            # Audio-visual sync analysis (if audio exists)
            av_sync = self.analyze_av_sync(video_path)
            if av_sync.get('sync_issue', False):
                results['anomalies'].append('Audio-visual sync inconsistency detected')
                results['inconsistency_score'] += 2
            
            # Calculate overall metrics
            if results['frame_analysis']:
                avg_consistency = np.mean([f.get('landmark_consistency', 0) 
                                          for f in results['frame_analysis'] 
                                          if f.get('face_detected', False)])
                results['average_face_consistency'] = avg_consistency
                results['inconsistency_score'] += (1 - avg_consistency) * 5
            
            # Calculate confidence
            results['confidence'] = min(100, results['inconsistency_score'] * 8 + 
                                       random.uniform(-5, 5))
            
            results['detection_methods'].extend(['temporal_analysis', 'av_sync', 'frame_sampling'])
            results['analysis_timestamp'] = datetime.now().isoformat()
            
            return results
            
        except Exception as e:
            print(f"Error analyzing video: {e}")
            results['error'] = str(e)
            return results
    
    def analyze_faces(self, image_rgb):
        """Analyze faces in image for inconsistencies"""
        results = {
            'faces_detected': 0,
            'face_anomalies': [],
            'landmark_consistency': 1.0,
            'blink_detected': False
        }
        
        try:
            # Face detection
            face_results = self.face_detector.process(image_rgb)
            
            if face_results.detections:
                results['faces_detected'] = len(face_results.detections)
                
                # Face mesh for detailed analysis
                mesh_results = self.face_mesh.process(image_rgb)
                
                if mesh_results.multi_face_landmarks:
                    landmarks = mesh_results.multi_face_landmarks[0]
                    
                    # Analyze facial symmetry
                    symmetry_score = self.analyze_facial_symmetry(landmarks)
                    if symmetry_score < 0.85:
                        results['face_anomalies'].append('Low facial symmetry detected')
                        results['landmark_consistency'] *= 0.8
                    
                    # Analyze eye consistency
                    eye_analysis = self.analyze_eyes(landmarks)
                    if not eye_analysis['consistent']:
                        results['face_anomalies'].append('Eye inconsistency detected')
                        results['landmark_consistency'] *= 0.7
                    
                    # Check for unnatural blinking (for videos)
                    results['blink_detected'] = eye_analysis['blink_detected']
                    
                    # Analyze mouth and lip consistency
                    mouth_analysis = self.analyze_mouth(landmarks)
                    if not mouth_analysis['consistent']:
                        results['face_anomalies'].append('Mouth/lip inconsistency detected')
                        results['landmark_consistency'] *= 0.75
                
                # Check multiple faces for consistency
                if len(face_results.detections) > 1:
                    face_sizes = []
                    for detection in face_results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        face_sizes.append(bbox.width * bbox.height)
                    
                    # Check if faces have consistent sizes (unnatural if not)
                    size_variance = np.var(face_sizes)
                    if size_variance > 0.01:
                        results['face_anomalies'].append('Inconsistent face sizes detected')
                        results['landmark_consistency'] *= 0.6
            
            return results
            
        except Exception as e:
            print(f"Error in face analysis: {e}")
            return results
    
    def analyze_facial_symmetry(self, landmarks):
        """Calculate facial symmetry score"""
        try:
            # Get left and right eye landmarks
            left_eye_indices = [33, 133, 157, 158, 159, 160, 161, 173]
            right_eye_indices = [362, 263, 386, 387, 388, 389, 390, 466]
            
            left_eye_points = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) 
                                        for i in left_eye_indices])
            right_eye_points = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) 
                                         for i in right_eye_indices])
            
            # Mirror right points for comparison
            right_eye_points_mirrored = right_eye_points.copy()
            right_eye_points_mirrored[:, 0] = 1 - right_eye_points_mirrored[:, 0]
            
            # Calculate symmetry score (1.0 = perfect symmetry)
            symmetry_score = 1.0 - np.mean(np.abs(left_eye_points - right_eye_points_mirrored))
            
            return max(0, min(1, symmetry_score))
            
        except:
            return 0.9  # Default score
    
    def analyze_eyes(self, landmarks):
        """Analyze eye consistency and blinking"""
        results = {
            'consistent': True,
            'blink_detected': False
        }
        
        try:
            # Eye aspect ratio calculation
            def eye_aspect_ratio(eye_points):
                # Vertical distances
                A = np.linalg.norm(eye_points[1] - eye_points[5])
                B = np.linalg.norm(eye_points[2] - eye_points[4])
                
                # Horizontal distance
                C = np.linalg.norm(eye_points[0] - eye_points[3])
                
                # Eye aspect ratio
                ear = (A + B) / (2.0 * C)
                return ear
            
            # Left eye landmarks
            left_eye_indices = [33, 133, 157, 158, 159, 160, 161, 173]
            left_eye_points = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) 
                                        for i in left_eye_indices[:6]])
            left_ear = eye_aspect_ratio(left_eye_points)
            
            # Right eye landmarks
            right_eye_indices = [362, 263, 386, 387, 388, 389, 390, 466]
            right_eye_points = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) 
                                         for i in right_eye_indices[:6]])
            right_ear = eye_aspect_ratio(right_eye_points)
            
            # Check consistency between eyes
            ear_difference = abs(left_ear - right_ear)
            if ear_difference > 0.2:
                results['consistent'] = False
            
            # Check for blink (closed eyes)
            avg_ear = (left_ear + right_ear) / 2.0
            if avg_ear < 0.2:
                results['blink_detected'] = True
            
            return results
            
        except:
            return results
    
    def analyze_mouth(self, landmarks):
        """Analyze mouth and lip consistency"""
        results = {'consistent': True}
        
        try:
            # Mouth outer landmarks
            mouth_outer_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
            mouth_points = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) 
                                     for i in mouth_outer_indices])
            
            # Calculate mouth symmetry
            left_points = mouth_points[:5]
            right_points = mouth_points[5:10]
            right_points_mirrored = right_points.copy()
            right_points_mirrored[:, 0] = 1 - right_points_mirrored[:, 0]
            
            symmetry_score = 1.0 - np.mean(np.abs(left_points - right_points_mirrored))
            
            if symmetry_score < 0.8:
                results['consistent'] = False
            
            return results
            
        except:
            return results
    
    def perform_ela_analysis(self, image_path):
        """Perform Error Level Analysis"""
        try:
            # Read and compress image
            original = Image.open(image_path).convert('RGB')
            
            # Save with compression
            temp_path = 'temp_compressed.jpg'
            original.save(temp_path, 'JPEG', quality=90)
            
            # Read compressed image
            compressed = Image.open(temp_path)
            
            # Convert to arrays
            orig_array = np.array(original).astype(np.float32)
            comp_array = np.array(compressed).astype(np.float32)
            
            # Calculate difference
            diff = np.abs(orig_array - comp_array)
            ela_score = np.mean(diff) / 255.0
            
            # Clean up
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return ela_score
            
        except Exception as e:
            print(f"ELA analysis error: {e}")
            return 0.0
    
    def analyze_metadata(self, file_path):
        """Analyze image metadata for inconsistencies"""
        results = {
            'suspicious': False,
            'issues': []
        }
        
        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
                
                # Check for common editing software tags
                editing_software = []
                for tag in tags:
                    tag_str = str(tag)
                    if 'software' in tag_str.lower() or 'creator' in tag_str.lower():
                        software = str(tags[tag])
                        editing_software.append(software)
                        
                        # Check for known editing software
                        suspicious_software = ['photoshop', 'gimp', 'after effects', 'premiere', 'davinci']
                        for sus in suspicious_software:
                            if sus in software.lower():
                                results['issues'].append(f'Edited with: {software}')
                                results['suspicious'] = True
                
                # Check date/time inconsistencies
                if 'EXIF DateTimeOriginal' in tags and 'EXIF DateTimeDigitized' in tags:
                    original = str(tags['EXIF DateTimeOriginal'])
                    digitized = str(tags['EXIF DateTimeDigitized'])
                    if original != digitized:
                        results['issues'].append('DateTimeOriginal != DateTimeDigitized')
                        results['suspicious'] = True
                
                # Check GPS data
                has_gps = any('gps' in str(tag).lower() for tag in tags)
                if has_gps:
                    # GPS data present - could be verified
                    results['has_gps'] = True
                else:
                    # No GPS data - not suspicious but noted
                    results['no_gps'] = True
                
                results['tags_found'] = len(tags)
                results['editing_software'] = editing_software
                
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def analyze_noise_patterns(self, image):
        """Analyze noise patterns for inconsistencies"""
        results = {
            'noise_consistent': True,
            'noise_inconsistency': 0.0
        }
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate noise variance in different regions
            height, width = gray.shape
            regions = []
            
            # Divide image into 4x4 grid
            grid_size = 4
            cell_h = height // grid_size
            cell_w = width // grid_size
            
            for i in range(grid_size):
                for j in range(grid_size):
                    y1 = i * cell_h
                    y2 = min((i + 1) * cell_h, height)
                    x1 = j * cell_w
                    x2 = min((j + 1) * cell_w, width)
                    
                    region = gray[y1:y2, x1:x2]
                    if region.size > 0:
                        # Calculate noise variance (using Laplacian)
                        variance = cv2.Laplacian(region, cv2.CV_64F).var()
                        regions.append(variance)
            
            # Check variance consistency across regions
            if len(regions) > 1:
                mean_var = np.mean(regions)
                std_var = np.std(regions)
                
                # High standard deviation indicates inconsistent noise
                if std_var > mean_var * 0.5:
                    results['noise_consistent'] = False
                    results['noise_inconsistency'] = min(1.0, std_var / mean_var)
        
        except Exception as e:
            print(f"Noise analysis error: {e}")
        
        return results
    
    def analyze_color_histograms(self, image):
        """Analyze color histograms for inconsistencies"""
        results = {
            'color_consistent': True,
            'histogram_analysis': {}
        }
        
        try:
            # Split channels
            channels = cv2.split(image)
            color_names = ['blue', 'green', 'red']
            
            for i, (channel, name) in enumerate(zip(channels, color_names)):
                # Calculate histogram
                hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
                hist = hist.flatten()
                
                # Normalize
                hist = hist / hist.sum() if hist.sum() > 0 else hist
                
                # Calculate statistics
                mean = np.mean(hist)
                std = np.std(hist)
                skew = np.mean((hist - mean) ** 3) / (std ** 3) if std > 0 else 0
                
                results['histogram_analysis'][name] = {
                    'mean': float(mean),
                    'std': float(std),
                    'skew': float(skew)
                }
            
            # Compare channel histograms
            if 'blue' in results['histogram_analysis'] and 'green' in results['histogram_analysis']:
                blue_skew = results['histogram_analysis']['blue']['skew']
                green_skew = results['histogram_analysis']['green']['skew']
                
                if abs(blue_skew - green_skew) > 0.3:
                    results['color_consistent'] = False
        
        except Exception as e:
            print(f"Color histogram error: {e}")
        
        return results
    
    def detect_double_compression(self, image_path):
        """Detect double JPEG compression artifacts"""
        results = {
            'double_compressed': False,
            'confidence': 0.0
        }
        
        try:
            # Read image as bytes
            with open(image_path, 'rb') as f:
                data = f.read()
            
            # Simple heuristic: check for multiple JFIF markers
            jfif_count = data.count(b'JFIF')
            if jfif_count > 1:
                results['double_compressed'] = True
                results['confidence'] = min(1.0, (jfif_count - 1) * 0.5)
        
        except:
            pass
        
        return results
    
    def cnn_deepfake_detection(self, image):
        """CNN-based deepfake detection"""
        try:
            # Preprocess image for CNN
            img_resized = cv2.resize(image, (256, 256))
            img_normalized = img_resized / 255.0
            img_expanded = np.expand_dims(img_normalized, axis=0)
            
            # Predict (using mock model for this example)
            # In production, this would use the actual trained model
            prediction = random.uniform(0.1, 0.9)  # Mock prediction
            
            return float(prediction)
            
        except Exception as e:
            print(f"CNN detection error: {e}")
            return 0.5
    
    def analyze_temporal_consistency(self, frame_analysis):
        """Analyze temporal consistency in video"""
        results = {
            'temporal_inconsistency': 0.0,
            'frame_jumps': 0
        }
        
        try:
            if len(frame_analysis) < 2:
                return results
            
            # Extract landmark consistencies
            consistencies = [f.get('landmark_consistency', 0.9) 
                            for f in frame_analysis if f.get('face_detected', False)]
            
            if len(consistencies) > 1:
                # Calculate variation in consistency
                mean_consistency = np.mean(consistencies)
                std_consistency = np.std(consistencies)
                
                results['temporal_inconsistency'] = min(1.0, std_consistency * 2)
                
                # Count significant drops in consistency
                jumps = 0
                for i in range(1, len(consistencies)):
                    if abs(consistencies[i] - consistencies[i-1]) > 0.3:
                        jumps += 1
                
                results['frame_jumps'] = jumps
        
        except Exception as e:
            print(f"Temporal analysis error: {e}")
        
        return results
    
    def analyze_av_sync(self, video_path):
        """Analyze audio-visual synchronization"""
        results = {
            'sync_issue': False,
            'sync_offset': 0.0
        }
        
        try:
            # Mock analysis - in production, use actual audio analysis
            # For now, simulate occasional sync issues
            if random.random() < 0.2:  # 20% chance of sync issue
                results['sync_issue'] = True
                results['sync_offset'] = random.uniform(0.1, 0.5)
        
        except:
            pass
        
        return results
    
    def generate_visualizations(self, file_path, analysis_results):
        """Generate visualization images for analysis results"""
        visualizations = {}
        
        try:
            # Generate heatmap (mock for this example)
            if analysis_results.get('analysis_type') == 'image':
                img = cv2.imread(file_path)
                if img is not None:
                    # Create a simple heatmap
                    heatmap = np.zeros(img.shape[:2], dtype=np.float32)
                    
                    # Add random "anomaly" spots for demonstration
                    height, width = heatmap.shape
                    for _ in range(10):
                        y = np.random.randint(0, height)
                        x = np.random.randint(0, width)
                        radius = np.random.randint(10, 50)
                        cv2.circle(heatmap, (x, y), radius, 0.5, -1)
                    
                    # Normalize and convert to color
                    heatmap = (heatmap * 255).astype(np.uint8)
                    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    
                    # Save heatmap
                    heatmap_path = file_path.replace('.', '_heatmap.')
                    cv2.imwrite(heatmap_path, heatmap_color)
                    
                    visualizations['heatmap'] = heatmap_path
            
            # Generate ELA visualization
            if analysis_results.get('ela_score', 0) > 0:
                ela_viz = self.generate_ela_visualization(file_path)
                if ela_viz:
                    visualizations['ela'] = ela_viz
        
        except Exception as e:
            print(f"Visualization generation error: {e}")
        
        return visualizations
    
    def generate_ela_visualization(self, image_path):
        """Generate ELA visualization image"""
        try:
            # Load image
            original = cv2.imread(image_path)
            if original is None:
                return None
            
            # Save with compression
            temp_path = 'temp_ela.jpg'
            cv2.imwrite(temp_path, original, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # Load compressed
            compressed = cv2.imread(temp_path)
            
            # Calculate difference
            diff = cv2.absdiff(original, compressed)
            
            # Enhance difference for visualization
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            diff = cv2.equalizeHist(diff)
            
            # Save visualization
            ela_path = image_path.replace('.', '_ela.')
            cv2.imwrite(ela_path, diff)
            
            # Clean up
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return ela_path
            
        except:
            return None
    
    def analyze_file_structure(self, file_path):
        """Analyze file structure and format"""
        results = {
            'file_integrity': 'valid',
            'format_analysis': {},
            'compression_analysis': {}
        }
        
        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            results['file_size_bytes'] = file_size
            results['file_size_mb'] = round(file_size / (1024 * 1024), 2)
            
            # Check file extension vs actual format
            import imghdr
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                actual_format = imghdr.what(file_path)
                results['format_analysis']['actual_format'] = actual_format
                
                if actual_format and file_path.lower().endswith(f'.{actual_format}'):
                    results['format_analysis']['extension_match'] = True
                else:
                    results['format_analysis']['extension_match'] = False
                    results['file_integrity'] = 'suspicious'
            
            # Check for embedded data or appended content
            with open(file_path, 'rb') as f:
                data = f.read()
                
                # Look for multiple file signatures
                signatures = {
                    b'\xff\xd8\xff': 'JPEG',
                    b'\x89PNG': 'PNG',
                    b'GIF': 'GIF',
                    b'RIFF': 'AVI/WAV',
                    b'\x00\x00\x00 ftyp': 'MP4'
                }
                
                found_signatures = []
                for sig, name in signatures.items():
                    if sig in data:
                        count = data.count(sig)
                        if count > 1:
                            found_signatures.append(f"{name} (x{count})")
                
                if found_signatures:
                    results['compression_analysis']['multiple_signatures'] = found_signatures
                    results['file_integrity'] = 'suspicious'
        
        except Exception as e:
            results['error'] = str(e)
        
        return results
