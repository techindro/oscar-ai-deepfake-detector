# backend/nlp_processor.py
import openai
import json
import re
from datetime import datetime
import random

class NLPProcessor:
    def __init__(self):
        # Initialize OpenAI (you need to set your API key)
        openai.api_key = "your-openai-api-key-here"  # Replace with actual key
        self.context_memory = {}
        self.knowledge_base = self.load_knowledge_base()
        
    def load_knowledge_base(self):
        """Load deepfake detection knowledge base"""
        return {
            "deepfake_indicators": [
                "Unnatural eye blinking or inconsistent eye reflections",
                "Facial asymmetry or inconsistent lighting on face",
                "Mismatched audio-video synchronization",
                "Inconsistent shadows or lighting direction",
                "Pixel-level inconsistencies in facial features",
                "Metadata timestamp mismatches",
                "Double JPEG compression artifacts",
                "Error Level Analysis (ELA) showing high differences",
                "Unnatural facial expressions or movements",
                "Inconsistent background details or blurring"
            ],
            "detection_methods": [
                "Error Level Analysis (ELA)",
                "Facial Landmark Analysis",
                "Metadata Forensics",
                "Noise Pattern Analysis",
                "Deep Learning CNN Models",
                "Temporal Consistency Analysis",
                "Audio-Visual Synchronization Check",
                "Compression Artifact Analysis"
            ],
            "common_tools": [
                "DeepFaceLab",
                "FaceSwap",
                "First Order Motion Model",
                "Wav2Lip",
                "StyleGAN",
                "Zao App",
                "Reface App",
                "DeepNude (now deprecated)"
            ],
            "prevention_tips": [
                "Use watermarking on original content",
                "Maintain original files with verified metadata",
                "Use blockchain for media provenance tracking",
                "Implement multi-factor authentication for sensitive media",
                "Educate about deepfake indicators",
                "Use digital signatures for official content"
            ]
        }
    
    def process_message(self, user_message, context, chat_history):
        """Process user message and generate response"""
        
        # Check for specific queries
        if self.is_greeting(user_message):
            return self.generate_greeting()
        
        if self.is_analysis_query(user_message):
            return self.generate_analysis_explanation(user_message)
        
        if self.is_technical_query(user_message):
            return self.generate_technical_response(user_message)
        
        if self.is_example_request(user_message):
            return self.provide_examples()
        
        # Default: Use knowledge base or generate creative response
        return self.generate_intelligent_response(user_message, context)
    
    def is_greeting(self, message):
        """Check if message is a greeting"""
        greetings = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        message_lower = message.lower()
        return any(greeting in message_lower for greeting in greetings)
    
    def is_analysis_query(self, message):
        """Check if message is about analysis results"""
        keywords = ['analyze', 'analysis', 'result', 'detect', 'deepfake', 'fake', 'authentic', 'real', 'fake']
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in keywords)
    
    def is_technical_query(self, message):
        """Check if message is technical"""
        technical_terms = ['ela', 'cnn', 'metadata', 'compression', 'artifact', 'algorithm', 'model', 'neural network']
        message_lower = message.lower()
        return any(term in message_lower for term in technical_terms)
    
    def is_example_request(self, message):
        """Check if user wants examples"""
        example_phrases = ['example', 'show me', 'demonstrate', 'sample', 'case study']
        message_lower = message.lower()
        return any(phrase in message_lower for phrase in example_phrases)
    
    def generate_greeting(self):
        """Generate greeting response"""
        greetings = [
            "Hello! I'm OSCAR AI, your deepfake detection assistant. How can I help you today?",
            "Hi there! I'm OSCAR AI, ready to help you understand and detect deepfakes.",
            "Welcome! I'm OSCAR AI, specialized in deepfake analysis and detection. What would you like to know?"
        ]
        return random.choice(greetings)
    
    def generate_analysis_explanation(self, user_message):
        """Generate analysis-related response"""
        # Extract keywords from user message
        keywords = {
            'face': 'Face analysis looks for inconsistencies in facial symmetry, eye reflections, and landmark consistency.',
            'video': 'Video analysis examines temporal consistency, frame-by-frame anomalies, and audio-visual synchronization.',
            'audio': 'Audio analysis checks for voice cloning artifacts, unnatural pauses, and spectral inconsistencies.',
            'metadata': 'Metadata forensics examines EXIF data, timestamps, editing software signatures, and compression history.',
            'ela': 'Error Level Analysis compares compression levels across an image to detect manipulated regions.',
            'confidence': 'Confidence scores represent the likelihood of manipulation based on multiple detection methods.'
        }
        
        for keyword, explanation in keywords.items():
            if keyword in user_message.lower():
                return explanation
        
        # Default analysis explanation
        explanations = [
            "Deepfake analysis involves multiple techniques: examining pixel-level inconsistencies, analyzing facial landmarks, checking metadata integrity, and verifying temporal consistency in videos.",
            "I analyze media using a combination of forensic techniques including Error Level Analysis, facial landmark detection, metadata examination, and machine learning models trained on authentic and manipulated content.",
            "The detection process looks for telltale signs of AI manipulation: inconsistent lighting, unnatural facial movements, compression artifacts, and metadata anomalies that don't match the content."
        ]
        
        return random.choice(explanations)
    
    def generate_technical_response(self, user_message):
        """Generate technical response"""
        technical_responses = {
            'ela': "Error Level Analysis works by re-compressing an image and comparing the error levels. Manipulated regions often show different compression artifacts than authentic areas.",
            'cnn': "Convolutional Neural Networks are trained on thousands of real and fake images to learn subtle patterns that distinguish authentic from manipulated content.",
            'metadata': "Metadata analysis examines EXIF data, GPS coordinates, timestamps, and software signatures. Inconsistencies can reveal editing history or manipulation.",
            'compression': "Compression artifacts appear when images are saved multiple times. Different compression levels across an image can indicate manipulation.",
            'face landmark': "Facial landmark analysis tracks 468 points on a face to detect inconsistencies in symmetry, expressions, and movements that might indicate manipulation."
        }
        
        for term, response in technical_responses.items():
            if term in user_message.lower():
                return response
        
        return "I use advanced machine learning models and forensic techniques to detect subtle inconsistencies that indicate digital manipulation. The system combines multiple detection methods for higher accuracy."
    
    def provide_examples(self):
        """Provide examples of deepfake detection"""
        examples = [
            "Example 1: A manipulated image might show inconsistent lighting - the face is lit from the left but shadows fall to the right.",
            "Example 2: In deepfake videos, you might notice unnatural eye blinking patterns or lips that don't perfectly sync with audio.",
            "Example 3: Metadata analysis might reveal that an image claiming to be from 2010 was actually created by Photoshop 2022.",
            "Example 4: Error Level Analysis can highlight manipulated regions showing different compression levels than the rest of the image."
        ]
        
        return random.choice(examples) + " Would you like more specific examples?"
    
    def generate_intelligent_response(self, user_message, context):
        """Generate intelligent response using context"""
        
        # Try using OpenAI API for complex queries
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are OSCAR AI, a deepfake detection expert. Provide accurate, technical but understandable responses about deepfake technology, detection methods, and media forensics. Keep responses concise and helpful."},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # Fallback to rule-based responses
            print(f"OpenAI API error: {e}")
            
            # Generate response based on knowledge base
            if 'how' in user_message.lower() and 'detect' in user_message.lower():
                methods = ', '.join(self.knowledge_base['detection_methods'][:3])
                return f"I detect deepfakes using multiple methods including {methods}. Each method looks for different types of inconsistencies that might indicate manipulation."
            
            elif 'prevent' in user_message.lower() or 'protect' in user_message.lower():
                tips = '\n- '.join(self.knowledge_base['prevention_tips'][:3])
                return f"To protect against deepfakes:\n- {tips}"
            
            elif 'indicator' in user_message.lower() or 'sign' in user_message.lower():
                indicators = '\n- '.join(self.knowledge_base['deepfake_indicators'][:5])
                return f"Common deepfake indicators include:\n- {indicators}"
            
            else:
                return "I analyze digital media for signs of manipulation using forensic techniques and machine learning. Could you be more specific about what you'd like to know?"
