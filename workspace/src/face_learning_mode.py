#!/usr/bin/env python3
"""
Face Learning Mode - Dedicated script for training and improving face recognition
This script helps collect high-quality face samples for better recognition performance
"""

import cv2
import os
import numpy as np
from ultralytics import YOLO
import face_recognition
import time
from collections import defaultdict
from typing import List, Dict, Tuple

class FaceLearningSystem:
    def __init__(self, dataset_path: str, model_path: str):
        """
        Initialize the face learning system
        
        Args:
            dataset_path: Path to store learned face images
            model_path: Path to YOLO face detection model
        """
        self.dataset_path = dataset_path
        self.model_path = model_path
        
        # Initialize YOLO model
        print("Loading YOLO face detection model...")
        self.yolo_model = YOLO(model_path)
        
        # Learning parameters
        self.quality_threshold = 0.7  # Minimum detection confidence
        self.min_face_size = 100  # Minimum face size in pixels
        self.max_samples_per_person = 50  # Maximum samples to collect
        
        # Tracking variables
        self.current_person = None
        self.collected_samples = defaultdict(int)
        self.last_capture_time = 0
        self.capture_interval = 0.5  # Seconds between captures
        
        # Create dataset directory if it doesn't exist
        os.makedirs(dataset_path, exist_ok=True)
    
    def assess_face_quality(self, face_image: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Assess the quality of a detected face for training
        
        Args:
            face_image: Cropped face image
            
        Returns:
            Tuple of (overall_score, quality_metrics)
        """
        metrics = {}
        
        # Size check
        height, width = face_image.shape[:2]
        size_score = min(1.0, min(height, width) / self.min_face_size)
        metrics['size'] = size_score
        
        # Brightness check
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        # Optimal brightness is around 0.4-0.7
        brightness_score = 1.0 - abs(brightness - 0.55) * 2
        brightness_score = max(0.0, min(1.0, brightness_score))
        metrics['brightness'] = brightness_score
        
        # Contrast check
        contrast = gray.std() / 128.0
        contrast_score = min(1.0, contrast)
        metrics['contrast'] = contrast_score
        
        # Sharpness check (using Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 1000.0)
        metrics['sharpness'] = sharpness_score
        
        # Face encoding quality check
        try:
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_face)
            encoding_score = 1.0 if len(encodings) > 0 else 0.0
            metrics['face_detected'] = encoding_score
        except:
            metrics['face_detected'] = 0.0
        
        # Calculate overall score (weighted average)
        weights = {
            'size': 0.2,
            'brightness': 0.2,
            'contrast': 0.2,
            'sharpness': 0.2,
            'face_detected': 0.2
        }
        
        overall_score = sum(metrics[key] * weights[key] for key in weights.keys())
        
        return overall_score, metrics
    
    def detect_and_assess_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces and assess their quality for training
        
        Args:
            frame: Input video frame
            
        Returns:
            List of dictionaries containing face info and quality metrics
        """
        results = self.yolo_model(frame, verbose=False)
        face_data = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    
                    if confidence > self.quality_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Extract face with padding
                        padding = 20
                        x1_pad = max(0, x1 - padding)
                        y1_pad = max(0, y1 - padding)
                        x2_pad = min(frame.shape[1], x2 + padding)
                        y2_pad = min(frame.shape[0], y2 + padding)
                        
                        face_image = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                        
                        if face_image.size > 0:
                            quality_score, metrics = self.assess_face_quality(face_image)
                            
                            face_data.append({
                                'bbox': (x1, y1, x2, y2),
                                'bbox_padded': (x1_pad, y1_pad, x2_pad, y2_pad),
                                'face_image': face_image,
                                'detection_confidence': confidence,
                                'quality_score': quality_score,
                                'quality_metrics': metrics
                            })
        
        # Sort by quality score (best first)
        face_data.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return face_data
    
    def save_face_sample(self, face_image: np.ndarray, person_name: str, quality_score: float) -> bool:
        """
        Save a face sample to the dataset
        
        Args:
            face_image: Face image to save
            person_name: Name of the person
            quality_score: Quality score of the face
            
        Returns:
            True if saved successfully
        """
        person_folder = os.path.join(self.dataset_path, person_name)
        os.makedirs(person_folder, exist_ok=True)
        
        # Count existing images
        existing_count = len([f for f in os.listdir(person_folder) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if existing_count >= self.max_samples_per_person:
            return False
        
        # Generate filename with timestamp and quality score
        timestamp = int(time.time() * 1000)
        filename = f"{person_name}_{timestamp}_q{quality_score:.2f}.jpg"
        filepath = os.path.join(person_folder, filename)
        
        # Save image
        success = cv2.imwrite(filepath, face_image)
        
        if success:
            self.collected_samples[person_name] += 1
            print(f"Saved sample {self.collected_samples[person_name]} for {person_name} (quality: {quality_score:.2f})")
        
        return success
    
    def draw_learning_interface(self, frame: np.ndarray, face_data: List[Dict]) -> np.ndarray:
        """
        Draw the learning mode interface
        """
        # Draw instructions
        instructions = [
            "Face Learning Mode",
            "Keys: 1-9=Select person slot, SPACE=Capture, ESC=Quit",
            f"Current person: {self.current_person or 'None selected'}",
            f"Samples collected: {dict(self.collected_samples)}"
        ]
        
        for i, text in enumerate(instructions):
            y_pos = 30 + i * 25
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
        
        # Draw face detections with quality info
        for i, face_info in enumerate(face_data):
            bbox = face_info['bbox']
            quality_score = face_info['quality_score']
            detection_conf = face_info['detection_confidence']
            
            x1, y1, x2, y2 = bbox
            
            # Color based on quality
            if quality_score > 0.8:
                color = (0, 255, 0)  # Green - excellent
            elif quality_score > 0.6:
                color = (0, 255, 255)  # Yellow - good
            else:
                color = (0, 0, 255)  # Red - poor
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw quality info
            info_text = f"Q:{quality_score:.2f} D:{detection_conf:.2f}"
            cv2.putText(frame, info_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw quality metrics
            metrics = face_info['quality_metrics']
            y_offset = 0
            for key, value in metrics.items():
                y_offset += 15
                metric_text = f"{key[:4]}:{value:.2f}"
                cv2.putText(frame, metric_text, (x2 + 5, y1 + y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw person selection slots
        slot_y = frame.shape[0] - 100
        for i in range(1, 10):
            person_name = f"person_{i}"
            samples_count = self.collected_samples.get(person_name, 0)
            
            slot_text = f"{i}: {samples_count}/{self.max_samples_per_person}"
            color = (0, 255, 0) if person_name == self.current_person else (255, 255, 255)
            
            cv2.putText(frame, slot_text, (10 + (i-1) * 80, slot_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def run_learning_mode(self):
        """
        Run the interactive face learning mode
        """
        print("Starting Face Learning Mode...")
        print("Instructions:")
        print("- Press keys 1-9 to select person slot")
        print("- Press SPACE to capture current best face")
        print("- Press ESC to quit")
        
        # Initialize webcam
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set high resolution for better quality
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        try:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break
                
                # Detect and assess faces
                face_data = self.detect_and_assess_faces(frame)
                
                # Draw interface
                frame = self.draw_learning_interface(frame, face_data)
                
                # Display frame
                cv2.imshow('Face Learning Mode', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                # Quit
                if key == 27:  # ESC
                    break
                
                # Select person slot (1-9)
                elif ord('1') <= key <= ord('9'):
                    person_number = key - ord('0')
                    self.current_person = f"person_{person_number}"
                    print(f"Selected person slot {person_number}")
                
                # Capture face
                elif key == ord(' '):  # SPACE
                    if self.current_person and face_data:
                        current_time = time.time()
                        if current_time - self.last_capture_time > self.capture_interval:
                            # Get best quality face
                            best_face = face_data[0]
                            if best_face['quality_score'] > 0.5:
                                self.save_face_sample(
                                    best_face['face_image'],
                                    self.current_person,
                                    best_face['quality_score']
                                )
                                self.last_capture_time = current_time
                            else:
                                print("Face quality too low, try again")
                    else:
                        print("Select a person slot first (press 1-9)")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
            print("Face learning mode stopped")
            
            # Show summary
            print("\nCollection Summary:")
            for person, count in self.collected_samples.items():
                print(f"  {person}: {count} samples")


def main():
    """
    Main function for face learning mode
    """
    dataset_path = "/home/dkhai/workspace/family_images"
    model_path = "/home/dkhai/workspace/src/yolov11n-face.pt"
    
    try:
        learner = FaceLearningSystem(dataset_path, model_path)
        learner.run_learning_mode()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()