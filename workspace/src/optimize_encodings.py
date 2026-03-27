#!/usr/bin/env python3
"""
Face Recognition Optimizer
Optimizes existing face encodings for better performance and accuracy
"""

import os
import pickle
import numpy as np
import face_recognition
import cv2
from ultralytics import YOLO
from collections import defaultdict
from typing import List, Dict, Tuple
import time

class FaceEncodingOptimizer:
    def __init__(self, dataset_path: str, model_path: str):
        """
        Initialize the face encoding optimizer
        
        Args:
            dataset_path: Path to the face images dataset
            model_path: Path to YOLO face detection model
        """
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.yolo_model = YOLO(model_path)
        
        # Optimization parameters
        self.quality_threshold = 0.6
        self.max_encodings_per_person = 20  # Optimal number of encodings per person
        self.clustering_threshold = 0.4  # Distance threshold for clustering similar faces
    
    def load_existing_encodings(self) -> Tuple[List[np.ndarray], List[str]]:
        """Load existing face encodings"""
        encodings_file = os.path.join(os.path.dirname(self.dataset_path), "face_encodings.pkl")
        
        if os.path.exists(encodings_file):
            print("Loading existing encodings...")
            with open(encodings_file, 'rb') as f:
                data = pickle.load(f)
                return data['encodings'], data['names']
        else:
            return [], []
    
    def extract_high_quality_faces(self, image_path: str) -> List[Tuple[np.ndarray, float]]:
        """
        Extract high-quality face encodings from an image using YOLO detection
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of (face_encoding, quality_score) tuples
        """
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        # Use YOLO to detect faces
        results = self.yolo_model(image, verbose=False)
        face_encodings_with_quality = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    
                    if confidence > 0.7:  # High confidence threshold
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Extract face with padding
                        padding = 20
                        x1_pad = max(0, x1 - padding)
                        y1_pad = max(0, y1 - padding)
                        x2_pad = min(image.shape[1], x2 + padding)
                        y2_pad = min(image.shape[0], y2 + padding)
                        
                        face_image = image[y1_pad:y2_pad, x1_pad:x2_pad]
                        
                        # Convert to RGB and get encoding
                        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                        encodings = face_recognition.face_encodings(rgb_face)
                        
                        if len(encodings) > 0:
                            # Calculate quality score based on face size and detection confidence
                            face_area = (x2 - x1) * (y2 - y1)
                            quality_score = confidence * min(1.0, face_area / 10000)  # Normalize by area
                            
                            face_encodings_with_quality.append((encodings[0], quality_score))
        
        return face_encodings_with_quality
    
    def cluster_similar_encodings(self, encodings: List[np.ndarray], qualities: List[float]) -> List[int]:
        """
        Cluster similar face encodings and return indices of best representatives
        
        Args:
            encodings: List of face encodings
            qualities: List of corresponding quality scores
            
        Returns:
            Indices of the best encoding from each cluster
        """
        if len(encodings) <= self.max_encodings_per_person:
            return list(range(len(encodings)))
        
        # Calculate pairwise distances
        distances = np.zeros((len(encodings), len(encodings)))
        for i in range(len(encodings)):
            for j in range(i + 1, len(encodings)):
                dist = face_recognition.face_distance([encodings[i]], encodings[j])[0]
                distances[i][j] = distances[j][i] = dist
        
        # Simple clustering: group encodings that are within threshold distance
        clusters = []
        used = set()
        
        for i in range(len(encodings)):
            if i in used:
                continue
            
            cluster = [i]
            used.add(i)
            
            for j in range(i + 1, len(encodings)):
                if j not in used and distances[i][j] < self.clustering_threshold:
                    cluster.append(j)
                    used.add(j)
            
            clusters.append(cluster)
        
        # Select best encoding from each cluster (highest quality)
        selected_indices = []
        for cluster in clusters:
            best_idx = max(cluster, key=lambda x: qualities[x])
            selected_indices.append(best_idx)
        
        # If we still have too many, select top quality ones
        if len(selected_indices) > self.max_encodings_per_person:
            selected_indices.sort(key=lambda x: qualities[x], reverse=True)
            selected_indices = selected_indices[:self.max_encodings_per_person]
        
        return selected_indices
    
    def optimize_person_encodings(self, person_name: str) -> Tuple[List[np.ndarray], int, int]:
        """
        Optimize encodings for a specific person
        
        Args:
            person_name: Name of the person
            
        Returns:
            Tuple of (optimized_encodings, original_count, final_count)
        """
        person_folder = os.path.join(self.dataset_path, person_name)
        
        if not os.path.exists(person_folder):
            return [], 0, 0
        
        print(f"\n🔄 Optimizing encodings for {person_name}...")
        
        # Extract all face encodings with quality scores
        all_encodings = []
        all_qualities = []
        
        image_files = [f for f in os.listdir(person_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for image_file in image_files:
            image_path = os.path.join(person_folder, image_file)
            face_data = self.extract_high_quality_faces(image_path)
            
            for encoding, quality in face_data:
                if quality > self.quality_threshold:
                    all_encodings.append(encoding)
                    all_qualities.append(quality)
        
        original_count = len(all_encodings)
        
        if len(all_encodings) == 0:
            print(f"  ❌ No quality faces found for {person_name}")
            return [], original_count, 0
        
        # Cluster and select best encodings
        selected_indices = self.cluster_similar_encodings(all_encodings, all_qualities)
        optimized_encodings = [all_encodings[i] for i in selected_indices]
        
        final_count = len(optimized_encodings)
        
        print(f"  ✅ {person_name}: {original_count} → {final_count} encodings")
        print(f"     📊 Quality range: {min(all_qualities):.2f} - {max(all_qualities):.2f}")
        print(f"     🎯 Selected avg quality: {np.mean([all_qualities[i] for i in selected_indices]):.2f}")
        
        return optimized_encodings, original_count, final_count
    
    def optimize_all_encodings(self) -> bool:
        """
        Optimize all face encodings in the dataset
        
        Returns:
            True if successful
        """
        print("🚀 Starting face encoding optimization...")
        
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset path not found: {self.dataset_path}")
            return False
        
        # Get all person folders
        person_folders = [d for d in os.listdir(self.dataset_path) 
                         if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        if not person_folders:
            print("❌ No person folders found in dataset")
            return False
        
        # Optimize encodings for each person
        all_optimized_encodings = []
        all_optimized_names = []
        total_original = 0
        total_final = 0
        
        for person_name in person_folders:
            encodings, original_count, final_count = self.optimize_person_encodings(person_name)
            
            all_optimized_encodings.extend(encodings)
            all_optimized_names.extend([person_name] * len(encodings))
            total_original += original_count
            total_final += final_count
        
        # Save optimized encodings
        if all_optimized_encodings:
            encodings_file = os.path.join(os.path.dirname(self.dataset_path), "face_encodings.pkl")
            backup_file = os.path.join(os.path.dirname(self.dataset_path), "face_encodings_backup.pkl")
            
            # Create backup of original encodings
            if os.path.exists(encodings_file):
                os.rename(encodings_file, backup_file)
                print(f"💾 Original encodings backed up to: {backup_file}")
            
            # Save new optimized encodings
            with open(encodings_file, 'wb') as f:
                pickle.dump({
                    'encodings': all_optimized_encodings,
                    'names': all_optimized_names
                }, f)
            
            print(f"\n✅ Optimization completed!")
            print(f"📊 Summary:")
            print(f"   👥 People optimized: {len(person_folders)}")
            print(f"   🔢 Total encodings: {total_original} → {total_final}")
            print(f"   📈 Reduction: {(total_original - total_final) / total_original * 100:.1f}%")
            print(f"   💾 Saved to: {encodings_file}")
            
            return True
        else:
            print("❌ No valid encodings generated")
            return False
    
    def benchmark_performance(self):
        """
        Benchmark the performance improvement
        """
        print("\n🏃 Running performance benchmark...")
        
        # Load optimized encodings
        encodings_file = os.path.join(os.path.dirname(self.dataset_path), "face_encodings.pkl")
        backup_file = os.path.join(os.path.dirname(self.dataset_path), "face_encodings_backup.pkl")
        
        if not os.path.exists(encodings_file):
            print("❌ No optimized encodings found")
            return
        
        with open(encodings_file, 'rb') as f:
            optimized_data = pickle.load(f)
        
        print(f"📊 Current encodings: {len(optimized_data['encodings'])}")
        
        if os.path.exists(backup_file):
            with open(backup_file, 'rb') as f:
                backup_data = pickle.load(f)
            print(f"📊 Original encodings: {len(backup_data['encodings'])}")
            
            # Simple speed test
            test_encoding = optimized_data['encodings'][0] if optimized_data['encodings'] else None
            
            if test_encoding is not None:
                # Test original
                start_time = time.time()
                for _ in range(100):
                    face_recognition.compare_faces(backup_data['encodings'], test_encoding)
                original_time = time.time() - start_time
                
                # Test optimized
                start_time = time.time()
                for _ in range(100):
                    face_recognition.compare_faces(optimized_data['encodings'], test_encoding)
                optimized_time = time.time() - start_time
                
                speedup = original_time / optimized_time if optimized_time > 0 else 1
                
                print(f"⚡ Performance improvement:")
                print(f"   Original: {original_time:.3f}s (100 comparisons)")
                print(f"   Optimized: {optimized_time:.3f}s (100 comparisons)")
                print(f"   Speedup: {speedup:.1f}x faster")


def main():
    """
    Main function for face encoding optimization
    """
    dataset_path = "/home/dkhai/workspace/family_images"
    model_path = "/home/dkhai/workspace/src/yolov11n-face.pt"
    
    print("🎯 FACE ENCODING OPTIMIZER")
    print("=" * 50)
    
    try:
        optimizer = FaceEncodingOptimizer(dataset_path, model_path)
        
        # Run optimization
        success = optimizer.optimize_all_encodings()
        
        if success:
            # Run benchmark
            optimizer.benchmark_performance()
        
        print("\n✨ Optimization process completed!")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()