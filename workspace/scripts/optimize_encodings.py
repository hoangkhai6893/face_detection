#!/usr/bin/env python3
"""
Face Recognition Optimizer
Optimizes existing face encodings for better performance and accuracy
"""

import logging
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import pickle
import numpy as np
import face_recognition
import cv2
from ultralytics import YOLO
from collections import defaultdict
from typing import List, Dict, Tuple
import time

import config
from core.face_utils import extract_face_region

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
        self.logger = logging.getLogger(__name__)
        self.yolo_model = YOLO(model_path)

        # Optimization parameters
        self.quality_threshold = config.OPTIMIZE_QUALITY_THRESHOLD
        self.max_encodings_per_person = config.MAX_ENCODINGS_PER_PERSON
        self.clustering_threshold = config.CLUSTERING_THRESHOLD
    
    def load_existing_encodings(self) -> Tuple[List[np.ndarray], List[str]]:
        """Load existing face encodings"""
        encodings_file = os.path.join(config.ENCODINGS_DIR, "face_encodings_hybrid.pkl")

        if os.path.exists(encodings_file):
            self.logger.info("Loading existing encodings...")
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
            self.logger.warning("Could not read image: %s", image_path)
            return []

        # Use YOLO to detect faces
        results = self.yolo_model(image, verbose=False)
        face_encodings_with_quality = []
        yolo_found = False

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                confidence = float(box.conf[0])

                if confidence > config.HIGH_CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face_image = extract_face_region(
                        image, x1, y1, x2, y2, config.OPTIMIZE_PADDING
                    )
                    if face_image is None:
                        continue

                    rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_face)

                    if len(encodings) > 0:
                        face_area = (x2 - x1) * (y2 - y1)
                        quality_score = confidence * min(1.0, face_area / config.FACE_AREA_NORMALIZATION)
                        face_encodings_with_quality.append((encodings[0], quality_score))
                        yolo_found = True

        # --- Fallback: YOLO missed the face (dark/blurry/angled image) ---
        # Run face_recognition directly on the full image.
        # These encodings are assigned a lower quality score so they don't outcompete
        # clean images in clustering, but they ARE preserved for adverse-condition coverage.
        if not yolo_found:
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                fb_encodings = face_recognition.face_encodings(rgb_image)
                if fb_encodings:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    brightness = np.mean(gray) / 255.0
                    sharpness = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0)
                    # Cap at 0.3 so fallback images are deprioritised in clustering
                    # but still pass the OPTIMIZE_QUALITY_THRESHOLD (now 0.15)
                    fallback_quality = min(0.3, (brightness + sharpness) / 2)
                    face_encodings_with_quality.append((fb_encodings[0], fallback_quality))
                    self.logger.debug(
                        "Fallback encoding used for: %s (quality=%.2f)",
                        os.path.basename(image_path), fallback_quality
                    )
            except Exception as e:
                self.logger.debug("Fallback encoding failed for %s: %s", image_path, e)

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
        
        # Select the most representative encoding from each cluster.
        # "Most representative" = closest to the cluster centroid (the average face).
        # This avoids always picking the clearest/best-lit image and discarding
        # adverse-condition examples that happen to fall in the same cluster.
        selected_indices = []
        for cluster in clusters:
            if len(cluster) == 1:
                selected_indices.append(cluster[0])
                continue
            cluster_encs = np.array([encodings[i] for i in cluster])
            centroid = cluster_encs.mean(axis=0)
            dist_to_centroid = [np.linalg.norm(encodings[i] - centroid) for i in cluster]
            representative = cluster[int(np.argmin(dist_to_centroid))]
            selected_indices.append(representative)

        # If we still have too many, keep a quality-diverse subset:
        # sort by quality and pick evenly spaced indices so we retain
        # both high-quality and low-quality (adverse-condition) representatives.
        if len(selected_indices) > self.max_encodings_per_person:
            selected_indices.sort(key=lambda x: qualities[x])
            step = len(selected_indices) / self.max_encodings_per_person
            selected_indices = [
                selected_indices[int(i * step)]
                for i in range(self.max_encodings_per_person)
            ]
        
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
        
        self.logger.info("Optimizing encodings for %s...", person_name)
        
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
            self.logger.warning("No quality faces found for %s", person_name)
            return [], original_count, 0
        
        # Cluster and select best encodings
        selected_indices = self.cluster_similar_encodings(all_encodings, all_qualities)
        optimized_encodings = [all_encodings[i] for i in selected_indices]
        
        final_count = len(optimized_encodings)
        
        self.logger.info(
            "%s: %d -> %d encodings | quality range %.2f-%.2f | selected avg %.2f",
            person_name, original_count, final_count,
            min(all_qualities), max(all_qualities),
            np.mean([all_qualities[i] for i in selected_indices])
        )
        
        return optimized_encodings, original_count, final_count
    
    def optimize_all_encodings(self) -> bool:
        """
        Optimize all face encodings in the dataset
        
        Returns:
            True if successful
        """
        self.logger.info("Starting face encoding optimization...")

        if not os.path.exists(self.dataset_path):
            self.logger.error("Dataset path not found: %s", self.dataset_path)
            return False
        
        # Get all person folders
        person_folders = [d for d in os.listdir(self.dataset_path) 
                         if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        if not person_folders:
            self.logger.error("No person folders found in dataset")
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
            encodings_file = os.path.join(config.ENCODINGS_DIR, "face_encodings_hybrid.pkl")
            backup_file = os.path.join(config.ENCODINGS_DIR, "face_encodings_hybrid_backup.pkl")
            
            # Create backup of original encodings
            if os.path.exists(encodings_file):
                os.rename(encodings_file, backup_file)
                self.logger.info("Original encodings backed up to: %s", backup_file)

            # Save new optimized encodings
            with open(encodings_file, 'wb') as f:
                pickle.dump({
                    'encodings': all_optimized_encodings,
                    'names': all_optimized_names
                }, f)

            reduction_pct = (total_original - total_final) / total_original * 100
            self.logger.info(
                "Optimization completed — people: %d | encodings: %d -> %d (%.1f%% reduction) | saved to: %s",
                len(person_folders), total_original, total_final, reduction_pct, encodings_file
            )
            return True
        else:
            self.logger.error("No valid encodings generated")
            return False
    
    def benchmark_performance(self):
        """
        Benchmark the performance improvement
        """
        self.logger.info("Running performance benchmark...")

        # Load optimized encodings
        encodings_file = os.path.join(config.ENCODINGS_DIR, "face_encodings_hybrid.pkl")
        backup_file = os.path.join(config.ENCODINGS_DIR, "face_encodings_hybrid_backup.pkl")

        if not os.path.exists(encodings_file):
            self.logger.error("No optimized encodings found")
            return

        with open(encodings_file, 'rb') as f:
            optimized_data = pickle.load(f)

        self.logger.info("Current encodings: %d", len(optimized_data['encodings']))

        if os.path.exists(backup_file):
            with open(backup_file, 'rb') as f:
                backup_data = pickle.load(f)
            self.logger.info("Original encodings: %d", len(backup_data['encodings']))
            
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

                self.logger.info(
                    "Performance: original %.3fs vs optimized %.3fs (100 comparisons) — %.1fx faster",
                    original_time, optimized_time, speedup
                )


def main():
    """
    Main function for face encoding optimization
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.info("FACE ENCODING OPTIMIZER")

    try:
        optimizer = FaceEncodingOptimizer(config.DATASET_PATH, config.MODEL_PATH)

        # Run optimization
        success = optimizer.optimize_all_encodings()

        if success:
            # Run benchmark
            optimizer.benchmark_performance()

        logger.info("Optimization process completed.")

    except Exception as e:
        logger.exception("Error during optimization: %s", e)


if __name__ == "__main__":
    main()