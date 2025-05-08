import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from loguru import logger

class VideoPreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the video preprocessor.
        
        Args:
            target_size (Tuple[int, int]): Target size for resizing frames
        """
        self.target_size = target_size
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            np.ndarray: Preprocessed frame
        """
        # Resize frame
        frame = cv2.resize(frame, self.target_size)
        
        # Convert to RGB (if needed)
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
        # Normalize pixel values
        frame = frame.astype(np.float32) / 255.0
        
        return frame
        
    def preprocess_frames(self, frame_paths: List[str]) -> List[np.ndarray]:
        """
        Preprocess multiple frames.
        
        Args:
            frame_paths (List[str]): List of paths to frames
            
        Returns:
            List[np.ndarray]: List of preprocessed frames
        """
        processed_frames = []
        
        for frame_path in frame_paths:
            try:
                frame = cv2.imread(frame_path)
                if frame is None:
                    logger.warning(f"Could not read frame: {frame_path}")
                    continue
                    
                processed_frame = self.preprocess_frame(frame)
                processed_frames.append(processed_frame)
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_path}: {str(e)}")
                continue
                
        return processed_frames
        
    def apply_face_detection(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Apply face detection to a frame.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            Tuple[np.ndarray, List[Tuple[int, int, int, int]]]: Processed frame and face bounding boxes
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Draw rectangles around faces
        frame_with_faces = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        return frame_with_faces, faces.tolist()
        
    def apply_motion_detection(self, frames: List[np.ndarray]) -> List[float]:
        """
        Calculate motion between consecutive frames.
        
        Args:
            frames (List[np.ndarray]): List of frames
            
        Returns:
            List[float]: List of motion scores
        """
        motion_scores = []
        
        for i in range(1, len(frames)):
            # Convert frames to grayscale
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(prev_gray, curr_gray)
            
            # Calculate motion score
            motion_score = np.mean(diff)
            motion_scores.append(motion_score)
            
        return motion_scores 