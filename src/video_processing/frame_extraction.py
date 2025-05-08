import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from loguru import logger

class VideoFrameExtractor:
    def __init__(self, output_dir: str, frame_rate: int = 1):
        """
        Initialize the video frame extractor.
        
        Args:
            output_dir (str): Directory to save extracted frames
            frame_rate (int): Number of frames to extract per second
        """
        self.output_dir = Path(output_dir)
        self.frame_rate = frame_rate
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_frames(self, video_path: str) -> List[str]:
        """
        Extract frames from a video file.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            List[str]: List of paths to extracted frames
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval
        frame_interval = int(fps / self.frame_rate)
        
        frame_paths = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    # Save frame
                    frame_path = self.output_dir / f"{video_path.stem}_frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(str(frame_path))
                    
                frame_count += 1
                
        finally:
            cap.release()
            
        logger.info(f"Extracted {len(frame_paths)} frames from {video_path}")
        return frame_paths
        
    def get_video_metadata(self, video_path: str) -> dict:
        """
        Get metadata from a video file.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            dict: Video metadata including duration, resolution, etc.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        metadata = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        }
        
        cap.release()
        return metadata 