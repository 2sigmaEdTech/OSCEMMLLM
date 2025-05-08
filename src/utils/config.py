import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

class Config:
    def __init__(self):
        """Initialize configuration with default values."""
        # Load environment variables
        load_dotenv()
        
        # Base directories
        self.BASE_DIR = Path(__file__).parent.parent.parent
        self.DATA_DIR = self.BASE_DIR / "data"
        self.RAW_VIDEOS_DIR = self.DATA_DIR / "raw_videos"
        self.PROCESSED_FRAMES_DIR = self.DATA_DIR / "processed_frames"
        
        # Create directories if they don't exist
        self.DATA_DIR.mkdir(exist_ok=True)
        self.RAW_VIDEOS_DIR.mkdir(exist_ok=True)
        self.PROCESSED_FRAMES_DIR.mkdir(exist_ok=True)
        
        # Video processing settings
        self.FRAME_RATE = int(os.getenv("FRAME_RATE", "1"))  # frames per second
        self.TARGET_SIZE = (
            int(os.getenv("TARGET_WIDTH", "224")),
            int(os.getenv("TARGET_HEIGHT", "224"))
        )
        
        # Model settings
        self.MODEL_NAME = os.getenv("MODEL_NAME", "gpt4-v")
        self.API_KEY = os.getenv("API_KEY", "")
        
        # Assessment settings
        self.CHECKLIST_THRESHOLD = float(os.getenv("CHECKLIST_THRESHOLD", "0.5"))
        self.ENTRUSTMENT_LEVELS = int(os.getenv("ENTRUSTMENT_LEVELS", "5"))
        
    def get_video_processing_config(self) -> Dict[str, Any]:
        """Get video processing configuration."""
        return {
            "frame_rate": self.FRAME_RATE,
            "target_size": self.TARGET_SIZE,
            "raw_videos_dir": str(self.RAW_VIDEOS_DIR),
            "processed_frames_dir": str(self.PROCESSED_FRAMES_DIR)
        }
        
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "model_name": self.MODEL_NAME,
            "api_key": self.API_KEY
        }
        
    def get_assessment_config(self) -> Dict[str, Any]:
        """Get assessment configuration."""
        return {
            "checklist_threshold": self.CHECKLIST_THRESHOLD,
            "entrustment_levels": self.ENTRUSTMENT_LEVELS
        } 