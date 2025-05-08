import argparse
from pathlib import Path
from loguru import logger
from utils.config import Config
from video_processing.frame_extraction import VideoFrameExtractor
from video_processing.preprocessing import VideoPreprocessor

def setup_logging():
    """Configure logging settings."""
    logger.add(
        "logs/app.log",
        rotation="500 MB",
        retention="10 days",
        level="INFO"
    )

def process_video(video_path: str, config: Config):
    """
    Process a single video file.
    
    Args:
        video_path (str): Path to the video file
        config (Config): Configuration object
    """
    try:
        # Initialize components
        frame_extractor = VideoFrameExtractor(
            output_dir=config.PROCESSED_FRAMES_DIR,
            frame_rate=config.FRAME_RATE
        )
        
        preprocessor = VideoPreprocessor(
            target_size=config.TARGET_SIZE
        )
        
        # Extract frames
        logger.info(f"Processing video: {video_path}")
        frame_paths = frame_extractor.extract_frames(video_path)
        
        # Get video metadata
        metadata = frame_extractor.get_video_metadata(video_path)
        logger.info(f"Video metadata: {metadata}")
        
        # Preprocess frames
        processed_frames = preprocessor.preprocess_frames(frame_paths)
        logger.info(f"Preprocessed {len(processed_frames)} frames")
        
        # Apply face detection to first frame as example
        if processed_frames:
            frame_with_faces, faces = preprocessor.apply_face_detection(processed_frames[0])
            logger.info(f"Detected {len(faces)} faces in first frame")
            
        # Calculate motion scores
        motion_scores = preprocessor.apply_motion_detection(processed_frames)
        logger.info(f"Calculated motion scores for {len(motion_scores)} frame pairs")
        
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")
        raise

def main():
    """Main entry point for the application."""
    # Setup argument parser
    parser = argparse.ArgumentParser(description="OSCE Video Analysis System")
    parser.add_argument(
        "--video",
        type=str,
        help="Path to the video file to process"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load configuration
    config = Config()
    
    if args.video:
        # Process single video
        process_video(args.video, config)
    else:
        # Process all videos in the raw videos directory
        video_files = list(config.RAW_VIDEOS_DIR.glob("*.mp4"))
        for video_path in video_files:
            process_video(str(video_path), config)

if __name__ == "__main__":
    main() 