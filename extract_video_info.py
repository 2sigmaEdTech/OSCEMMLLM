import cv2
from pathlib import Path
from loguru import logger
from src.utils.config import Config
from src.video_processing.frame_extraction import VideoFrameExtractor
from src.video_processing.preprocessing import VideoPreprocessor

# Configure logger for the script
logger.remove() # Remove default handlers
logger.add(lambda msg: print(msg, end=""), format="{time:YYYY-MM-DD HH:mm:ss} | {level}    | {message}", level="INFO")


def main():
    # Load configuration
    config = Config()
    
    sample_video_path = config.RAW_VIDEOS_DIR / "OSCETrainingVideo-Assessment.mp4"
    
    if not sample_video_path.exists():
        logger.error(f"Sample video not found: {sample_video_path}")
        return

    logger.info(f"Processing sample video: {sample_video_path}")

    # Create a specific output directory for this sample processing
    sample_output_base_dir = config.PROCESSED_FRAMES_DIR / "sample_video_analysis"
    sample_extracted_frames_dir = sample_output_base_dir / "extracted_frames"
    sample_output_visuals_dir = sample_output_base_dir / "visuals"
    
    sample_extracted_frames_dir.mkdir(parents=True, exist_ok=True)
    sample_output_visuals_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    frame_extractor = VideoFrameExtractor(
        output_dir=str(sample_extracted_frames_dir),
        frame_rate=config.FRAME_RATE
    )
    
    preprocessor = VideoPreprocessor(
        target_size=config.TARGET_SIZE
    )

    # 1. Get video metadata
    try:
        logger.info("--- Video Metadata ---")
        metadata = frame_extractor.get_video_metadata(str(sample_video_path))
        for key, value in metadata.items():
            logger.info(f"  {key.replace('_', ' ').title()}: {value}")
        

        # 2. Extract frames
        logger.info("--- Frame Extraction ---")
        # This will save frames to sample_extracted_frames_dir
        frame_paths = frame_extractor.extract_frames(str(sample_video_path))
        
        if not frame_paths:
            logger.warning("No frames were extracted based on the current configuration.")
            return

        logger.info(f"Extracted {len(frame_paths)} frames to: {sample_extracted_frames_dir}")
        logger.info(f"Configured FRAME_RATE: {config.FRAME_RATE} fps.")
        if frame_paths:
            logger.info(f"Path to first extracted frame: {frame_paths[0]}")


        # 3. Preprocess the first extracted frame (as a demo)
        logger.info("--- Frame Preprocessing (First Frame Demo) ---")
        # We'll use the path of the first extracted frame
        first_frame_path_list = [frame_paths[0]] 
        
        # preprocess_frames returns a list of NumPy arrays (the preprocessed frames)
        processed_frames_np_list = preprocessor.preprocess_frames(first_frame_path_list) 

        if processed_frames_np_list:
            first_processed_frame_np = processed_frames_np_list[0]
            logger.info(f"Successfully preprocessed the first frame.")
            logger.info(f"  Shape after preprocessing (Height, Width, Channels): {first_processed_frame_np.shape}")
            logger.info(f"  Data type: {first_processed_frame_np.dtype}")
            logger.info(f"  Min/Max pixel values: {first_processed_frame_np.min()}/{first_processed_frame_np.max()}")
        else:
            logger.warning("Could not preprocess the first frame.")

        # 4. Apply face detection to the first extracted frame (using its raw, resized version)
        logger.info("--- Face Detection (First Frame Demo) ---")
        # Read the raw first frame (it's BGR uint8)
        raw_first_frame_bgr = cv2.imread(frame_paths[0])
        if raw_first_frame_bgr is not None:
            # Convert BGR to RGB for consistency with VideoPreprocessor.apply_face_detection expectations
            raw_first_frame_rgb = cv2.cvtColor(raw_first_frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Resize the RGB frame to the target size (as preprocessor would do before normalization)
            raw_first_frame_rgb_resized = cv2.resize(raw_first_frame_rgb, config.TARGET_SIZE)

            # apply_face_detection expects an RGB frame and returns an RGB frame with detections
            frame_with_faces_rgb, faces = preprocessor.apply_face_detection(raw_first_frame_rgb_resized)
            
            logger.info(f"Detected {len(faces)} faces in the first frame (after resizing).")
            if faces:
                logger.info(f"  Face bounding boxes (x, y, w, h): {faces}")
            
            # Convert the resulting RGB frame (with drawings) back to BGR for saving with cv2.imwrite
            frame_with_faces_bgr = cv2.cvtColor(frame_with_faces_rgb, cv2.COLOR_RGB2BGR)
            
            output_face_detection_path = sample_output_visuals_dir / f"{Path(frame_paths[0]).stem}_with_faces.jpg"
            cv2.imwrite(str(output_face_detection_path), frame_with_faces_bgr)
            logger.info(f"Saved frame with face detection visualization to: {output_face_detection_path}")
        else:
            logger.warning(f"Could not read raw frame {frame_paths[0]} for face detection visualization.")

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
    except ValueError as e:
        logger.error(f"Value error during processing: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main() 