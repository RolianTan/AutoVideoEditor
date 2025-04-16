import cv2
import os

def extract_frames(video_file, output_folder, frame_interval=3):
    """
    Extracts frames from a video file at a specified interval and saves them as images.
    
    :param video_file: Path to the input video file
    :param output_folder: Directory where extracted frames will be saved
    :param frame_interval: Time interval (in seconds) between frames to be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    frame_count = 0
    frame_save_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        if frame_count % (fps * frame_interval) == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_save_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_save_count += 1
            print(f"Saved: {frame_filename}")
        
        frame_count += 1
    
    cap.release()
    print("Frame extraction complete.")

# Example usage
extract_frames("input.mp4", "vlog_frames_1")
