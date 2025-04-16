import os
import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

def extract_scene_keyframes(video_path, output_folder, threshold=30.0):
    """
    Uses PySceneDetect to detect scenes and extract keyframes from each (start, middle, end),
    ignoring scenes shorter than 1 second.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Set up PySceneDetect manager
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.set_downscale_factor()  # optional, speed up detection
    video_manager.start()

    # Detect scenes
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    print(f"Detected {len(scene_list)} scenes (before filtering).")

    # OpenCV video access
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    scene_save_count = 0
    for i, (start, end) in enumerate(scene_list):
        start_frame = start.get_frames() + 10
        end_frame = end.get_frames() - 10  # first and last frames tends to be transitions
        duration_frames = end_frame - start_frame + 20

        # Skip if the scene duration is less than 1.5 second
        if duration_frames < fps * 1.5:
            print(f"Skipping scene {i} (duration {duration_frames/fps:.2f}s)")
            continue

        mid_frame = (start_frame + end_frame) // 2
        
        for j, frame_num in enumerate([start_frame, mid_frame, end_frame]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_num} for scene {i}")
                continue
            frame_path = os.path.join(output_folder, f"scene_{scene_save_count}_frame{j}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Saved: {frame_path}")

        scene_save_count += 1

    cap.release()
    print("Scene keyframe extraction complete.")

# Example usage
extract_scene_keyframes("AutoLabel/mad_template.mp4", "scene_frames")
