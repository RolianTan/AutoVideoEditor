import cv2
import base64
import json
from scenedetect import VideoManager, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector
from tqdm import tqdm
from openai import OpenAI
import os
import openai
import random
from AutoLabel_Eval import label_frames
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Extract all frames & convert to base64-encoded JPG
def extract_all_frames(video_path, frame_size=512):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    all_frames = []
    while video.isOpened():
        stat, frame = video.read()
        if not stat:
            break
        frame = cv2.resize(frame, (frame_size, frame_size))
        _, buffer = cv2.imencode(".jpg", frame)
        base64_frame = base64.b64encode(buffer).decode("utf-8")
        all_frames.append(base64_frame)
    video.release()
    return all_frames, fps

# seperate the scene and get the scene list
def scene_segmentation(video_path, seg_threshold=27.0):
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector(threshold=seg_threshold))

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager, show_progress=True)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    # round the time to 2 decimal numbers
    scene_intervals = [
        (round(scene_time[0].get_seconds(), 2), round(scene_time[1].get_seconds(), 2))
        for scene_time in scene_list
    ]

    return scene_intervals

def get_frames_for_intervals(all_frames, scene_intervals, fps):
    interval_frames = []
    for start_time, end_time in scene_intervals:
        start_idx = int(start_time * fps)
        end_idx = int(end_time * fps)
        interval_frames.append(all_frames[start_idx:end_idx])
    return interval_frames

def generate_captions_for_intervals(video_path, scene_intervals, interval_frames):
    video_caption = []

    for i, base64_frames in tqdm(enumerate(interval_frames)):
        start_time, end_time = scene_intervals[i]
        # derive start, mid and end frames
        total_frames = len(base64_frames)
        target_video = base64_frames[(total_frames // 2)]
        # if total_frames >= 3:
        #     start_frame = base64_frames[0]
        #     middle_frame = base64_frames[total_frames // 2]
        #     end_frame = base64_frames[-1]
        #     target_video = [start_frame, middle_frame, end_frame]
        # else:
        #     # Handle cases where there are fewer than 3 frames
        #     target_video = base64_frames
        # sampled_frames = base64_frames[::10]

        # choose random other video scenes, and derive 3 frames, create frame list for comparison evaluation
        current_idx = i
        other_indices = [n for n in range(len(interval_frames)) if n != current_idx]
        picked_videos = random.sample(other_indices, k=min(3, len(other_indices)))

        distractor_videos = []
        for j in picked_videos:
            # other_scene = interval_frames[j]
            # if len(other_scene) >= 3:
            #     distractor_videos.extend([
            #         other_scene[0],
            #         other_scene[len(other_scene) // 2],
            #         other_scene[-1]
            #     ])
            # else:
            #     distractor_videos.extend(other_scene)
            mid_point = len(interval_frames[j]) // 2
            distractor_videos.append(interval_frames[j][mid_point])

        # caption generation
        caption = label_frames.get_frame_description(target_video, distractor_videos, idx=i)

        # Store the information
        video_caption.append({
            "video_path": video_path,
            "start_time": start_time,
            "end_time": end_time,
            "caption": caption
        })

    return video_caption


if __name__ == '__main__':
    # configs
    video_folder = 'training_videos/dataset01/'
    frame_size = 512
    save_path = 'training_videos/dataset01/dataset_01.json'
    final_list = []
    # os.makedirs(video_folder, exist_ok=True)
    # videos path
    video_list = [os.path.join(video_folder, v) for v in os.listdir(video_folder)]
    # process the videos one by one
    for path in video_list:
        all_frames, fps = extract_all_frames(path, frame_size)
        # get scene intervals and fps information
        scene_intervals = scene_segmentation(path)
        # extract frames for each interval
        frames_list = get_frames_for_intervals(all_frames, scene_intervals, fps)
        # generate captions for each video clip
        video_caption = generate_captions_for_intervals(path, scene_intervals, frames_list)
        final_list.extend(video_caption)
        # write to json
        with open(save_path, 'w') as f:
            json.dump(final_list, f, indent=4)
        print(f"Finish and write JSON file to {save_path}.")
