import cv2
import base64
import json
from scenedetect import VideoManager, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector
from tqdm import tqdm
from openai import OpenAI
import os

client = OpenAI(api_key="sk-CxLi-7SJvgBVHbOfefj7-w", base_url="https://nova-litellm-proxy.onrender.com")
#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "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"))

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
    general_prompt = (
    "These frames are sampled from a video. Generate a professional and concise description that captures the video's visual characteristics. \
    Focus on *high-level photography keywords*, using commas to separate each term. \
    Do not include specific objects, summarized keywords should include scene, color, style, view, lighting, motion, texture, composition, mood/tone, perspective/angle, and depth of field. \
    Format the response as follows: [scene:..., color:..., style:..., view:..., lighting:..., motion:..., texture:..., composition:..., mood/tone:..., perspective/angle:..., depth of field:...]. \
    Fill each category with professional photography descriptive keywords. \
    Limit the description to 77 tokens and focus only on visual and graphical elements, ignoring any text or dialogue. \
    Prioritize keywords that best encapsulate the primary visual style and mood, with the most representative terms listed first."
    )

    for i, base64_frames in tqdm(enumerate(interval_frames)):
        start_time, end_time = scene_intervals[i]

        sampled_frames = base64_frames[::10]

        # Build messages with images
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": general_prompt,
            }
        ]

        # Add each sampled frame as a separate message with the image
        for frame in sampled_frames:
            PROMPT_MESSAGES.append({
                "role": "user",
                "content": "",  # Empty content for image-only messages
                "image": {"base64": frame, "resize": 512}
            })

        params = {
            "model": "openai/gpt-4o",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 77,
        }

        # Call the ChatGPT Vision API
        result = client.chat.completions.create(**params)
        caption = result.choices[0].message.content

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
