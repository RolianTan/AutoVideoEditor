import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import CLIPTokenizer
import cv2
from torch.utils.data import Dataset

# Function to extract all frames from a video using OpenCV
def extract_video_frames(video_path, frame_size, target_fps):
    video = cv2.VideoCapture(video_path)
    frames = []
    original_fps = video.get(cv2.CAP_PROP_FPS)
    extract_gap = int(original_fps / target_fps)  # calculate the fps gap to match target FPS
    step = 0

    while video.isOpened():
        stat, frame = video.read()
        if not stat:
            break
        # Only keep frames at the specified interval
        if step % extract_gap == 0:
            frame = cv2.resize(frame, (frame_size, frame_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        step += 1

    video.release()
    return frames

# Function to sample N frames from a list of frames
def sample_frames(frames, n_frames):
    if len(frames) >= n_frames:
        sampled_indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
        sampled_frames = [frames[i] for i in sampled_indices]
    else:
        # Pad with the last frame if there are not enough frames
        sampled_frames = frames + [frames[-1]] * (n_frames - len(frames))
    return sampled_frames


# Function to load and preprocess the data from the JSON file
def preprocess(json_file, n_frames, frame_size, target_fps):
    # Load the JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)

    sampled_frames_list = []
    captions = []
    current_video_path = None
    all_frames = []

    # Iterate over each entry in the JSON
    for entry in data:
        video_path = entry['video_path']
        start_time = entry['start_time']
        end_time = entry['end_time']
        caption = entry['caption']

        # If we encounter a new video, load and extract all frames
        if video_path != current_video_path:
            current_video_path = video_path
            all_frames = extract_video_frames(video_path, frame_size, target_fps)  # Extract frames

        # Convert interval times to frame indices
        start_idx = int(start_time * target_fps)
        end_idx = int(end_time * target_fps)
        interval_frames = all_frames[start_idx:end_idx]

        # Sample N frames from the interval
        sampled_frames = sample_frames(interval_frames, n_frames)

        # Store the sampled frames and caption
        sampled_frames_list.append(sampled_frames)
        captions.append(caption)

    return sampled_frames_list, captions

# VideoTextDataset class
class VideoTextDataset(Dataset):
    def __init__(self, sampled_frames_list, captions):
        self.sampled_frames_list = sampled_frames_list
        self.captions = captions
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.sampled_frames_list)

    def __getitem__(self, idx):
        sampled_frames = self.sampled_frames_list[idx]
        caption = self.captions[idx]
        video_frames = torch.Tensor(np.array(sampled_frames)).permute(0, 3, 1, 2)

        # Tokenizer
        caption_tokens = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=77,
            return_tensors='pt'
        )

        return {
            'video': video_frames,  # (N, c, h, w)
            'caption_input_ids': caption_tokens['input_ids'].squeeze(),  # (max_length,)
            'caption_attention_mask': caption_tokens['attention_mask'].squeeze()  # (max_length,)
        }

# Function to create the dataset
def create_dataset(json_file, n_frames, frame_size, target_fps):
    sampled_frames_list, captions = preprocess(json_file, n_frames, frame_size, target_fps)
    dataset = VideoTextDataset(sampled_frames_list, captions)
    return dataset