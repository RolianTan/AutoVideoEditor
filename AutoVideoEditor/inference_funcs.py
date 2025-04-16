import argparse
import torch
import os
from tqdm import tqdm
import numpy as np
from model import AutoClipModel
from transformers import CLIPTokenizer, CLIPModel, CLIPVisionModel
from moviepy import VideoFileClip, AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
from concurrent.futures import ThreadPoolExecutor
import json
import cv2
import random

# PART: Utility functions
# check the overlap video clips
def intervals_overlap(start1, end1, start2, end2):
    return max(start1, start2) < min(end1, end2)

# extract video all the frames for the footage video
def extract_video_frames(video_path, frame_size, target_fps):
    video = cv2.VideoCapture(video_path)
    frames = []
    original_fps = video.get(cv2.CAP_PROP_FPS)
    extract_gap = int(original_fps / target_fps)
    step = 0

    while video.isOpened():
        stat, frame = video.read()
        if not stat:
            break
        if step % extract_gap == 0:
            frame = cv2.resize(frame, (frame_size, frame_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        step += 1

    video.release()
    return np.array(frames)

# for a period of video clip, sample n frames.
# TODO: more efficient way to sample frames
def sample_frames(frames, n_frames):
    if len(frames) >= n_frames:
        sampled_indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
        sampled_frames = [frames[i] for i in sampled_indices]
    else:
        sampled_frames = frames + [frames[-1]] * (n_frames - len(frames))
    return sampled_frames

# extract fixed length 'seg_frames' frames from the video (parallel)
def extract_segment(i, seg_frames, all_video_frames, n_frames):
    interval_frames = all_video_frames[(i * seg_frames):((i + 1) * seg_frames)]
    return sample_frames(interval_frames, n_frames)

def generate_final_video(selected_clips, bgm_audio, bgm_duration, output_file):
    final_video = concatenate_videoclips(selected_clips, method="compose")
    # Crop the final video to match the BGM duration
    if final_video.duration > bgm_duration:
        final_video = final_video.subclipped(0, bgm_duration)

    # Set the extracted BGM as the audio track for the final video
    # final_video.audio = bgm_audio

    # Write the final video file
    final_video.write_videofile(output_file, audio_codec="aac", threads=12)

    return final_video


