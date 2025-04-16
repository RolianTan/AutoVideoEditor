from inference_funcs import extract_video_frames, extract_segment, generate_final_video, intervals_overlap
from flask import Flask, request, jsonify
from moviepy.editor import VideoFileClip, AudioFileClip
import tempfile
import base64
import io
import argparse
import torch
import os
import numpy as np
from model import AutoClipModel
from transformers import CLIPTokenizer, CLIPModel, CLIPVisionModel
from concurrent.futures import ThreadPoolExecutor
import json

app = Flask(__name__)

# Global variables
config = None
intervals_data = []
total_time = 0
tokenizer = None
model = None
all_video_frames = []
used_video_clips = []
candidate_clips_videos = []
candidates_clips_time = []
mashup_videos = []
raw_video = None
bgm_audio = None
bgm_duration = 0
N = 0 # total steps
step = 0 # current step

temp_dir = '/temp'
selected_idx = []

# PART: Utility
def video_transfer():
    v_64_list = []
    for v in candidate_clips_videos:
        v = v.resize(height=360) # resize the video to 360p
        # store the candidates in the temp_file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        v.write_videofile(temp_file.name, codec='libx264', audio=False)

        # read to buffer and change to base64
        with open(temp_file.name, "rb") as video_file:
            buffer = io.BytesIO(video_file.read())
        v_64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        v_64_list.append(v_64)

        # close and clear
        os.unlink(temp_file.name)
        v.close()
    return v_64_list


# PART: Inference
@app.route('/inference_prep', methods=['POST'])
def inference_prep():
    global config, intervals_data, total_time, tokenizer, model, raw_video, bgm_audio, bgm_duration, all_video_frames, N
    config = argparse.Namespace(**request.get_json())

    # PART: Load the model and pre-trained
    # clip-vit
    vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(config.device)
    vision_input_dim = vision_model.config.hidden_size
    # clip-text
    text_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").text_model.to(config.device)
    text_input_dim = text_model.config.hidden_size
    model = AutoClipModel(vision_model, vision_input_dim, text_model, text_input_dim, config.projection_dim,
                          config.t).to(config.device)
    model.load_state_dict(torch.load(os.path.join(config.exp_dir, config.pretrained)))
    model.eval()

    # PART: Load JSON file
    json_path = os.path.join(config.data_dir, config.json_name)
    with open(json_path, 'r') as f:
        intervals_data = json.load(f)
    N = len(intervals_data)

    # PART: Extract all frames (footage video)
    video_path = os.path.join(config.data_dir, config.video_name)
    raw_video = VideoFileClip(video_path)
    total_time = raw_video.duration
    all_video_frames = extract_video_frames(video_path, config.size, config.fps)  # Shape: (total_frames, c, h, w)

    # PART: Load BGM
    bgm_path = os.path.join(config.music_dir, config.reference_video)
    bgm_audio = AudioFileClip(bgm_path)
    bgm_duration = bgm_audio.duration

    # Set up tokenizer for text processing
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    return jsonify({"message": "Set the parameter successfully and finished the inference preparation."}), 200


@app.route('/inference', methods=['GET'])
def inference():
    # PART: Process each interval and find the best video segment
    global step, used_video_clips, candidate_clips_videos, candidates_clips_time
    if step >= N:
        generate_final_video(mashup_videos, bgm_audio, bgm_duration, os.path.join(config.data_dir, config.out_mash_video))
        # inference finished
        return jsonify({"video": []}), 201
    else:
        # PART: Slice from footage all frames (duration length)
        start_time = intervals_data[step]['start_time']
        end_time = intervals_data[step]['end_time']
        caption = intervals_data[step]['caption']
        t = end_time - start_time
        # number of segments
        n = int(total_time // t)
        # number of frames for each segment
        seg_frames = int(t * config.fps)
        # Tokenize the caption
        text_input = tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=77).to(config.device)
        # Parallel: Extract segments
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(extract_segment, i, seg_frames, all_video_frames, config.n_frames) for i in
                       range(n)]
            video_seg_frames = [future.result() for future in futures]
        # Stack all frames for vectorization
        video_seg_frames = np.array(video_seg_frames)
        video_seg_frames = torch.from_numpy(video_seg_frames).permute(0, 1, 4, 2, 3).to(config.device)

        # PART: pair the best duration for the caption
        with torch.no_grad():
            video_features = model.vision_model(video_seg_frames)
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)

            text_features = model.text_model(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            similarity = logit_scale * text_features @ video_features.t()
            sorted_idx = torch.argsort(similarity, dim=1, descending=True).squeeze()

        # PART: filter used clips
        filtered_idx = []
        for idx in sorted_idx:
            idx = idx.item()
            candidate_start = round(idx * t, 2)
            candidate_end = min(round((idx + 1) * t, 2), total_time)
            if not any(
                    intervals_overlap(candidate_start, candidate_end, used_start, used_end) for used_start, used_end in
                    used_video_clips):
                filtered_idx.append(idx)

        # PART: return topk clips
        # If filtered_idx has not sufficient clips (less than top k)
        # TODO: any better way to handle this?
        if len(filtered_idx) < config.topk:
            candidates_clips = sorted_idx[:config.topk].tolist()
        else:
            candidates_clips = filtered_idx[:config.topk]
        # PART: update current candidate video and time list
        candidate_clips_videos = []
        candidates_clips_time = []
        for i in range(len(candidates_clips)):
            candidate_start = candidates_clips[i] * t
            candidate_end = min((candidates_clips[i] + 1) * t, total_time)
            candidates_clips_time.append((candidate_start, candidate_end))
            candidate_clips_videos.append(raw_video.subclip(candidate_start, candidate_end))
        # PART: json format videos and send to client
        v_64_list = video_transfer()
        step += 1
        return jsonify({"videos": v_64_list}), 200

@app.route('/select_video', methods=['POST'])
def select_video():
    global mashup_videos, used_video_clips
    idx = request.get_json().get("idx")
    # for testing
    selected_idx.append(idx)

    # Append this chosen clip video to the mashup list
    mashup_videos.append(candidate_clips_videos[idx])
    # Include this chosen clip time in used_video_clips
    used_video_clips.append((candidates_clips_time[idx][0], candidates_clips_time[idx][1]))

    return jsonify({"message": "Success"}), 200


@app.route('/show_list', methods=['GET'])
def show_list():
    global selected_idx
    return jsonify({"selected_idx": selected_idx}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)