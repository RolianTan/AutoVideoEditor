from flask import Flask, request, jsonify
from inference_funcs import extract_video_frames, sample_frames, extract_segment, generate_final_video, intervals_overlap
from moviepy import VideoFileClip, AudioFileClip
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

# Flask-Session configuration
# app.config["SESSION_TYPE"] = "filesystem"
# app.config["SESSION_PERMANENT"] = True
# app.config["SESSION_FILE_DIR"] = "./flask_session/"
# app.config["SECRET_KEY"] = "supersecretkey"
#
# Session(app)
session = {}

# PART: Utility
def video_transfer(video_list, with_audio):
    v_64_list = []
    for v in video_list:
        v = v.resized(height=360)  # resize the video to 360p
        temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".mp4")
        v.write_videofile(temp_file.name, codec='libx264', audio=with_audio)

        # read to buffer and change to base64
        with open(temp_file.name, "rb") as f:
            buffer = io.BytesIO(f.read())
        v_64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        v_64_list.append(v_64)
        # clear
        v.close()
    return v_64_list


# PART: Inference Preparation
@app.route('/inference_prep', methods=['POST'])
def inference_prep():
    # PART: Parse configuration
    config = argparse.Namespace(**request.get_json())
    session["config"] = config

    # PART: Load the model and pre-trained
    vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(config.device)
    vision_input_dim = vision_model.config.hidden_size
    text_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").text_model.to(config.device)
    text_input_dim = text_model.config.hidden_size
    model = AutoClipModel(vision_model, vision_input_dim, text_model, text_input_dim, config.projection_dim,
                          config.t).to(config.device)
    model.load_state_dict(torch.load(config.pretrained, weights_only=True))
    model.eval()
    session["model"] = model

    # PART: Load JSON file
    json_path = os.path.join(config.data_dir, config.json_name)
    with open(json_path, 'r') as f:
        session["intervals_data"] = json.load(f)
    # session["N"] = len(session["intervals_data"])
    # debug
    session["N"] = 3
    if "step" not in session:
        session["step"] = 0

    # PART: Extract all frames (footage video)
    video_path = os.path.join(config.data_dir, config.video_name)
    raw_video = VideoFileClip(video_path)
    session["raw_video"] = raw_video
    session["total_time"] = raw_video.duration
    session["all_video_frames"] = extract_video_frames(video_path, config.size, config.fps)

    # PART: Load BGM
    bgm_path = os.path.join(config.data_dir, config.music_name)
    bgm_audio = AudioFileClip(bgm_path)
    session["bgm_audio"] = bgm_audio
    session["bgm_duration"] = bgm_audio.duration

    # PART: Set up tokenizer for text processing
    session["tokenizer"] = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    return jsonify({"message": "Set the parameter successfully and finished the inference preparation."}), 200


@app.route('/inference', methods=['POST'])
def inference():
    client_feature = request.get_json()
    if "used_video_clips" not in session:
        session["used_video_clips"] = []
    if "mashup_videos" not in session:
        session["mashup_videos"] = []
    # PART: Process each interval and find the best video segment
    # PART: End of the inference
    if session["step"] < session["N"]:
        # PART: Slice from footage all frames (duration length)
        start_time = client_feature['start_time']
        end_time = client_feature['end_time']
        caption = client_feature['caption']
        t = end_time - start_time
        # number of segments
        n = int(session["total_time"] // t)
        # print(start_time)
        # print(end_time)
        # number of frames for each segment
        seg_frames = int(t * session["config"].fps)
        # Tokenize the caption
        text_input = session["tokenizer"](caption, return_tensors='pt', padding='max_length', truncation=True,
                                          max_length=77).to(session["config"].device)
        # Parallel: Extract segments
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(extract_segment, i, seg_frames, session["all_video_frames"],
                                       session["config"].n_frames) for i in range(n)]
            video_seg_frames = [future.result() for future in futures]
        # Stack all frames for vectorization
        video_seg_frames = np.array(video_seg_frames)
        video_seg_frames = torch.from_numpy(video_seg_frames).permute(0, 1, 4, 2, 3).to(session["config"].device)
        # print(f"{video_seg_frames.shape}")
        # PART: pair the best duration for the caption
        with torch.no_grad():
            video_features = session["model"].vision_model(video_seg_frames)
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)

            text_features = session["model"].text_model(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = session["model"].logit_scale.exp()
            # print(f"video_features shape: {video_features.shape}")
            # print(f"text_features shape: {text_features.shape}")
            # N = session["N"]
            # all_frame_shape = session["all_video_frames"].shape
            # print(N)
            # print(f"all_frame {all_frame_shape}")
            similarity = logit_scale * text_features @ video_features.t()
            sorted_idx = torch.argsort(similarity, dim=1, descending=True).squeeze()

        # PART: filter used clips
        filtered_idx = []
        for idx in sorted_idx:
            idx = idx.item()
            candidate_start = round(idx * t, 2)
            candidate_end = min(round((idx + 1) * t, 2), session["total_time"])
            if not any(
                    intervals_overlap(candidate_start, candidate_end, used_start, used_end) for used_start, used_end in
                    session["used_video_clips"]):
                filtered_idx.append(idx)

        # PART: return topk clips
        if len(filtered_idx) < session["config"].topk:
            candidates_clips = sorted_idx[:session["config"].topk].tolist()
        else:
            candidates_clips = filtered_idx[:session["config"].topk]
        # PART: update current candidate video and time list
        session["candidate_clips_videos"] = []
        session["candidates_clips_time"] = []
        for i in range(len(candidates_clips)):
            candidate_start = candidates_clips[i] * t
            candidate_end = min((candidates_clips[i] + 1) * t, session["total_time"])
            session["candidates_clips_time"].append((candidate_start, candidate_end))
            session["candidate_clips_videos"].append(session["raw_video"].subclipped(candidate_start, candidate_end))
        # PART: json format videos and send to client
        v_64_list = video_transfer(session["candidate_clips_videos"], False)
        # iterate to next step of the interval
        session["step"] = session["step"] + 1
        if session["step"] >= session["N"]:
            final_video = generate_final_video(session["mashup_videos"], session["bgm_audio"], session["bgm_duration"],
                                               os.path.join(session["config"].data_dir,
                                                            session["config"].out_mash_video))
            final_video_v64 = video_transfer([final_video], True)
            # inference finished
            return jsonify({"final_video": final_video_v64}), 201
        else:
            return jsonify({"videos": v_64_list}), 200


@app.route('/select_video', methods=['POST'])
def select_video():
    if "selected_idx" not in session:
        session["selected_idx"] = []
    # for testing
    session["selected_idx"].append(request.get_json().get("idx"))

    # Append this chosen clip video to the mashup list
    session["mashup_videos"].append(session["candidate_clips_videos"][request.get_json().get("idx")])
    # Include this chosen clip time in used_video_clips
    session["used_video_clips"].append((
        session["candidates_clips_time"][request.get_json().get("idx")][0],
        session["candidates_clips_time"][request.get_json().get("idx")][1],
    ))

    return jsonify({"message": "Success"}), 200

@app.route('/show_list', methods=['GET'])
def show_list():
    return jsonify({"selected_idx": session["selected_idx"]}), 200


@app.route('/get_feature', methods=['GET'])
def get_feature():
    feature_info = session["intervals_data"][session["step"]]
    return jsonify({"feature_info": feature_info}), 200

@app.route('/get_music', methods=['GET'])
def get_music():
    start_time = session["intervals_data"][session["step"]]["start_time"]
    end_time = session["intervals_data"][session["step"]]["end_time"]
    # load instant raw music
    bgm_path = os.path.join(session["config"].data_dir, session["config"].music_name)
    raw_music = AudioFileClip(bgm_path)
    # crop the seg
    music_seg = raw_music.subclipped(start_time, end_time)
    # encode to base64
    temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".mp3")
    music_seg.write_audiofile(temp_file.name)
    with open(temp_file.name, "rb") as f:
        buffer = io.BytesIO(f.read())
    v_64_music_seg = base64.b64encode(buffer.getvalue()).decode('utf-8')
    # close
    music_seg.close()
    return jsonify({"music_seg": v_64_music_seg}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)