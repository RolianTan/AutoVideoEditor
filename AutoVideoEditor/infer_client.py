import streamlit as st
import plotly.graph_objects as go
import requests
import base64

SERVER_URL = "http://44.197.249.242:5000"

# Initialize session state to manage pages and selected segment
if "current_page" not in st.session_state:
    st.session_state.current_page = "param_setting"

if "selected_segment" not in st.session_state:
    st.session_state.selected_segment = None

# PART: Parameter_setting page
if st.session_state.current_page == "param_setting":
    # Streamlit UI for collecting user inputs
    st.title("Inference Client")
    st.write("Provide the arguments settings for inference.")

    # Input fields for arguments
    exp_dir = st.text_input("Experiment Directory", "exp01")
    pretrained = st.text_input("Pretrained Weight File (.pt)", "../exp01/best_model.pt")
    data_dir = st.text_input("Data Directory", "inference")
    music_name = st.text_input("Music Directory", "bgm.mp3")
    video_name = st.text_input("Video Name", "footage_video.mp4")
    json_name = st.text_input("JSON File Name", "../inference/feature_code.json")
    device = st.selectbox("Device", ["cuda", "cpu"], index=0)
    fps = st.number_input("Inference FPS", min_value=1, max_value=60, value=10)
    size = st.number_input("Frame Size", min_value=1, value=224)
    n_frames = st.number_input("Number of Frames", min_value=1, value=8)
    out_mash_video = st.text_input("Output Mashup Video Name", "mash_up.mp4")
    t = st.number_input("Temperature for Projection", min_value=0.01, max_value=1.0, value=0.07, step=0.01)
    projection_dim = st.number_input("Projection Dimension", min_value=1, value=768)
    topk = st.number_input("Top-K Video Candidates", min_value=1, value=3)

    # Button to submit
    if st.button("Start Inference Preparation"):
        # Collect arguments into a dictionary
        args = {
            "exp_dir": exp_dir,
            "pretrained": pretrained,
            "data_dir": data_dir,
            "music_name": music_name,
            "video_name": video_name,
            "json_name": json_name,
            "device": device,
            "fps": fps,
            "size": size,
            "n_frames": n_frames,
            "out_mash_video": out_mash_video,
            "t": t,
            "projection_dim": projection_dim,
            "topk": topk,
        }

        # Send the arguments to the server
        response = requests.post(f"{SERVER_URL}/inference_prep", json=args)
        # Store the server response in session state
        result = response.json().get("message", "Fail")
        st.success(result)

    if st.button("Next"):
        # navigate to the next page
        st.session_state.current_page = "main"
        st.rerun()

# PART: Inference timeline page
elif st.session_state.current_page == "main":
    st.set_page_config(layout="wide")
    st.title("Automatic Video Editing (AVE) - Inference")
    #PART: feature editing container
    with st.container(border=True):
        st.subheader("Feature Editing")
        if "feature_info" not in st.session_state:
            st.session_state.feature_info = {"start_time": 0, "end_time": 0, "caption": ""}
        time_start = st.session_state.get("feature_info").get('start_time')
        time_end = st.session_state.get("feature_info").get('end_time')
        feature = st.session_state.get("feature_info").get('caption')
        # feature infor
        st.write(f"Time Start:{time_start}")
        st.write(f"Time End:{time_end}")
        st.session_state.feature_info['caption'] = st.text_input("Feature", value=feature)
    #PART: music review container
    with st.container(border=True):
        st.subheader("Music Segment Review")
        if "music_seg" not in st.session_state:
            st.session_state.music_seg = None
        music_seg = st.session_state.get("music_seg")
        st.audio(music_seg, format='audio/mp3')
    #PART: video selection container
    with st.container(border=True):
        st.subheader("Video Selection")
        # Initialize session state for videos
        if "videos" not in st.session_state:
            st.session_state.videos = []
        if "final" not in st.session_state:
            st.session_state.final = False
        # continuously display the video for selection
        videos = st.session_state.get("videos")
        if videos:
            if "video_idx" not in st.session_state:
                st.session_state.video_idx = -1
            # display videos in a row
            cols = st.columns(len(videos))
            for i, video_bs64 in enumerate(videos):
                video = base64.b64decode(video_bs64)
                with cols[i]:
                    st.video(video, format='video/mp4')
                    if st.checkbox(f"Select Video {i}", key=f"{i}"):
                        st.session_state.video_idx = i
        if st.session_state.final:
            st.success("Final Video is ready! Click 'Preview'.")

    #PART: button container
    with st.container(border=True):
        cols = st.columns(6)
        # PART: Get Feature
        if cols[0].button("Get Feature", use_container_width=True):
            response = requests.get(f"{SERVER_URL}/get_feature")
            feature_info = response.json().get("feature_info")
            st.session_state.feature_info = feature_info
            st.rerun()
        if cols[1].button("Get Music Seg", use_container_width=True):
            response = requests.get(f"{SERVER_URL}/get_music")
            music_seg = response.json().get("music_seg")
            music_seg = base64.b64decode(music_seg)
            st.session_state.music_seg = music_seg
            st.rerun()
        # PART: Inference
        if cols[2].button("Inference", use_container_width=True):
            response = requests.post(f"{SERVER_URL}/inference", json=st.session_state.feature_info)
            if response.status_code == 200:
                videos = response.json().get("videos")
                st.success(f"{len(videos)}")
                # Update new videos in session state
                st.session_state.videos = videos
                st.rerun()
            elif response.status_code == 201:
                st.session_state.final_video = response.json().get("final_video")[0]
                st.session_state.final = True
                st.rerun()
            else:
                st.error("Fail")
        # PART: Continue
        if cols[3].button("Continue", use_container_width=True):
            selected_video_idx = st.session_state.get("video_idx")
            response = requests.post(f"{SERVER_URL}/select_video", json={"idx": selected_video_idx})
            msg = response.json().get("message", "Fail")
            st.success(msg)
            # TODO: create a runtime cache, can be initialized for each step
            # clear videos, go to next inference step
            st.session_state.videos = []
            st.rerun()
        # PART: Show Selection (Debug)
        if cols[4].button("Show_Selection", use_container_width=True):
            request = requests.get(f"{SERVER_URL}/show_list")
            if request.status_code == 200:
                selected_idx = request.json().get("selected_idx")
                st.write(selected_idx)
            else:
                st.error("Fail")
        if cols[5].button("Preview", use_container_width=True):
            st.session_state.current_page = "preview_final_video"
            st.rerun()

elif st.session_state.current_page == "preview_final_video":
    st.title("Automatic Video Editing (AVE) - Final Video Preview")
    final_video = st.session_state.get("final_video")
    video = base64.b64decode(final_video)
    st.video(video, format='video/mp4')





