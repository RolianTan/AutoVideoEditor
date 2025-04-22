import openai
import base64
from .util import read_file, load_prompt
import json
import os
import ast
from PIL import Image
import io
import time
from .caption_eval import eval_captions
# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def get_frame_description(image_path, output_path):
def get_frame_description(sampled_frames, frames_compare, idx, size=3):
    system_prompt = load_prompt("AutoLabel_Eval/prompts_mad/system.md")
    user_prompt = load_prompt("AutoLabel_Eval/prompts_mad/user.md")
    assistant_prompt = load_prompt("AutoLabel_Eval/prompts_mad/assistant.md")

    # create image input content list
    image_contents = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{sampled_frames}"}}
        # for img in sampled_frames
    ]
    # generate captions
    caption_candidates = []
    for ct in range(size):
        ct += 1
        while True:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_prompt},
                            # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            *image_contents
                        ]},
                        {"role": "assistant", "content": assistant_prompt}
                    ],
                    max_tokens=1000
                )
                video_caption = response.choices[0].message.content
                caption_candidates.append(video_caption)
                break
            except Exception as e:
                print(f"Error: {e}; Retrying.")
                time.sleep(1)  # Wait for a bit before retrying

    # Print the candidate response
    print(caption_candidates)

    # evaluate captions
    # compare with other random video frames, and find best candidates
    eval_models = ['gpt-4o', 'gpt-4.5-preview', 'gpt-4-turbo']
    best_caption = eval_captions(
        captions=caption_candidates,
        target_video=sampled_frames,
        distractor_videos=frames_compare,
        eval_models=eval_models,
        idx=idx
    )

    return best_caption
    # frame_description_dict = ast.literal_eval(frame_description)
    # with open(output_path, "w", encoding="utf-8") as json_file:
    #     json.dump(frame_description_dict, json_file, indent=4)
# get_frame_description("test_frames/city1", "test_frames_labels/city1.")
