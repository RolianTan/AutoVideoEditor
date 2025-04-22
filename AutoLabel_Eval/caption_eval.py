import random
import csv
import openai
import pandas as pd
import os

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def image_selection(caption, model_name, all_videos, position):
    prompt = (
        "You are shown multiple anime keyframes, representing the start frame of different anime video scenes."
        "Based on the given caption, pick the video that best matches the visual content. "
        "Only return the *index number* of your selected video, starting from 0. " 
        "RESPONSE NO TEXT AT ANY SITUATION, NUMBER ONLY."
        "For example, if video 0 is your choice, response 0. Do not add any explanation. If you can't choose, just return the most likely one's index. \n\n"
        "Here is the caption:\n" + caption
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    # 1 to 3
    # Add images grouped by video index
    # for i, video in enumerate(all_videos):
    #     messages[0]["content"].append({
    #         "type": "text",
    #         "text": f"\nVideo {i}:"
    #     })
    #     for frame_b64 in video:
    #         messages[0]["content"].append({
    #             "type": "image_url",
    #             "image_url": {
    #                 "url": f"data:image/jpeg;base64,{frame_b64}",
    #                 "detail": "low"
    #             }
    #         })

    # 1 to 1
    for i, frame in enumerate(all_videos):
        messages[0]["content"].append({
            "type": "text",
            "text": f"\nVideo {i}:"
        })
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame}",
                "detail": "low"
            }
        })
    # print('here')
    # API call
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=4,
    )

    # print(response.choices[0].message.content.strip())

    if int(response.choices[0].message.content.strip()) == position:
        return 1
    else:
        return 0

def eval_captions(captions, target_video, distractor_videos, eval_models, idx):
    print("Start Eval")
    scores = {}
    model_votes = {}

    # Randomize frame order
    position = random.randint(0, len(distractor_videos))
    all_videos = distractor_videos[:]
    all_videos.insert(position, target_video)

    for caption in captions:
        votes = 0
        correct_models = []
        for model in eval_models:
            if image_selection(caption, model, all_videos, position):
                votes += 1
                correct_models.append(model)
        scores[caption] = votes
        model_votes[caption] = correct_models

    # Create and store dataframe
    df = pd.DataFrame([
        {"caption": c, "score": scores[c], "models_correct": ", ".join(model_votes[c])}
        for c in captions
    ])
    os.makedirs("caption_process", exist_ok=True)
    df.to_csv(f"caption_process/{idx}_caption_list.csv", index=False)

    best_caption = max(scores, key=scores.get)
    best_score = scores[best_caption]

    print(f"Scene {idx} Caption Saved with Accuracy: {best_score / len(eval_models)}")

    return best_caption
