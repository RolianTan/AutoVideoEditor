import cv2
import base64
import json
from openai import OpenAI
import os
import random

client = OpenAI()
# export OPENAI_API_KEY="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"
def load_to_base64(img_path):
    img_base64 = []
    # load the comp images
    for path in img_path:
        img = cv2.imread(path)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        # convert to base64 type
        # encode: img -> numpy array
        _, buffer = cv2.imencode('.png', img)
        # encode: array -> base64 byte string
        img_byte = base64.b64encode(buffer)
        # decode: byte string -> string
        img_string = img_byte.decode('utf-8')
        img_base64.append(img_string)
    return img_base64


def image_selection(caption, model_name, comp_img_b64, position):
    general_prompt = "According to the caption, select the most appropriate image I provide you, only return the number of order of your selected image, start from 0 (eg. if it's the second image, return 1. if it's the first image, return 0. only the number no other tokens.). Here is the caption: " + caption
    # 1. create the prompt message with a general prompt 2. append the images to the prompt message
    # the prompt content should be [{text}{img}{img}....]
    PROMPT_CONTENT = [
        {
            "type": "text",
            "text": general_prompt,
        },
    ]
    # append all image
    for com_img in comp_img_b64:
        PROMPT_CONTENT.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{com_img}",
                "detail": "low"}
        })

    # 2. get the response
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": PROMPT_CONTENT,
            }
        ],
        max_tokens=10,
    )
    # print(response)
    select_no = response.choices[0].message.content
    # print(int(select_no))
    if int(select_no) == position:
        return True
    return False

if __name__ == '__main__':
    captions = [
        "[scene: Urban nightscape featuring towering skyscrapers, city lights illuminating the architecture, bustling atmosphere conveyed through visual elements. style: Long exposure enhances luminosity, emphasizes contrast between lights and shadows, captures vibrant city life essence. composition: Centralized focus on converging lines of buildings, balanced skyline, dynamic perspective inviting upward gaze. mood/tone: Energetic yet serene, vibrant yet tranquil, a harmonious juxtaposition of liveliness and calmness. story/meaning: Chronicles the ceaseless pulse of urban existence, narrates modernity and growth, reveals a city alive with energy and possibility.]",
        "[scene: A bustling urban environment at night with towering city skyscrapers, style: Night photography with a focus on artificial light, emphasizing the contrast between illuminated windows and the dark sky, composition: Wide-angle shot capturing the grandeur and scale of the architecture, mood/tone: Modern, vibrant, and dynamic, conveying the energetic pulse of city life, story/meaning: Represents the unyielding pace and resilience of urban environments, highlighting the allure and complexity of metropolitan life.]",
        "[scene: Urban nighttime cityscape with towering skyscrapers, style: Modern and vibrant with a focus on architectural symmetry, composition: Balanced with a strong vertical emphasis, using lines and angles to draw the eye upward, mood/tone: Energetic yet serene, capturing the city's lively but calm ambiance, story/meaning: Reflects the coexistence of human industry and tranquility, showcasing the continuous heartbeat of urban life.]",
    ]
    eval_models = ['gpt-4.5-preview', 'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo']
    comp_img_folder = 'img/img_comp/'
    target_img_path = ['img/test1.jpg']
    save_path = 'test_img_caption.json'
    # create the comp file path
    comp_img_path = [os.path.join(comp_img_folder, p) for p in os.listdir(comp_img_folder)]

    # load as base64 str
    comp_img_b64 = load_to_base64(comp_img_path)
    target_img_b64 = load_to_base64(target_img_path)[0]

    # put the target_img at random position in comp_img_list
    position = random.randint(0, len(comp_img_b64))
    comp_img_b64.insert(position, target_img_b64)


    vote_result = {}
    # image evaluation
    for i, caption in enumerate(captions):
        total_vote = 0
        correct_model = []
        for model_name in eval_models:
            if image_selection(caption, model_name, comp_img_b64, position):
                total_vote += 1
                correct_model.append(model_name)
        # log
        vote_result[caption] = total_vote
        print(f"Caption {i} - Total_Vote: {total_vote}; Correct_Model: {correct_model}")

    # find max count caption
    best_caption = max(vote_result, key=vote_result.get)
    print(f"The best caption is: \n {best_caption} \n with vote {vote_result[best_caption]}.")
    # print(result)










