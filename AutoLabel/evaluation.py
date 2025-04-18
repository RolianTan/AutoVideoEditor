import cv2
import base64
import json
from openai import OpenAI
import os
import random
import numpy as np

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# export OPENAI_API_KEY="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"
def load_to_base64(img_path):
    img_base64 = []
    # load the comp images
    for path in img_path:
        img = cv2.imread(path)
        img = cv2.resize(img, (640,480), interpolation=cv2.INTER_AREA)
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
    # Convert caption (a dict) to string representation
    if isinstance(caption, dict):
        caption = json.dumps(caption, ensure_ascii=False)
    general_prompt = "According to the label, select the most appropriate image I provide you, only return the number of order of your selected image, start from 0 (eg. if it's the second image, return 1. if it's the first image, return 0. only the number no other tokens.). Here is the caption: " + caption
    # 1. create the prompt message with a general prompt 2. append the images to the prompt message
    # the prompt content should be [{text}{img}{img}....]
    PROMPT_CONTENT = [
        {
            "type": "text",
            "text": general_prompt,
        },
    ]
    # append all image

    for comp_img in comp_img_b64:
        PROMPT_CONTENT.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{comp_img}",
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

def evaluate_dir(image_dir, labels_dirs, visualize = True):
    # Iterate through each image in the image directory
    for image_file in os.listdir(image_dir):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Find the corresponding labels file (same name but with .json)
        image_name, _ = os.path.splitext(image_file)
        labels = []
        for label_dir in labels_dirs:
            label_path = os.path.join(label_dir, f"{image_name}.json")
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    data = json.load(f)
                    labels.append(data)

        if not labels:
            print(f"No labels found for image: {image_file}")
            continue

        # Prepare the image paths
        image_path = os.path.join(image_dir, image_file)
        comp_img_folder = image_dir  # Assuming comparison images are in the same folder
        comp_img_path = [os.path.join(comp_img_folder, p) for p in os.listdir(comp_img_folder) if p != image_file]
        
        sample_comp_img_path = random.sample(comp_img_path, min(3, len(comp_img_path)))
        position = random.randint(0, len(sample_comp_img_path))
        sample_comp_img_path.insert(position, image_path)
        # Load images as base64
        all_img_b64 = load_to_base64(sample_comp_img_path)
        #target_img_b64 = load_to_base64([image_path])[0]

        # Insert the target image at a random position in the comparison list
        #sample_comp_img_b64 = random.sample(comp_img_b64, min(3, len(comp_img_b64)))
        

        # Evaluate captions
        eval_models = ['gpt-4o-mini', 'gpt-4o']
        vote_result = []
        for i, label in enumerate(labels):
            total_vote = 0
            correct_model = []
            for model_name in eval_models:
                if image_selection(label, model_name, all_img_b64, position):
                    total_vote += 1
                    correct_model.append(model_name)
            vote_result.append(total_vote)
            print(f"Image: {image_file}, Caption {i} - Total_Vote: {total_vote}; Correct_Model: {correct_model}")

        # Find the best caption
        if vote_result:
            best_label_id = vote_result.index(max(vote_result))
            print(f"Image: {image_file}, Best label: \n {labels[best_label_id]} \n with vote {vote_result[best_label_id]}.")
        if visualize:
            import matplotlib.pyplot as plt

            # Display the images with their votes
            fig, axes = plt.subplots(1, len(sample_comp_img_path), figsize=(15, 5))
            for idx, (img_b64, ax) in enumerate(zip(all_img_b64, axes)):
                # Decode the base64 image
                img_data = base64.b64decode(img_b64)
                img_array = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

                # Draw a green box around the true image
                if idx == position:
                    img_array = cv2.rectangle(img_array, (0, 0), (img_array.shape[1], img_array.shape[0]), (0, 255, 0), 10)

                # Display the image
                ax.imshow(img_array)
                ax.axis('off')

            # Display the caption
            plt.suptitle(f"Votes {vote_result[0]}: {labels[0]} \n Votes {vote_result[1]}: {labels[1]}", fontsize=12, y=0.85)
            
            plt.tight_layout()
            plt.show()

if __name__ == '__main__':

    image_dir = 'mad_scene_frames'
    labels_dirs = ['mad_scene_labels', "mad_scene_labels_c"]
    evaluate_dir(image_dir, labels_dirs)

    # captions = [
    #     "[scene: Urban nightscape featuring towering skyscrapers, city lights illuminating the architecture, bustling atmosphere conveyed through visual elements. style: Long exposure enhances luminosity, emphasizes contrast between lights and shadows, captures vibrant city life essence. composition: Centralized focus on converging lines of buildings, balanced skyline, dynamic perspective inviting upward gaze. mood/tone: Energetic yet serene, vibrant yet tranquil, a harmonious juxtaposition of liveliness and calmness. story/meaning: Chronicles the ceaseless pulse of urban existence, narrates modernity and growth, reveals a city alive with energy and possibility.]",
    #     "[scene: A bustling urban environment at night with towering city skyscrapers, style: Night photography with a focus on artificial light, emphasizing the contrast between illuminated windows and the dark sky, composition: Wide-angle shot capturing the grandeur and scale of the architecture, mood/tone: Modern, vibrant, and dynamic, conveying the energetic pulse of city life, story/meaning: Represents the unyielding pace and resilience of urban environments, highlighting the allure and complexity of metropolitan life.]",
    #     "[scene: Urban nighttime cityscape with towering skyscrapers, style: Modern and vibrant with a focus on architectural symmetry, composition: Balanced with a strong vertical emphasis, using lines and angles to draw the eye upward, mood/tone: Energetic yet serene, capturing the city's lively but calm ambiance, story/meaning: Reflects the coexistence of human industry and tranquility, showcasing the continuous heartbeat of urban life.]",
    # ]
    # #eval_models = ['gpt-4.5-preview', 'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo']
    # eval_models = ['gpt-4o-mini', 'gpt-4o']
    # comp_img_folder = 'mad_scene_frames'
    # target_img_path = ['mad_scene_frames/scene_8_frame0.jpg']
    # save_path = 'test_img_caption.json'
    # # create the comp file path
    # comp_img_path = [os.path.join(comp_img_folder, p) for p in os.listdir(comp_img_folder)]
    

    # # load as base64 str
    # comp_img_b64 = load_to_base64(comp_img_path)
    # target_img_b64 = load_to_base64(target_img_path)[0]

    # # put the target_img at random position in comp_img_list
    # # Sample 3 random images from comp_img_b64
    # sample_comp_img_b64 = random.sample(comp_img_b64, min(3, len(comp_img_b64)))
    # position = random.randint(0, len(sample_comp_img_b64))
    # sample_comp_img_b64.insert(position, target_img_b64)


    # vote_result = {}
    # # image evaluation
    # for i, caption in enumerate(captions):
    #     total_vote = 0
    #     correct_model = []
    #     for model_name in eval_models:
    #         if image_selection(caption, model_name, sample_comp_img_b64, position):
    #             total_vote += 1
    #             correct_model.append(model_name)
    #     # log
    #     vote_result[caption] = total_vote
    #     print(f"Caption {i} - Total_Vote: {total_vote}; Correct_Model: {correct_model}")

    # # find max count caption
    # best_caption = max(vote_result, key=vote_result.get)
    # print(f"The best caption is: \n {best_caption} \n with vote {vote_result[best_caption]}.")
    # # print(result)










