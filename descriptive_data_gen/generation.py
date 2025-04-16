import cv2
import base64
import json
from openai import OpenAI
import os

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


def caption_gen(target_img_b64, comp_img_b64):
    general_prompt = ("Generate a professional and concise description for the first image that captures the video's visual characteristics, this description should distinguish the first image from the rest of images I provide you. \
    Focus on *high-level photography features*, using commas to separate each term. \
    Do not include specific objects, summarized description should include scene, style, composition, mood/tone, story/meaning. \
    Format the response as follows: [scene:..., style:..., composition:..., mood/tone:..., story/meaning:...]. \
    Fill each category with professional photography descriptive sentences. \
    Limit the overall description to 300 tokens and focus only on visual and graphical elements, ignoring any text or dialogue. \
    Your response should only be the description of the first image. Do not include 'first image', 'other image' etc any comparison words. Don't include newline sign.")

    # 1. create the prompt message with a general prompt 2. append the images to the prompt message
    # the prompt content should be [{text}{img}{img}....
    # add the target image
    target_img = target_img_b64[0]
    PROMPT_CONTENT = [
        {
            "type": "text",
            "text": general_prompt,
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{target_img}",
                          "detail": "low"}
        },
    ]
    # append all comp image
    for com_img in comp_img_b64:
        PROMPT_CONTENT.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{com_img}",
                "detail": "low"}
        })

    # 2. get the response
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                "role": "user",
                "content": PROMPT_CONTENT,
            }
        ],
        max_tokens=500,
    )
    # print(response)
    caption = response.choices[0].message.content
    return caption

if __name__ == '__main__':
    comp_img_folder = 'img/img_comp/'
    target_img_path = ['img/test1.jpg']
    save_path = 'test_img_caption.json'
    # create the comp file path
    comp_img_path = [os.path.join(comp_img_folder, p) for p in os.listdir(comp_img_folder)]

    # load as base64 str
    comp_img_b64 = load_to_base64(comp_img_path)
    target_img_b64 = load_to_base64(target_img_path)

    # write the caption to the json output
    caption = caption_gen(target_img_b64, comp_img_b64)
    with open(save_path, 'w') as f:
        json.dump(caption, f, indent=4)
    print(caption)










