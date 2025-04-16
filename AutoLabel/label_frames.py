import openai
import base64
from util import read_file, load_prompt
import json
import os
import ast
from PIL import Image
import io
import time
# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Convert the image to base64 format
def encode_image(image_path):
    # Open the image
    with Image.open(image_path) as img:
        print(f"Original size: {img.size}")  # (width, height)

        # Resize to (width=640, height=480)
        resized_img = img.resize((640, 480))

        # Save to buffer in memory
        buffer = io.BytesIO()
        resized_img.save(buffer, format="JPEG")
        buffer.seek(0)

        # Encode to base64
        return base64.b64encode(buffer.read()).decode("utf-8")

# Path to your image

def get_frame_description(image_path, output_path):
    base64_image = encode_image(image_path)
    system_prompt = load_prompt("AutoLabel/prompts_mad/system.md")
    user_prompt = load_prompt("AutoLabel/prompts_mad/user.md")
    assistant_prompt = load_prompt("AutoLabel/prompts_mad/assistant.md")
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}  # Include base64 image
                    ]},
                    {"role": "assistant", "content": assistant_prompt}
                ],
                max_tokens=100
            )
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)  # Wait for a bit before retrying

    frame_description = response.choices[0].message.content
    # Print the response
    print(frame_description)
    frame_description_dict = ast.literal_eval(frame_description)
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(frame_description_dict, json_file, indent=4)

def process_frames(input_folder, output_folder, c = 1):
    """Processes every c-th frame in the input folder and saves JSON output in the output folder."""
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files in the folder (sorted to maintain order)
    image_files = sorted([
        f for f in os.listdir(input_folder) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    # Process every c-th frame
    for i in range(0, len(image_files), c):
        image_filename = image_files[i]
        image_path = os.path.join(input_folder, image_filename)

        # Generate corresponding JSON output filename
        json_filename = os.path.splitext(image_filename)[0] + ".json"
        output_path = os.path.join(output_folder, json_filename)

        # Call function to process frame
        get_frame_description(image_path, output_path)

        print(f"Processed: {image_filename} → {json_filename}")

process_frames("scene_frames", "mad_template_labels", 1)
# get_frame_description("test_frames/city1", "test_frames_labels/city1.")
