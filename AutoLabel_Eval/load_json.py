import json
import ast
# Load JSON file into a dictionary
with open("test_frames_labels/city1.json", "r") as file:
    data = json.load(file)

print(data)