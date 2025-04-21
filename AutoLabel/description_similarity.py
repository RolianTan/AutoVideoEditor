from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load model
model = SentenceTransformer('all-mpnet-base-v2')  # Good trade-off between speed and quality

def flatten_all_description(desc_dict):
    """
    Convert dict-style description to a single string for encoding.
    """
    text_parts = [
        f"shot: {desc_dict['shot']}",
        f"angle: {desc_dict['angle']}",
        f"lighting: {desc_dict['lighting']}",
        f"text: {desc_dict['text']}",
        f"subject: {' '.join(desc_dict['subject'])}",
        f"description: {desc_dict['description']}"
    ]
    return ' | '.join(text_parts)

def compare_descriptions(desc1, desc2):
    """
    Compare two video frame descriptions using cosine similarity in embedding space.
    """
    text1 = desc1['description']
    text2 = desc2['description']

    embeddings = model.encode([text1, text2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    return similarity

# Example usage
desc1 = {
    "description": "Cinematic composition, warm natural lighting with pronounced sun flare, dynamic upward perspective, rich forest color palette, tranquil and nostalgic tone, strong depth and contrast, suggestive of spiritual or emotional narrative."
}

desc2 = {
    "description": "A visually striking scene characterized by a vibrant night sky, illuminated by celestial phenomena, with a dynamic composition suggesting cosmic wonder, blending tranquil tones with a sense of awe and mystery."
}

desc3 = {
    "description": "Dramatic and high-energy scene, vibrant colors with light trails from projectiles, expansive view of space, dynamic movement suggests action and intensity, evokes a sense of conflict or battle amidst a cosmic backdrop."
}
desc4 = {
    "description": "Intimate scene with close attention to detail, warm and soft lighting highlighting personal items, mixed emotions conveyed through character expressions, reflective and contemplative mood, sense of isolation amidst a cluttered workspace."
}

similarity_score1 = compare_descriptions(desc1, desc2)
similarity_score2 = compare_descriptions(desc2, desc3)
similarity_score3 = compare_descriptions(desc3, desc4)
print(similarity_score1)
print(similarity_score2)
print(similarity_score3)