import clip
import pickle
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dotenv import load_dotenv

load_dotenv()

images_path = os.getenv('IMAGES_PATH')
model_name = os.getenv("MODEL_NAME")
device = "cuda" if clip.available() else "cpu"

image_paths = []
files = os.listdir(images_path)
for i, file in enumerate(files):
    image_paths.append(os.path.join(images_path, file))

# Load the CLIP model
model, preprocess = clip.load(model_name, device=device)

# Load the saved image embeddings
with open("image_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)


def search(query):
    # Encode the text
    text_input = clip.tokenize([query]).to(device)
    text_features = model.encode_text(text_input)

    # Convert the image embeddings back to Torch tensors
    image_features = np.concatenate(embeddings, axis=0)

    # Perform similarity search
    with torch.no_grad():
        similarity_scores = (text_features @ image_features.T).squeeze(0)
    sorted_indices = similarity_scores.argsort(descending=True)

    # Print the top matching images
    top_k = 3
    for i in range(top_k):
        image_path = image_paths[sorted_indices[i]]
        similarity_score = similarity_scores[sorted_indices[i]].item()
        st.write(
            f"Image: {image_path}, Similarity Score: {similarity_score:.4f}")
        img = mpimg.imread(image_path)
        st.image(img)


# Streamlit app
st.title("Image Search")
query = st.text_input("Enter a search query:")
if query:
    search(query)
