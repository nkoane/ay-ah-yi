import clip
import pickle
import sys
import os
import numpy
import torch
from dotenv import load_dotenv
from util import get_device
import streamlit as st

load_dotenv()

images_path = os.getenv('IMAGES_PATH')
model_name = os.getenv("MODEL_NAME")
device = "cuda" if torch.cuda.is_available() else "cpu"

image_paths = []
files = os.listdir(images_path)
for i, file in enumerate(files):
    image_paths.append(os.path.join(images_path, file))

model, preprocess = clip.load(model_name, device=device)


def seed():
    from PIL import Image
    embeddings = []
    for i, image_path in enumerate(image_paths):
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_embedding = model.encode_image(image)
        with torch.no_grad():
            embeddings.append(image_embedding.cpu().numpy())
        print(f"image_embedded: {i}:{len(image_paths)}")

    # Save the embeddings to a file
    with open("image_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    print(f"image_embedded done.")


def query(text):
    # Load the saved image embeddings
    with open("image_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    # Encode the text
    text_input = clip.tokenize([text]).to(device)
    text_features = model.encode_text(text_input)

    # image_features = torch.tensor(embeddings).to(device)
    image_features = numpy.concatenate(embeddings, axis=0)

    # Perform similarity search
    with torch.no_grad():
        similarity_scores = (text_features @ image_features.T).squeeze(0)
    sorted_indices = similarity_scores.argsort(descending=True)

    top_k = 3
    image_results = []
    for i in range(top_k):
        image_path = image_paths[sorted_indices[i]]
        similarity_score = similarity_scores[sorted_indices[i]].item()
        image_results.append({
            'image_path': image_path,
            'similarity_score': similarity_score
        })

    return image_results


def search():
    while True:
        text = input('Query: ')
        if text == 'q':
            break
        results = query(text)
        print(results)


def ui():
    st.title("Image Search")
    question = st.text_input("Query:")
    if query:
        results = query(question)
        for result in results:
            st.write(
                f"Image: {result['image_path']}, Similarity Score: {result['similarity_score']:.4f}"
            )
            st.image(result['image_path'])


if len(sys.argv) < 2:
    print('please provide a command')
    exit(1)
elif sys.argv[1] and sys.argv[1] not in ['seed', 'search', 'ui']:
    print('please provide a valid command: search + seed')
    exit(1)
elif sys.argv[1] == 'seed':
    seed()
elif sys.argv[1] == 'search':
    search()
elif sys.argv[1] == 'ui':
    ui()
