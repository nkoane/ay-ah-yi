import clip
import pickle
import sys
import os
import numpy
import torch
from dotenv import load_dotenv
from util import get_device

load_dotenv()

images_path = os.getenv('IMAGES_PATH')
model_name = os.getenv("MODEL_NAME")
device = "cuda" if torch.cuda.is_available() else "cpu"

image_paths = []
files = os.listdir(images_path)
for i, file in enumerate(files):
    image_paths.append(os.path.join(images_path, file))


def seed():
    from PIL import Image

    # Load the CLIP model
    model, preprocess = clip.load(model_name, device=device)

    # Encode the images and save the embeddings

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


def search():

    # Load the CLIP model
    model, preprocess = clip.load(model_name, device=device)

    # Load the saved image embeddings
    with open("image_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    # Encode the text
    while True:
        text = input('Query: ')
        if text == 'q':
            break
        text_input = clip.tokenize([text]).to(device)
        text_features = model.encode_text(text_input)

        # Convert the image embeddings back to Torch tensors
        # image_features = torch.from_numpy(embeddings).to(device)

        # image_features = torch.tensor(embeddings).to(device)
        image_features = numpy.concatenate(embeddings, axis=0)

        # Perform similarity search
        with torch.no_grad():
            similarity_scores = (text_features @ image_features.T).squeeze(0)
        sorted_indices = similarity_scores.argsort(descending=True)

        # Print the top matching images
        top_k = 3
        for i in range(top_k):
            image_path = image_paths[sorted_indices[i]]
            similarity_score = similarity_scores[sorted_indices[i]].item()
            print(
                f"Image: {image_path}, Similarity Score: {similarity_score:.4f}")
            # BEGIN: 8f7d6b3f5d0c
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.show()


if len(sys.argv) < 2:
    print('please provide a command')
    exit(1)
elif sys.argv[1] and sys.argv[1] not in ['seed', 'search']:
    print('please provide a valid command: search + seed')
    exit(1)
elif sys.argv[1] == 'seed':
    seed()
elif sys.argv[1] == 'search':
    search()
