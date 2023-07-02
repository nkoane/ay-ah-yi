import random
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
import clip
from math import sqrt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from util import *


def display_image(image_path):
    # plt.close()
    img = mpimg.imread(image_path)
    imgplot = plt.imshow(img)
    plt.show(block=False)


device = 'cpu'  # get_device()
directory = "./data/images/"
img_path = directory + random.choice(os.listdir(directory))

paths = []

# BEGIN: 8f6b7d5d7d5d


def display_images(image_paths):
    for path in image_paths:
        plt.imshow(plt.imread(path))
        plt.show(block=False)


directory = "./data/images/"
image_paths = [directory + filename for filename in os.listdir(directory)]
random.shuffle(image_paths)
display_images(image_paths[:3])
# END: 8f6b7d5d7d5d

while True:
    query = input("Enter query: ")
    if query == 'q':
        break

exit()


# model, preprocess = clip.load("ViT-B/32", device=device)
model, preprocess = clip.load("ViT-L/14", device=device)
image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)


while True:
    query = input("Enter query: ")
    if query == 'q':
        break
    text_labels = [query]
    text = clip.tokenize(text_labels).to(device)
    # print(query, "\n", text, "\n", image)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        distance = cosine_similarity(image_features, text_features)
        print(distance)  # image_features - text_features)

        # logits_per_image, logits_per_text = model(image, text)
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # print("Label probs:", logits_per_image)


'''
with torch.no_grad():
    image_features = model.encode_image(image)
    print(image_features.shape)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# prints: [[0.9927937  0.00421068 0.00299572]]
    print("Label probs:", probs, probs.argmax(), text_labels[probs.argmax()])

    plt.title(text_labels[probs.argmax()])
    img = mpimg.imread(img_path)
    imgplot = plt.imshow(img)

    plt.show()

exit()

'''
