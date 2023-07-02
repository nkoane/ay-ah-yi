import random
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
import clip
from PIL import Image

from util import *


directory = "./data/images/"
img_path = directory + random.choice(os.listdir(directory))

device = get_device()
model, preprocess = clip.load("ViT-B/32", device=device)


image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
text_labels = ["a diagram", "a dog", "a cat"]
text = clip.tokenize(text_labels).to(device)

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
