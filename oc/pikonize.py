from PIL import Image
# from IPython.display import Image
import open_clip
import torch
import os
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# device = get_device()

if torch.backends.mps.is_available():
    device = torch.device("mps:0")
# data = data.to(mps_device)

model, _, transform = open_clip.create_model_and_transforms(
    model_name="coca_ViT-B-32",
    # model_name="coca_ViT-L-14",
    pretrained="mscoco_finetuned_laion2B-s13B-b90k",
    # device=device
)

figureWindow = plt.figure("Pikonize")
while True:
    query = input("Enter a query: ")
    if query == "q":
        break
    for _ in range(1):
        plt.close()
        directory = '../data/images/'
        img_path = directory + random.choice(os.listdir(directory))
        print("\n-> ", img_path)

        # plt.title(text_labels[probs.argmax()])
        img = mpimg.imread(img_path)
        # figureWindow
        imgplot = plt.imshow(img)

        plt.show(block=False)

        im = Image.open(img_path).convert("RGB")
        im = transform(im).unsqueeze(0)

        # with torch.no_grad(), torch.cuda.amp.autocast():
        with torch.no_grad(), torch.cuda.amp.autocast():
            generated = model.generate(im)  # .to(device=device)

        print("-> ", open_clip.decode(generated[0]))

# plt.title(text_labels[probs.argmax()])
# img = mpimg.imread(img_path)
# imgplot = plt.imshow(img)

# plt.show()
# slow_conv2d_forward_mps: input(device='cpu') and weight(device=mps:0')  must be on the same device
