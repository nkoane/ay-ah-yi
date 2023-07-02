
import datetime
from dotenv import load_dotenv
import os
import sys
import clip
import json
import vecs
import torch
from util import get_device
from PIL import Image

load_dotenv()

dir = '../data/images/'
files = os.listdir(dir)
device = get_device()

data_path = os.getenv("DATA_PATH")
model_name = os.getenv("MODEL_NAME")
db_connection = os.getenv("DB_CONNECTION")


def seed():

    model, preprocess = clip.load(model_name, device=device)
    vx = vecs.create_client(db_connection)
    try:
        images = vx.create_collection("images", dimension=512)
    except:
        images = vx.get_collection("images")

    # vx.delete_collection(name="image_vectors")
    # images = vx.create_collection(name="image_vectors", dimension=512)

    for i, file in enumerate(files):
        image_path = os.path.join(dir, file)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            vector = model.encode_image(image).squeeze().tolist()
        '''
        data.append({
            "file_id": file.split(".")[0],
            "file_name": file,
            "embeddings": vector
        })
        '''

        images.upsert(
            [
                # "file_id": file.split(".")[0],
                file,
                vector,
                {
                    "file_id": file.split(".")[0],
                    "file_name": file,
                    # "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            ]
        )

        print(f"done [{i+1}/{len(files)}], with {file}")

    # with open(data_path, "w") as f:
    #    json.dump(data, f)
    print(f"done, {len(files)} images seeded.")
    images.create_index()


def search():
    query = input("Enter query: ")
    if query == 'q':
        return
    model, _ = clip.load("ViT-L/14", device=device)
    # clip.tokenize(text_labels).to(device)
    query_tokens = clip.tokenize([query]).to(device)
    vector = model.encode_text(query_tokens)
    print(len(vector), len(query_tokens), query_tokens.shape, vector.shape)
    # exit()

    vx = vecs.create_client(db_connection)
    images = vx.get_collection("images")

    results = images.query(
        query_vector=vector,              # required
        limit=1,                            # number of records to return
        # filters={"type": {"$eq": "jpg"}},   # metadata filters
    )

    print(results)


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
