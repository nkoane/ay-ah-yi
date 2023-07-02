# BEGIN: abpxx6d04wxr

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from .env file


def get_device():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")

    else:
        device = "mps"

    return device


# BEGIN: 8d9c7f5d7f1c
device = torch.device(get_device())
# get_device()
print(f"Using device: {device}")

model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name_or_path=model_name, device=device)

# Our sentences we like to encode
# BEGIN: 1b2c3d4e5f6g

areWeTestingTheFile = True
whichFileAreWeTesting = '../data/docs/arundathi-roy/come-september.txt'

if (areWeTestingTheFile):
    with open(whichFileAreWeTesting, 'r') as f:
        sentences = [line.strip() for line in f if line.strip()]
    #    sentences = f.readlines()

else:
    sentences = ['This framework generates embeddings for each input sentence',
                 'Sentences are passed as a list of string.',
                 'The quick brown fox jumps over the lazy dog.']

# Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

while True:
    # Take a user input
    query = input("Enter a query: ")
    # If the user enters an empty string, exit the loop
    if not query:
        break
    # Encode the query
    query_embedding = model.encode([query])[0]

    # Calculate the cosine similarity between the query embedding and the sentence embeddings
    similarities = cosine_similarity([query_embedding], embeddings)

    # Print the most similar sentence
    most_similar_index = similarities.argmax()
    most_similar_sentence = sentences[most_similar_index]
    most_similar_similarity = similarities[0][most_similar_index]

    print(f"ArgMax? -> {similarities.argmax()}")
    print(f"Most similar sentence: {most_similar_sentence}")
    print(f"Similarity score: {most_similar_similarity}")

    print(f"\n-------------------------\n\nMax -> {similarities.max()}")

    counter = 0
    for i in sorted(range(len(similarities[0])), key=lambda k: similarities[0][k], reverse=True):
        # print(f"Similarity score [{i}]: {similarities[0][i]}")
        if counter < 5:
            print(
                f"->Most similar sentence [{i} : {similarities[0][i]}]\n{sentences[i]}")
            counter += 1
        else:
            break
    # print(similarities[0].__len__())
    break
    # print(f"Most similar sentence: {sentences[similarities.max()]}")
    # print(f"Similarity score: {similarities.max()}")
    # END: abpxx6d04wxr
