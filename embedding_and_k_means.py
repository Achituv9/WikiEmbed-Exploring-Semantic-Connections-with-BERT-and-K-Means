import glob
import json
import os
import re
import shutil
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from kmeans_pytorch import kmeans
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

##############################################################################
# PART 1: BERT EMBEDDING EXTRACTION
##############################################################################

# Directory containing JSON files
INPUT_DIR = "filtered_wikipedia_data"

# Directory to store partial .npz embeddings
NPZ_DIR = "npz files"

# Final combined file
COMBINED_NPZ_PATH = "Bert_wikipidia_data.npz"


# Create npz folder if it doesn't exist
if not os.path.exists(NPZ_DIR):
    os.makedirs(NPZ_DIR)

print("Loading BERT tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Put model in evaluation mode


def clean_text(text):
    """Remove HTML tags, URLs, special characters/numbers, etc."""
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^A-Za-z0-9 ]+", "", text)  # Remove non-alphanumeric (except spaces)
    text = text.lower().strip()  # Lowercase & trim
    return text


def get_bert_embedding(text):
    """Return the [CLS] embedding from BERT for the given text."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Use the [CLS] token embedding (index 0) as the sentence-level embedding
    sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return sentence_embedding


##############################################################################
# PART 1 (continued): PROCESS ARTICLES
##############################################################################

start_time = time.time()
total_articles = 0
embeddings = []
ids = []

print("Starting to process JSON files and extract embeddings...")

for filename in os.listdir(INPUT_DIR):
    file_path = os.path.join(INPUT_DIR, filename)
    if not file_path.endswith(".json"):
        continue

    print(f"Processing file: {filename}")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

            for article in data:
                clean_article_text = clean_text(article["text"])
                embedding = get_bert_embedding(clean_article_text)

                embeddings.append(embedding)
                ids.append(article["id"])
                total_articles += 1

                # Every 25k articles, save a partial .npz file
                if total_articles % 25000 == 0:
                    embeddings = np.array(embeddings)
                    print(
                        f"Processed {total_articles} articles so far... "
                        f"Saving partial NPZ..."
                    )

                    now = datetime.now()
                    formatted_datetime = now.strftime("%d%m%Y%H%M%S")

                    npz_path = os.path.join(
                        NPZ_DIR, f"embeddings_{formatted_datetime}.npz"
                    )
                    np.savez(npz_path, embeddings=embeddings, ids=ids)

                    print(f"Partial embeddings saved to {npz_path}")
                    # Reset in-memory lists
                    ids = []
                    embeddings = []
    except json.JSONDecodeError:
        print(f"JSON decoding error in file: {filename}")
    except Exception as e:
        print(f"Error processing file {filename}: {str(e)}")

# If there are remaining embeddings that haven't been saved yet, save them now
if len(embeddings) > 0:
    embeddings = np.array(embeddings)
    now = datetime.now()
    formatted_datetime = now.strftime("%d%m%Y%H%M%S")
    npz_path = os.path.join(NPZ_DIR, f"embeddings_{formatted_datetime}.npz")
    np.savez(npz_path, embeddings=embeddings, ids=ids)
    print(f"Final batch saved to {npz_path}")

print(f"Total number of articles processed: {total_articles}")
total_elapsed_time = time.time() - start_time
print(f"Total elapsed time: {timedelta(seconds=int(total_elapsed_time))}")

##############################################################################
# PART 2: COMBINE ALL PARTIAL NPZ INTO A SINGLE FILE & REMOVE NPZ DIR
##############################################################################

print("\nCombining all partial .npz files into one big NPZ...")

all_embeddings = []
all_ids = []

for npz_file in glob.glob(os.path.join(NPZ_DIR, "*.npz")):
    data = np.load(npz_file)
    batch_embeddings = data["embeddings"]
    batch_ids = data["ids"]

    all_embeddings.append(batch_embeddings)
    all_ids.extend(batch_ids)

all_embeddings = np.concatenate(all_embeddings, axis=0)
all_ids = np.array(all_ids)

np.savez(COMBINED_NPZ_PATH, embeddings=all_embeddings, ids=all_ids)
print(f"Combined embeddings saved to {COMBINED_NPZ_PATH}")
print(f"Shape of combined embeddings: {all_embeddings.shape}")
print(f"Number of IDs: {all_ids.shape}")

print("Removing 'npz files' folder...")
shutil.rmtree(NPZ_DIR, ignore_errors=True)
print("'npz files' folder removed.")

##############################################################################
# PART 3: KMEANS CLUSTERING WITH k=1..25 AND CSV OUTPUT
##############################################################################

print("\nRunning k-means clustering on the combined embeddings...")

CSV_CLUSTER_DIR = "k_means_cluster"
if not os.path.exists(CSV_CLUSTER_DIR):
    os.makedirs(CSV_CLUSTER_DIR)

tensor_embeddings = torch.from_numpy(all_embeddings).float().to(device)

for i in range(1, 26):
    kmeans_start = time.time()

    cluster_ids_x, cluster_centers = kmeans(
        X=tensor_embeddings,
        num_clusters=i,
        distance="euclidean",
        device=device,
    )

    cluster_labels = cluster_ids_x.cpu().numpy()

    cluster_df = pd.DataFrame({"ID": all_ids, "Cluster": cluster_labels})

    csv_filename = f"id_cluster_mapping_{i}_k_means_cluster.csv"
    csv_path = os.path.join(CSV_CLUSTER_DIR, csv_filename)
    cluster_df.to_csv(csv_path, index=False)

    print(f"CSV file '{csv_filename}' created with ID-to-cluster mapping.")
    kmeans_end = time.time()
    print(f"Total time for {i} clusters: {kmeans_end - kmeans_start:.2f} seconds")

print("All KMeans clusterings (k=1..25) are done.")
