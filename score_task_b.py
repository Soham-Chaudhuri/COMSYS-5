import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from collections import Counter
from deepface import DeepFace
from sklearn.metrics import classification_report
import chromadb
from chromadb.config import Settings

def l2_normalize(x):
    return x / np.linalg.norm(x)

def load_json_collection(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def recognize_person(query_img, collection):
    query_embedding = DeepFace.represent(
        img_path=query_img,
        model_name='ArcFace',
        detector_backend='skip',
        enforce_detection=False
    )[0]["embedding"]
    query_embedding = l2_normalize(np.array(query_embedding))

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5
    )

    top_docs = results['documents'][0]
    label_counts = Counter(top_docs)
    best_match = label_counts.most_common(1)[0][0]
    return best_match

def build_collection_from_json(client, name, json_data, batch_size=5000):
    collection = client.get_or_create_collection(name=name)
    total = len(json_data['ids'])

    for i in range(0, total, batch_size):
        collection.add(
            ids=json_data['ids'][i:i+batch_size],
            embeddings=json_data['embeddings'][i:i+batch_size],
            documents=json_data.get('documents', [])[i:i+batch_size],
            metadatas=json_data.get('metadatas', [])[i:i+batch_size],
        )
    return collection


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python score_task_b.py <val_folder_path> <json_embedding_file>")
        sys.exit(1)

    val_path = sys.argv[1]
    json_path = sys.argv[2]

    # Load collection from JSON
    chroma_client = chromadb.Client()
    json_data = load_json_collection(json_path)
    collection = build_collection_from_json(chroma_client, "faces", json_data)

    # Prepare test images
    y_true, y_pred = [], []

    for person_folder in os.listdir(val_path):
        person_path = os.path.join(val_path, person_folder)
        if not os.path.isdir(person_path):
            continue

    # Check main images and any in distortion/ subfolder
    image_paths = []

    # Add direct images
    for fname in os.listdir(person_path):
        fpath = os.path.join(person_path, fname)
        if os.path.isfile(fpath):
            image_paths.append(fpath)

    # Add distorted images if distortion/ subfolder exists
    distortion_path = os.path.join(person_path, "distortion")
    if os.path.exists(distortion_path):
        for fname in os.listdir(distortion_path):
            fpath = os.path.join(distortion_path, fname)
            if os.path.isfile(fpath):
                image_paths.append(fpath)

    # Process all collected image paths
    for img_path in image_paths:
        try:
            predicted_label = recognize_person(img_path, collection)
            is_match = 1 if predicted_label == person_folder else 0
            y_true.append(1)
            y_pred.append(is_match)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue


    print("\n📊 Face Recognition (Binary) Report:")
    print(classification_report(y_true, y_pred, digits=4))
