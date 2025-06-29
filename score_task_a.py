import os
import cv2
from tqdm import tqdm
from deepface import DeepFace
from sklearn.metrics import classification_report

def load_data(data_dir):
    data = {'male': [], 'female': []}
    for gender in ['male', 'female']:
        gender_dir = os.path.join(data_dir, gender)
        for fname in os.listdir(gender_dir):
            data[gender].append(os.path.join(gender_dir, fname))
    return data

def preprocess_faces(data, target_size=(112, 112)):
    def lighting_normalization(img):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def denoise(img):
        return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

    processed = {}
    for person in data:
        images = data[person]
        processed_images = []
        for img_path in tqdm(images, desc=f"Preprocessing {person}"):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = denoise(img)
            img = lighting_normalization(img)
            img = cv2.resize(img, target_size)
            processed_images.append(img)
        processed[person] = processed_images
    return processed

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python score_task_a.py <val_folder_path>")
        sys.exit(1)

    val_path = sys.argv[1]
    data = load_data(val_path)

    y_true, y_pred = [], []

    for label in ['female', 'male']:
        for img_path in tqdm(data[label], desc=f"Predicting {label}"):
            try:
                gender_pred = DeepFace.analyze(img_path=img_path, actions=["gender"], detector_backend='skip')[0]['gender']
                predicted = 'female' if gender_pred['Woman'] > gender_pred['Man'] else 'male'
                y_true.append(label)
                y_pred.append(predicted)
            except Exception as e:
                print(f"Error with {img_path}: {e}")
                continue

    print("\n📊 Gender Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
