import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel

# ------------- Face Dataset -----------------
class FaceDataset(Dataset):
    def __init__(self, root_dir, processor, transform=None):
        self.processor = processor
        self.transform = transform
        self.data = []

        for label, class_name in enumerate(['male', 'female']):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append((os.path.join(class_dir, img_name), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, torch.tensor(label)

# ------------- Model -----------------
class GenderClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feature_dim = self.clip.get_image_features(pixel_values=dummy).shape[1]
        self.classifier = torch.nn.Linear(feature_dim, 2)

    def forward(self, pixel_values):
        features = self.clip.get_image_features(pixel_values=pixel_values)
        return self.classifier(features)

# ------------- Main Evaluation Logic -----------------
def evaluate(val_dir, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    transform = T.Compose([T.Resize((224, 224))])
    val_dataset = FaceDataset(val_dir, processor, transform)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Instantiate the model and load weights
    model = GenderClassifier().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for pixel_values, labels in tqdm(val_loader, desc="Evaluating"):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            logits = model(pixel_values)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Male", "Female"]))

# ------------- Script Entry Point -----------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python score_task_a.py <val_dir> <model_path>")
        sys.exit(1)

    val_dir = sys.argv[1]
    model_path = sys.argv[2]
    evaluate(val_dir, model_path)
