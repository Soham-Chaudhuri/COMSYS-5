# 🧪 Face Analysis Evaluation Scripts

This repository provides scoring scripts for:

- ✅ **Task A: Gender Classification**
- ✅ **Task B: Face Recognition (Binary Verification)**

---

## 🔧 Setup

Install required libraries:

```bash
pip install -r requirements.txt
```

Required config files:

[Download Face Embeddings from Google Drive](https://drive.google.com/file/d/1ObtkkYKOJGIJucQQTOQF7R_G9tOU7NLz/view?usp=sharing)

[Download Gender model weights from Google Drive](https://drive.google.com/file/d/10WenJR1PqJGp_stcZkEERY7AxE1v231f/view?usp=sharing)

[Download ArcFace model weights from Google Drive](https://drive.google.com/file/d/1p7LM_NhbGcf6eff-2pdmuZn83nIx7XSs/view?usp=sharing)



## 🧠 Task A – Gender Classification

### 📁 Validation Folder Structure
```

val_task_a/
├── male/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── female/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...

```

```bash
python score_task_a.py --val_path ./val_task_a --weights_path ./saved_model/model_weights.pt
```

## 🧠 Task B – Face Recognition

### 📁 Validation Folder Structure

```
val_task_b/
├── Person_A/
│   ├── img.jpg
│   └── distortion/
│       └── distorted_1.jpg
├── Person_B/
│   ├── ...
```

```bash
python score_task_b.py --val_path ./val_task_b --collection faces_collection.json
```


