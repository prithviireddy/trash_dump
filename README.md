# ♻️ Garbage Classification System

> AI-powered waste segregation using transfer learning on EfficientNet-B3 — classifies garbage images into 10 categories to support automated recycling pipelines.

---

## 📌 Overview

This project trains a high-accuracy image classifier to categorize waste into 10 distinct classes using PyTorch and transfer learning. The model is built on top of a pretrained EfficientNet-B3 backbone and fine-tuned on the **Garbage Classification V2** dataset.

The pipeline covers everything end-to-end: data loading, class-imbalance handling, two-stage training, evaluation, and export of deployment-ready artifacts for frontend or backend integration.

---

## 🗂️ Dataset

| Property | Value |
|---|---|
| Source | Garbage Classification V2 (Kaggle) |
| Total Images | 12,259 |
| Number of Classes | 10 |
| Input Resolution | 224 × 224 (RGB) |

**Classes:** `battery` · `biological` · `cardboard` · `clothes` · `glass` · `metal` · `paper` · `plastic` · `shoes` · `trash`

---

## 🧠 Model Architecture

| Component | Detail |
|---|---|
| Backbone | EfficientNet-B3 (ImageNet pretrained) |
| Classifier Head | Linear(1536 → 10) |
| Training Strategy | Two-stage: frozen features → full fine-tune |
| Framework | PyTorch |

### Why EfficientNet-B3?
EfficientNet-B3 offers a strong accuracy-to-parameter ratio compared to alternatives like ResNet-50 or MobileNetV3. It achieves higher accuracy with fewer parameters, making it suitable for both cloud and edge deployment. The notebook also supports swapping in `efficientnet_b0`, `resnet50`, or `mobilenet_v3_large` via a single config change.

---

## ⚙️ Training Pipeline

### Two-Stage Strategy

**Stage 1 — Classifier warmup (feature extractor frozen)**
The backbone weights are frozen and only the new classification head is trained. This prevents destroying pretrained features early in training.

| Setting | Value |
|---|---|
| Epochs | 10 |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Early Stopping Patience | 4 |

**Stage 2 — Full fine-tuning (entire network unfrozen)**
All layers are unfrozen and trained end-to-end with a much smaller learning rate, allowing the backbone to adapt to the garbage domain.

| Setting | Value |
|---|---|
| Epochs | 6 |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-5 |
| Early Stopping Patience | 3 |

### Other Training Details

| Component | Detail |
|---|---|
| Optimizer | AdamW |
| LR Scheduler | ReduceLROnPlateau (factor=0.3, patience=2) |
| Loss Function | CrossEntropyLoss |
| Batch Size | 32 |
| Class Imbalance Handling | WeightedRandomSampler |
| Best Model Selection | Saved on lowest validation loss |

### Data Augmentation

| Split | Transforms Applied |
|---|---|
| Train | Resize → RandomHorizontalFlip → RandomRotation(15°) → ColorJitter → ToTensor → Normalize |
| Val / Test | Resize → ToTensor → Normalize |

Normalization uses ImageNet statistics: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

---

## 📁 Project Structure

```
├── dataset/
│   └── original/
│       ├── battery/
│       ├── biological/
│       ├── cardboard/
│       ├── clothes/
│       ├── glass/
│       ├── metal/
│       ├── paper/
│       ├── plastic/
│       ├── shoes/
│       └── trash/
├── artifacts/
│   ├── garbage_classifier_best.pt       ← PyTorch weights (state dict)
│   ├── garbage_classifier_scripted.pt   ← TorchScript export
│   ├── garbage_classifier.onnx          ← ONNX export (optional)
│   ├── class_names.json
│   ├── frontend_config.json
│   ├── metadata.json
│   └── sample_request.json
└── trash.ipynb
```

---

## 🚀 Quickstart

### 1. Install dependencies

```bash
pip install torch torchvision scikit-learn pandas matplotlib seaborn tqdm pillow
```

### 2. Run the notebook

Open `trash.ipynb` and run cells sequentially. The notebook auto-detects the dataset path from `candidate_paths`.

> **Windows users:** Set `NUM_WORKERS = 0` and remove `prefetch_factor` from all DataLoader calls to avoid multiprocessing deadlocks in Jupyter.

### 3. Run inference on a single image

```python
import torch
import json
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# Load artifacts
with open("artifacts/metadata.json") as f:
    meta = json.load(f)

class_names = meta["class_names"]
IMG_SIZE    = meta["img_size"]

# Rebuild model
model = models.efficientnet_b3(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))

ckpt = torch.load("artifacts/garbage_classifier_best.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Preprocess and predict
tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(meta["mean"], meta["std"]),
])

img    = Image.open("your_image.jpg").convert("RGB")
tensor = tfms(img).unsqueeze(0)

with torch.no_grad():
    probs = torch.softmax(model(tensor), dim=1).squeeze().numpy()

pred  = class_names[np.argmax(probs)]
conf  = float(np.max(probs))
print(f"Predicted: {pred}  |  Confidence: {conf*100:.1f}%")
```

---

## 📊 Evaluation

The notebook generates the following evaluation outputs on the held-out test set:

- Overall accuracy, precision, recall, F1-score (macro and weighted)
- Per-class breakdown sorted by F1-score
- Confusion matrix heatmap

---

## 📦 Deployment Artifacts

All artifacts are saved to the `artifacts/` folder after training.

| File | Purpose |
|---|---|
| `garbage_classifier_best.pt` | PyTorch state dict + metadata dict — use for reloading in Python |
| `garbage_classifier_scripted.pt` | TorchScript export — use for simple backend serving without Python ML dependencies |
| `garbage_classifier.onnx` | ONNX export — use for cross-framework / edge deployment |
| `class_names.json` | Ordered list of class labels matching model output indices |
| `frontend_config.json` | Input spec, output schema, normalization params — for frontend integration |
| `metadata.json` | Full training config: backbone, class names, img size, normalization, test accuracy |
| `sample_request.json` | Example API request payload |

### `frontend_config.json` schema

```json
{
  "input_type": "image_file",
  "accepted_extensions": ["jpg", "jpeg", "png"],
  "output": {
    "predicted_class": "string",
    "confidence": "float",
    "top_k": [{ "class_name": "string", "probability": "float" }]
  },
  "normalization": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "resize": [224, 224]
  }
}
```

---

## 🔬 Extending the Project

- **Swap backbone:** Change `BACKBONE = "efficientnet_b3"` to `"resnet50"`, `"efficientnet_b0"`, or `"mobilenet_v3_large"` — the rest of the pipeline adapts automatically.
- **Larger input size:** Try `IMG_SIZE = 256` or `300` if GPU memory allows — may improve accuracy on fine-grained classes like shoes vs. clothes.
- **Test-Time Augmentation (TTA):** Average predictions across multiple augmented views of each test image for a free accuracy boost.
- **Longer fine-tuning:** Replace `ReduceLROnPlateau` with a cosine annealing schedule and increase Stage 2 epochs.
- **Compare backbones side-by-side:** Run separate training runs with ResNet-50 vs EfficientNet-B3 and compare test F1 scores.
