# Pakistan-Currency-Note-Classifier-ML-Model
A complete end-to-end deep learning pipeline that classifies Pakistani currency notes (Rs. 10, 20, 50, 100, 500, 1000, 5000) from real-world photographs. The project covers the full ML lifecycle — dataset curation, preprocessing, augmentation, custom CNN design from scratch, and transfer learning with MobileNetV2 and EfficientNetB0. 

# PakCurrency-Classifier

> Multiclass deep learning classifier for all 7 Pakistani Rupee denominations — built with custom CNNs and transfer learning.

---

## Overview

This project implements a complete image classification pipeline to identify Pakistani currency note denominations (Rs. 10, 20, 50, 100, 500, 1000, and 5000) from real-world photographs captured across varied devices, lighting conditions, and angles.

Nine model variants were trained and benchmarked across three architectures:
- **Custom CNN (from scratch)** — 3 variants of increasing depth and complexity
- **MobileNetV2 (transfer learning)** — 3 fine-tuning configurations
- **EfficientNetB0 (transfer learning)** — 3 fine-tuning configurations

---

## Results

| Model         | Test Accuracy | Test Loss | Macro F1 |
|---------------|--------------|-----------|----------|
| CNN_A1_Small  | 90.51%       | 0.3171    | 0.9056   |
| CNN_A2_Medium | 98.48%       | 0.0415    | 0.9848   |
| **CNN_A3_Deep**   | **98.86%**   | **0.0312**| **0.9885** |
| MNV2_B1       | 96.46%       | 0.1162    | 0.9645   |
| MNV2_B2       | 97.59%       | 0.0882    | 0.9759   |
| MNV2_B3       | 95.57%       | 0.1128    | 0.9553   |
| **EffB0_C1**  | **99.11%**   | **0.0298**| **0.9913** |
| EffB0_C2      | 98.99%       | 0.0373    | 0.9898   |
| EffB0_C3      | 98.23%       | 0.0494    | 0.9820   |

> **Best overall:** `EffB0_C1` — EfficientNetB0 with top-20 layers unfrozen, fine-tuned at LR 1e-4.  
> **Best from scratch:** `CNN_A3_Deep` — 3 conv blocks [64→128→256], BatchNorm, LR 0.0005.

---

## Architecture Details

### Custom CNN (Section A)

Built entirely without pretrained weights using TensorFlow/Keras.

| Variant       | Filters       | BatchNorm | Dropout | LR     |
|---------------|---------------|-----------|---------|--------|
| CNN_A1_Small  | [32, 64]      | ✗         | 0.3     | 0.001  |
| CNN_A2_Medium | [32, 64, 128] | ✓         | 0.3     | 0.001  |
| CNN_A3_Deep   | [64, 128, 256]| ✓         | 0.3     | 0.0005 |

Each block: `Conv2D → (BatchNorm) → Conv2D → (BatchNorm) → MaxPool2D`  
Head: `GlobalAveragePooling2D → Dense(128, relu) → Dropout → Dense(7, softmax)`  
Optimizer: Adam | Loss: Categorical Crossentropy | Input: 224×224 RGB

### Transfer Learning (Sections B & C)

Two-phase training strategy:
1. **Phase 1 — Feature Extraction** (20 epochs): base model frozen, only the custom classification head trained at LR 1e-3.
2. **Phase 2 — Fine-tuning** (15 epochs): top N layers of base unfrozen at a reduced LR; BatchNorm layers kept frozen throughout.

| Variant   | Base Model    | Dense Units | Unfreeze Top N | Fine-tune LR |
|-----------|---------------|-------------|----------------|--------------|
| MNV2_B1   | MobileNetV2   | 128         | 20             | 1e-4         |
| MNV2_B2   | MobileNetV2   | 256         | 30             | 5e-5         |
| MNV2_B3   | MobileNetV2   | 128         | 50             | 1e-5         |
| EffB0_C1  | EfficientNetB0| 128         | 20             | 1e-4         |
| EffB0_C2  | EfficientNetB0| 256         | 30             | 5e-5         |
| EffB0_C3  | EfficientNetB0| 128         | 50             | 1e-5         |

---

## Dataset Pipeline

Raw images were collected from multiple mobile devices under varied conditions. The preprocessing pipeline includes:

- HEIC → JPEG format conversion (via `pillow-heif`)
- Perceptual hash-based duplicate removal (`imagehash.phash`)
- Corruption detection and safe removal
- Standardized resizing to **224×224 RGB**
- 70/15/15 train/val/test stratified split

**Training augmentation:** rotation (±30°), zoom (25%), width/height shift (15%), shear (0.2), brightness range [0.6–1.4], horizontal flip, `fill_mode='nearest'`.

---

## Requirements

```
tensorflow>=2.12
scikit-learn
pillow
pillow-heif
imagehash
pandas
numpy
matplotlib
seaborn
tqdm
```

Install all dependencies:
```bash
pip install tensorflow scikit-learn pillow pillow-heif imagehash pandas numpy matplotlib seaborn tqdm
```

---

## Key Findings

- A hand-crafted CNN (CNN_A3_Deep) reached **98.86%** accuracy — within 0.25% of the best pretrained model — demonstrating that careful architecture design and hyperparameter tuning can rival transfer learning on domain-specific tasks.
- Unfreezing too many base layers during fine-tuning (MNV2_B3, EffB0_C3) degraded performance, a characteristic sign of catastrophic forgetting on small datasets.
- EfficientNetB0's compound scaling (balancing width, depth, and resolution) made it the strongest transfer learning backbone for this task.

---

## Authors

| Name                  | Roll No  |
|-----------------------|----------|
| Muhammad Anas Yazdani | CS23102  |
| Mashaal Ali           | CS23091  |
| Osama Humayun         | CS23139  |

CS-324 Machine Learning — CEP Project  
Department of Computer and Information Systems, NEDUET.

---

## License

This project is licensed under the MIT License.
