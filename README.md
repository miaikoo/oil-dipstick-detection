# YOLOv11 Dipstick Segmentation + CatBoost Oil Color Classification

## 📋 Project Overview

Two-stage pipeline for automatic dipstick analysis:

1. **YOLOv11 Segmentation** - Detects dipstick, oil region, and min-max mark
2. **CatBoost Classifier** - Classifies oil condition (Clean or Brown)

---

## 🔄 Workflow

```
Raw Image
  ↓
[YOLOv11 Segmentation] → Dipstick + Oil + Min-Mark regions
  ↓                      ↓
  ├── Extract Features (H, S, V, texture, contrast)
  │
[CatBoost Classifier] → Clean or Brown
  ↓
Oil Condition Decision
```

---

## 🎯 Oil Color Classification

Classifies engine oil condition into two categories:

- **Clean** (Clear/Yellow) → ✓ Good condition, no action needed
- **Brown** (Dark/Dirty) → ⚠️ Oil change recommended

### Getting Started

**1. Install Dependencies**

```bash
pip install torch torchvision opencv-python matplotlib seaborn pandas scikit-learn ultralytics catboost
```

**2. Prepare Dataset** (if training custom model)

- Annotated images with dipstick, oil, and min-mark segmentation masks
- Expected: YOLO segmentation format (`.txt` label files with polygon points)
- Ground truth labels: dipstick images labeled as "clean" or "brown"
- Place in: `data/color_classification/[train|val|test]/`

**3. Train Models**

- Open: `script/ColorTraining.ipynb`
- Cell 1-5: Train YOLOv11 segmentation model for regions
- Cell 6-10: Extract features and train CatBoost classifier
- Review results and confusion matrices

**4. Get Models**

- YOLOv11 model: `models/p-91.pt` (segmentation)
- CatBoost model: `models/oil_quality_catboost_v3.cbm` (classification)

---

## 📁 Project Structure

```
Dipstick-detection/
│
├── 📂 data/                              (⚠️ Not in repo - confidential)
│   ├── _background/
│   ├── _cvat-annot/
│   ├── filtered-split/
│   ├── oil-stick-detection/
│   └── color_classification/             (Images labeled: clean or brown)
│       ├── train/, val/, test/
│       │   ├── images/
│       │   └── labels/
│
├── 📂 script/
│   ├── ColorTraining.ipynb               (YOLOv11 + CatBoost pipeline) ⭐
│   ├── ConfusionMatrix.ipynb
│   ├── DipstickDetection.ipynb           (Legacy)
│   └── Parsing.ipynb
│
├── 📂 models/                            (⚠️ Not in repo)
│   ├── p-91.pt                           (YOLOv11 segmentation)
│   ├── whole-dipstick.pt                 (Full dipstick segmentation)
│   └── oil_quality_catboost_v3.cbm       (CatBoost classifier)
│
└── README.md
```

**Note**:

- 🔴 `data/` is **not pushed** (confidential)
- ✓ `models/` and `script/` are pushed
- 📦 Deployment is in separate repository (not included here)

---

## Deployment
This model is deployed as a Hugging Face Space:
- Space: https://huggingface.co/spaces/miaikoo/oil-quality
- Framework: Gradio

---

## 🚀 Key Features

### YOLOv11 Segmentation

- **Architecture**: YOLOv11 instance segmentation
- **Input**: RGB images (arbitrary size)
- **Output**: Three segmentation masks: dipstick, oil region, min-mark
- **Speed**: Real-time inference (GPU)
- **Training**: ~30-50 min (GPU), ~3-5 hours (CPU)

### CatBoost Color Classifier

- **Architecture**: Gradient Boosting Classification
- **Input**: Color & texture features extracted from oil region (H, S, V, contrast, saturation)
- **Output**: Binary classification (Clean or Brown) + confidence score
- **Speed**: Instant inference
- **Training**: ~5-10 min (GPU/CPU)

### Training Features

- Real-time YOLOv11 metrics (mAP, precision, recall)
- Feature extraction with erosion and IQR-based statistics
- Class imbalance handling via weighted CatBoost
- Threshold tuning for optimal F1 score
- Confusion matrix and classification reports
- Visual comparison of predictions vs ground truth

### Output & Evaluation

- YOLOv11: Segmentation masks and training plots
- CatBoost: Classification metrics (precision, recall, F1, confusion matrix)
- Feature importance ranking
- Visual analysis comparing predictions with ground truth
- Performance benchmark plots

---

## 📊 Performance Metrics

**YOLOv11 Segmentation**:

- **mAP50**: Mean Average Precision at IoU 0.5
- **mAP75**: Mean Average Precision at IoU 0.75
- **Precision & Recall**: Per-class detection metrics

**CatBoost Classification**:

- **Accuracy**: Overall correctness rate
- **Precision**: Percentage correct for "Brown" predictions
- **Recall**: Percentage of actual "Brown" cases found
- **F1-Score**: Balanced precision-recall metric
- **Confusion Matrix**: True positives, false positives, false negatives, true negatives

Example results:

```
Segmentation (YOLOv11):
  mAP50: 0.85
  mAP75: 0.72
  Speed: 45ms per image

Classification (CatBoost):
  Accuracy:  0.92
  Precision: 0.89
  Recall:    0.88
  F1-Score:  0.88
```

---

## ⚙️ Technical Details

### Stage 1: YOLOv11 Segmentation

```
Input Image (RGB, any size)
  ↓
YOLOv11 Encoder (CSPDarknet backbone)
  ├─ Feature pyramid network
  └─ Multi-scale feature maps
  ↓
Segmentation Head (3 classes):
  ├─ Output 1: Dipstick mask
  ├─ Output 2: Oil region mask
  └─ Output 3: Min-mark mask
  ↓
Polygon points (normalized coordinates)
```

### Stage 2: Feature Extraction

```
For detected Oil Region:
  1. Extract RGB pixels from segmentation mask
  2. Apply erosion (1 iteration) to ignore edges
  3. Convert to HSV color space

  Features extracted:
  ├─ H (Hue): Color tone
  ├─ S (Saturation): Color intensity
  ├─ V (Value): Brightness
  ├─ IQR_S (Saturation texture): Robust variation
  ├─ IQR_V (Brightness texture): Robust variation
  ├─ brown_index: S / (V + offset) → Differentiates golden vs brown
  ├─ rel_darkness: V_oil / (V_stick + offset) → Relative to reference
  ├─ sat_boost: S_oil / (S_stick + offset) → Color addition
  └─ diff_V: V_stick - V_oil → Contrast
```

### Stage 3: CatBoost Classification

```
Input: 9 features (from Stage 2)
  ↓
CatBoost Classifier
  ├─ 1500 trees (iterations)
  ├─ Learning rate: 0.02
  ├─ Tree depth: 5
  ├─ L2 regularization: 7
  └─ Class weights: Balanced (handles imbalance)
  ↓
Output: P(Brown) probability
  ↓
Apply threshold (default: 0.5)
  ↓
Classification: Clean or Brown
```

### Training Configuration

**YOLOv11**:

```python
Epochs:        50
Batch Size:    32
Image Size:    1024×1024
Early Stop:    Patience=10
Optimizer:     AdamW
Learning Rate: 1e-4 (with cosine annealing)
Augmentation:  HSV, Rotation, Scale, Flip, CopyPaste
```

**CatBoost**:

```python
Iterations:         1500
Learning Rate:      0.02
Tree Depth:         5
L2 Regularization:  7
Loss Function:      Logloss (binary classification)
Eval Metric:        F1
Early Stopping:     100 rounds
Class Weights:      Balanced
```

---

## 📚 Main Notebook

- **script/ColorTraining.ipynb** - YOLOv11 + CatBoost complete pipeline ⭐

---

## 🔄 Current Status

✅ **Available**:

- YOLOv11 segmentation models (oil region, dipstick, min-mark detection)
- CatBoost color classifier (Clean vs Brown)
- Complete training pipeline in one notebook
- Feature extraction and visualization tools

📋 **To Train Models**:

1. Prepare annotated images with dipstick/oil/min-mark masks
2. Label images as "clean" or "brown" in ground truth
3. Place data in `data/color_classification/[train|val|test]/`
4. Open `script/ColorTraining.ipynb` and run all cells
5. Models saved to `models/` directory:
   - `p-91.pt` → YOLOv11 segmentation
   - `oil_quality_catboost_v3.cbm` → CatBoost classifier

---

## ⚠️ Important Notes

1. **Data Confidentiality**: Training data is not included (confidential). Prepare your own annotated dataset.
2. **Pre-trained Models Included**: `models/` contains p-91.pt and oil_quality_catboost_v3.cbm for inference
3. **Mask Quality Matters**: Accurate segmentation annotations lead to better feature extraction → better classification
4. **GPU Recommended**: YOLOv11 training: 30-50 min (GPU) vs 3-5 hours (CPU)
5. **Balanced Dataset**: Similar number of clean and brown samples for best results
6. **Feature Engineering**: HSV-based features are robust to glare and lighting variations
7. **Deployment Separate**: Web deployment code is in a separate repository

---

## 💻 System Requirements

**Minimum**:

- Python 3.8+
- 8GB RAM
- 5GB disk space (data + model)

**Recommended**:

- GPU (NVIDIA RTX 3060 or better)
- 16GB+ RAM
- SSD storage

**PyTorch Requirements**:

```bash
# NVIDIA GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision

# AMD GPU (ROCm)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

---

## 🎓 Learning Resources

- [YOLOv11 Documentation](https://docs.ultralytics.com/models/yolov11/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [OpenCV Color Spaces](https://docs.opencv.org/master/df/d9d/tutorial_py_colorspaces.html)
- [Scikit-learn Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

## 📞 Troubleshooting

**Common Issues**:

- ❌ "No images found" → Verify folder structure: `data/color_classification/[train|val|test]/images/`
- ❌ "Low classification accuracy" → Check image annotation quality and label consistency
- ❌ "GPU memory error" → Reduce YOLOv11 batch size (default: 32) in notebook
- ❌ "Poor segmentation masks" → Review training data annotations for accuracy
- ❌ "Module not found" → Run pip install from Getting Started section
- ❌ "CatBoost threshold issues" → Adjust threshold after training based on F1 score curve

---

## 📝 Version History

**v3.0 (Current)** - YOLOv11 + CatBoost Pipeline

- YOLOv11 instance segmentation (dipstick, oil, min-mark)
- CatBoost binary classification (Clean vs Brown)
- HSV feature extraction with erosion and IQR
- Threshold tuning for optimal F1 score
- Complete training and evaluation pipeline
- Visual analysis toolkit

**v2.0** - Oil Color Classification (Legacy)

- ASTM grade classification
- ResNet50 architecture
- Deprecated

**v1.0** - Dipstick Detection

- YOLOv11 dipstick detection
- Instance segmentation

---

## 📄 License & Attribution

This project uses:

- **PyTorch** - Deep learning framework
- **YOLOv11** - Instance segmentation model
- **CatBoost** - Gradient boosting classifier
- **OpenCV** - Image processing
- **Scikit-learn** - Metrics and evaluation

---

## 🙏 Acknowledgments

- Ultralytics for YOLOv11 documentation
- CatBoost team for excellent gradient boosting library
- OpenCV community for image processing tools

---

**Last Updated**: February 2026  
**Main Notebook**: `script/ColorTraining.ipynb`
