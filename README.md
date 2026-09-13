# Chest-X-ray-Disease-Classification

### Overview

Built a **multi-label chest X-ray classification system** using **ResNet50 transfer learning** and TensorFlow/Keras on a patient-level subset of the NIH ChestX-ray14 dataset.

### Classification Scope

The model predicts the presence of **14 thoracic findings** independently
**No Finding** is also represented as an output when none of the 14 findings are present. Since this is a multi-label task, a single X-ray can contain multiple findings simultaneously.

### Data & Preprocessing

* Used a patient-level subset of approximately **20,000 chest X-rays**.
* Split patients into **70% training, 15% validation, and 15% test sets** to prevent patient-level data leakage.
* Resized images to **244 × 244** and applied ResNet50 preprocessing.
* Used Hugging Face streaming to load the dataset without downloading the full dataset locally.

### Model

* **ResNet50** pretrained on ImageNet used as a **frozen feature extractor**.
* Added Global Average Pooling followed by a **15-output sigmoid classification layer**.
* Used **weighted binary cross-entropy** to address severe class imbalance.

### Evaluation

* Evaluated using **multi-label ROC-AUC**.
* Achieved a **0.712 test AUC** on the selected test subset.
* Used validation AUC for model checkpointing and selected the best-performing model.

### Model Interpretability

Implemented **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize the image regions contributing to individual disease predictions. This provides class-specific heatmaps for interpreting the model's predictions.
