# ESC-50 audio classification assignment

This folder contains a complete solution for the four-class ESC-50 assignment.

## Run

From the ESC-50 project root:

```powershell
python -m pip install -r assignment\requirements.txt
python assignment\esc50_audio_classification.py
```

The first run extracts MFCC features for 160 audio files and may take a few
minutes. Later runs reuse `assignment/results/mfcc_features.npz`.

Generated outputs:

- `report_zh.md`: Chinese experiment report with actual measured results
- `summary.json`: main numerical results
- `data_split.csv`: reproducible 70/30 train/test assignment
- `pca_3d.png`: PCA visualization
- `knn_k_performance.png` and `knn_k_scores.csv`: k selection results
- `confusion_matrices.png`: kNN and SVM confusion matrices
- `classification_report.csv`: precision, recall, and F1-score

To choose another four ESC-50 categories:

```powershell
python assignment\esc50_audio_classification.py --classes dog rain siren clock_tick
```
