# Multiclass Skin Lesion Classification: A Leakage-Free, Explainable Baseline

Research project for internship applications. Classifies dermoscopic images into 7 diagnostic classes (HAM10000): akiec, bcc, bkl, df, mel, nv, vasc.

## Research questions
1. How much does **lesion-level splitting** (vs. naive image-level splitting) change reported performance? (Many papers inflate results through leakage.)
2. Which **imbalance strategy** (none / weighted loss / weighted sampler) best improves minority-class recall, especially melanoma?
3. Do **Grad-CAM** maps focus on the lesion, or on artifacts (rulers, skin markers, vignettes)?

## Setup
```bash
pip install -r requirements.txt
# Download HAM10000 (Harvard Dataverse / Kaggle) and unzip both image parts + HAM10000_metadata.csv into ./HAM10000
python skin_classifier.py --data_dir ./HAM10000 --backbone efficientnet_b0 --imbalance loss --epochs 20
```
Outputs per run in `runs/<backbone>_<imbalance>_s<seed>/`: `results.json`, `confusion_matrix.png`, `gradcam_samples.png`, `best.pt`.

## Experiment plan (run these, then fill the table)
| Backbone | Imbalance | Seeds | Bal. Acc | Macro-F1 | Macro-AUROC | Melanoma recall |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | none | 3 | | | | |
| EfficientNet-B0 | loss | 3 | | | | |
| EfficientNet-B0 | sampler | 3 | | | | |
| ResNet-50 | loss | 3 | | | | |

Run seeds 0, 1, 2 and report **mean ± std**. Optional extras: naive-split ablation (to quantify leakage), external validation on ISIC 2019 or PH2, calibration (ECE), and mask-out-artifact Grad-CAM analysis.

## Limitations (state these honestly in your application)
- HAM10000 is skewed toward lighter skin tones; results may not generalize across skin types.
- Research prototype only, **not a diagnostic tool**.
- Single-dataset results; external validation is future work.

## How to use this for your application
- Put the results table, one confusion matrix, and one Grad-CAM figure in a 1-page summary.
- Link a clean GitHub repo and mention your specific research question (e.g. "quantifying data leakage") in your cover letter.
- Tailor it to the lab: if they work on fairness, add skin-tone analysis; on explainability, deepen the Grad-CAM study.
- Only claim numbers you actually produced by running the code.
