#  Anomaly Detection in Musculoskeletal X-rays

##  Overview
This project implements an autoencoder-based anomaly detection system on the MURA dataset to identify abnormalities in X-ray images using reconstruction error analysis.

## Methodology
- Autoencoder model for image reconstruction
- Reconstruction error used as anomaly score
- Pixel intensity differences analyzed
- ROC and AUC used for evaluation

## Results
- AUC Score: ~0.496 (near-random)
- Accuracy (Rule-based): ~29.24%
- Poor separation between healthy and abnormal samples

##  Visualization
- Original vs Reconstructed images
- Reconstruction Error maps
- Scaled subtraction error highlights anomalies

## Project Structure
phase3/
│── phase3_roc.py
│── phase3_optionC.py
│── phase3_montage.py
│── phase3_scores.csv
│── montage_healthy_test.png
│── montage_abnormal.png
│── phase3_roc_curve.png

##  How to Run
python3 phase3_roc.py
python3 phase3_optionC.py
python3 phase3_montage.py

## Conclusion
The model shows low performance, indicating need for improved methods and deeper models.

