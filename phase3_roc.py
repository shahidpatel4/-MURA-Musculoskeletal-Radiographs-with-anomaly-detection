import os
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import csv

# base directory containing abnormal image subdirectories #
# (healthy_test abnormal) produced in phase 2 #

BASE_PATH = "/Users/shahidpatel/Documents/results/anomaly_images"

def get_scores(folder):
    scores = []
    for file in os.listdir(folder):

        # match only unscaled subtraction reconstruction error image #
        if file.endswith("subtraction_reconstruction_error.png") and "scaled" not in file:
            img = Image.open(os.path.join(folder, file)).convert("L")
            arr = np.array(img)
    
            #sum all pixel values as the anomaly score for the image #
            score = np.sum(arr)
            scores.append(score)
    return scores

# healthy test samples are labelled 0 the negative class #
healthy_scores = get_scores(os.path.join(BASE_PATH, "healthy_test"))

# abnormal/pathological samples are labelled 1 positive class #
abnormal_scores = get_scores(os.path.join(BASE_PATH, "abnormal"))

# build ground truth labels and combine score list for ROC analysis #
y_true = [0]*len(healthy_scores) + [1]*len(abnormal_scores)
y_scores = healthy_scores + abnormal_scores

# export all the scores with their labels for further inspection or analysis #
with open("phase3_scores.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["score", "label"])
    for s in healthy_scores:
        writer.writerow([s, 0])
    for s in abnormal_scores:
        writer.writerow([s, 1])

# compute false positive rate, ture positive rate and AUC #
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# plot and sav ethe ROC curve #
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("phase3_roc_curve.png")
plt.show()

print("AUC:", roc_auc)