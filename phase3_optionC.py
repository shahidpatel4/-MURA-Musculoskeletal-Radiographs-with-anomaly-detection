import os
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score

# base directory where containing anomaly image sub directories #
BASE_PATH = "/Users/shahidpatel/Documents/results/anomaly_images"

def get_scores(folder):
    scores = []
    for file in os.listdir(folder):

        # Only process scaled substraction reconstruction error image #
        if "scaled_subtraction_reconstruction_error.png" in file:
            img = Image.open(os.path.join(folder, file)).convert("L")
            arr = np.array(img)

            # count the pixel exceeding intensity 50 as anomlous pixels #
            score = np.sum(arr > 50)
            scores.append(score)
    return scores

# healthy test samples ground truth label 0 #
healthy = get_scores(os.path.join(BASE_PATH, "healthy_test"))

# abnormal samples ground truth label 1 #
abnormal = get_scores(os.path.join(BASE_PATH, "abnormal"))

# compute the means score across all samples as a reference threshold #
computed_threshold = np.mean(healthy + abnormal)

# fixed threshold chosen based on the computed mean for classification #
threshold = 2298.40

# build ground truth labels 0 for healthy and 1 for abnormal #
y_true = [0] * len(healthy) + [1] * len(abnormal)

# apply threshold rule: score greater than or equal to threshold normal is 1 and healthy is 0 #
y_pred = [0 if s < threshold else 1 for s in (healthy + abnormal)]

# compute overall classification accuracy #
accuracy = accuracy_score(y_true, y_pred)

print("Computed Threshold:", computed_threshold)
print("Used Threshold:", threshold)
print("Accuracy:", accuracy)