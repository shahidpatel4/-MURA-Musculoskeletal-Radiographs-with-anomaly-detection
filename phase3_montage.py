import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# directory contains the anomaly images subdirectories #
BASE_PATH = "/Users/shahidpatel/Documents/results/anomaly_images"

def create_montage(folder, output_name, title):
    # create and save a 5x4 montage image with clearly labelled rows and columns #

    files = [f for f in os.listdir(folder) if "original.png" in f]

    # randomly selects five patients for display in montage #
    sample_files = random.sample(files, 5)

    # column labels for the four image types #
    col_labels = [
        "Original",
        "Reconstructed",
        "Reconstruction Error",
        "Scaled Subtraction\nReconstruction Error"
    ]

    # create a 5x4 subplot grid one row per patient #
    fig, axes = plt.subplots(5, 4, figsize=(14, 14))

    # add overall title #
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.01)

    # add column labels at the top #
    for j, col_label in enumerate(col_labels):
        axes[0, j].set_title(col_label, fontsize=11, fontweight='bold', pad=8)

    for i, file in enumerate(sample_files):
        # derive the common filename prefix for this patient #
        base = file.replace("original.png", "")

        # add row label on the left side #
        axes[i, 0].set_ylabel(f"Patient {i + 1}", fontsize=11, fontweight='bold',
                               rotation=90, labelpad=10)

        # the four image types to display for each patient #
        images = [
            base + "original.png",
            base + "reconstructed.png",
            base + "reconstruction_error.png",
            base + "scaled_subtraction_reconstruction_error.png"
        ]

        for j, img_name in enumerate(images):
            path = os.path.join(folder, img_name)
            img = Image.open(path)
            # display in grayscale; only hide tick marks not the labels #
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            # keep the spine visible for clean grid appearance #
            for spine in axes[i, j].spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_edgecolor('gray')

    plt.tight_layout()
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_name}")

# montage for healthy-test dataset (label = 0) #
create_montage(
    os.path.join(BASE_PATH, "healthy_test"),
    "montage_healthy_test.png",
    "Healthy Test Samples — Autoencoder Reconstruction Analysis"
)

# montage for abnormal/pathological dataset (label = 1) #
create_montage(
    os.path.join(BASE_PATH, "abnormal"),
    "montage_abnormal.png",
    "Abnormal/Pathological Samples — Autoencoder Reconstruction Analysis"
)