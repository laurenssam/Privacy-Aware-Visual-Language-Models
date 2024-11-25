import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Affine2D


def create_table_image(model_names, scores, score_names, output_destination):
    fig, ax = plt.subplots(figsize=(24, 12))  # Adjust the figure size as necessary
    ax.axis("tight")
    ax.axis("off")

    # Transpose the data so that models are rows and score names are columns
    table_data = np.array(scores).T.tolist()

    # Create table headers (score names as columns)
    headers = ["Model"] + score_names

    # Add model names as the first column in each row
    table_data_with_models = [
        [model_names[i]] + table_data[i] for i in range(len(model_names))
    ]

    # Create the table
    table = ax.table(
        cellText=table_data_with_models,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )

    # Save the table as an image
    plt.savefig(output_destination, bbox_inches="tight", pad_inches=0.1, dpi=100)
    plt.close()


def create_heatmap(model_names, scores, score_names, output_destination, dpi=300):
    # Convert the scores into a DataFrame for easier plotting with seaborn
    df = (
        pd.DataFrame(np.array(scores).T, index=model_names, columns=score_names)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    # Create a heatmap
    plt.figure(figsize=(8, 6))  # Set the figure size for better resolution
    sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5)

    # Save the heatmap as an image with high DPI
    plt.savefig(output_destination, bbox_inches="tight", dpi=dpi)
    plt.close()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


# In your plotting loop, create a transformation for moving the text
trans = Affine2D().translate(0, 10)


def create_spider_plot(class_names, models, performance_data, location):
    """
    Create a radar chart (spider plot) comparing different models across various class metrics.

    Parameters:
        class_names (list of str): Labels for each class.
        models (list of str): Model names.
        performance_data (ndarray): Data matrix where each row represents a model and columns represent class metrics.
        location (Path): The path object where the plot image will be saved.
        score_name (str): Metric name to display in the plot title. Default is 'f1_score'.
    """
    # Create angles for the radar chart.
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(class_names), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    performance_data = np.concatenate(
        (performance_data, performance_data[:, [0]]), axis=1
    )

    # Plot data
    max_val = 1
    color_map = get_cmap("Accent", len(models))

    for i, model in enumerate(models):
        color = color_map(i)
        if "chatgpt" in model:
            model = "GPT-4"
            color = "red"
        elif "tinyllava" in model:
            model = "TinyLLaVa + PrivTune"
            color = "blue"
        elif "llava" in model:
            model = "LLaVa-1.5"
            color = "green"
        ax.plot(
            angles,
            performance_data[i],
            color=color,
            linewidth=2,
            linestyle="solid",
            label=model,
        )
        ax.fill(angles, performance_data[i], color=color, alpha=0.1)

    # Set the display of grid lines and labels
    ax.set_thetagrids(np.degrees(angles[:-1]), labels=[])
    fontsize = 16
    # Custom placement of class names
    # for label, angle in zip([class_name.replace("_", " ") for class_name in class_names], angles[:-1]):
    #     ha = 'center'  # horizontal alignment
    #     va = 'center'  # vertical alignment
    #     trans=None
    #     adjustment = max_val * 1.14  # normal vertical adjustment
    #
    #     # Check if this is the overlapping class name and adjust its position
    #     if label == "Debit Card":  # replace "YourClassName" with the actual name
    #         trans = Affine2D().translate(90, 0)  # Shifts 10 pixels up in the y-direction
    #     elif label == "License Plate":
    #         trans = Affine2D().translate(-20,  0)
    #     elif label == "Private Chat":
    #         trans = Affine2D().translate(0, 70)
    #     elif label == "Fingerprint":
    #         trans = Affine2D().translate(0, -30)
    #     if angle > np.pi:
    #         va = 'top'
    #     else:
    #         va = 'bottom'
    #     if trans:
    #         ax.text(angle, adjustment, label, transform=ax.transData + trans, ha=ha, va=va, fontsize=fontsize, color='black', rotation=0)
    #     else:
    #         ax.text(angle, adjustment, label, ha=ha, va=va, fontsize=fontsize, color='black', rotation=0)

    # Set the radial limits and labels
    ax.set_ylim(0, max_val)
    ax.set_rgrids(
        [0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0.2", "0.4", "0.6", "0.8", "1.0"],
        angle=90,
        fontsize=10,
    )
    # plt.title(f'AUC-ROC on PrivPix IQ Benchmark', size=15, color='black', y=1.1)
    plt.legend(loc="upper right", fontsize=fontsize, bbox_to_anchor=(1.4, 1.15))
    # Save the figure
    plt.savefig(location, dpi=300)
    plt.close()
