import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def create_table_image(model_names, scores, score_names, output_destination):
    fig, ax = plt.subplots(figsize=(24, 12))  # Adjust the figure size as necessary
    ax.axis('tight')
    ax.axis('off')

    # Transpose the data so that models are rows and score names are columns
    table_data = np.array(scores).T.tolist()

    # Create table headers (score names as columns)
    headers = ["Model"] + score_names

    # Add model names as the first column in each row
    table_data_with_models = [[model_names[i]] + table_data[i] for i in range(len(model_names))]

    # Create the table
    table = ax.table(cellText=table_data_with_models, colLabels=headers, cellLoc='center', loc='center')

    # Save the table as an image
    plt.savefig(output_destination, bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close()

def create_heatmap(model_names, scores, score_names, output_destination, dpi=300):
    # Convert the scores into a DataFrame for easier plotting with seaborn
    df = pd.DataFrame(np.array(scores).T, index=model_names, columns=score_names)

    # Create a heatmap
    plt.figure(figsize=(8, 6))  # Set the figure size for better resolution
    sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=.5)

    # Save the heatmap as an image with high DPI
    plt.savefig(output_destination, bbox_inches='tight', dpi=dpi)
    plt.close()