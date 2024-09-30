from pathlib import Path
import re
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import cv2
import pickle

def in_split_file(split_integers, prediction_path):
    file_name = prediction_path.stem
    if "public" in file_name and int(file_name.split("_")[1]) in split_integers:
        return True
    else:
        return False

def list_to_file(int_list, filename):
    """
    Writes a list of integers to a text file, one integer per line.

    :param int_list: List of integers to write.
    :param filename: Name of the file to write to.
    """
    try:
        with open(filename, 'w') as file:
            for number in int_list:
                file.write(f"{number}\n")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")

def file_to_list(filename):
    """
    Reads a text file containing integers (one per line) and returns a list of integers.

    :param filename: Name of the file to read from.
    :return: List of integers read from the file.
    """
    int_list = []
    try:
        with open(filename, 'r') as file:
            for line_number, line in enumerate(file, start=1):
                stripped_line = line.strip()
                if stripped_line:  # Skip empty lines
                    try:
                        number = int(stripped_line)
                        int_list.append(number)
                    except ValueError:
                        print(f"Warning: Line {line_number} ('{stripped_line}') is not a valid integer and will be skipped.")
    except FileNotFoundError:
        print(f"The file '{filename}' does not exist.")
    except IOError as e:
        print(f"An error occurred while reading the file: {e}")
    return int_list

def save_pickle(obj, filename):
    """
    Save a Python object to a pickle file.

    Parameters:
    obj (any): The Python object to be pickled.
    filename (str): The path to the file where the object will be saved.
    """
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load_pickle(filename):
    """
    Load a Python object from a pickle file.

    Parameters:
    filename (str): The path to the pickle file.

    Returns:
    any: The Python object that was loaded from the pickle file.
    """
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    return obj


def create_experiment_folder(base_path):
    """
    Creates a new folder for an experiment. The folder name will be an incremented integer,
    starting from 1 if no folders exist.

    Args:
    - base_path (str): The base directory where the new experiment folder should be created.

    Returns:
    - Path: The path to the newly created experiment folder.
    """
    base_dir = Path(base_path)

    # Ensure the base directory exists
    base_dir.mkdir(parents=True, exist_ok=True)

    # Start with folder name "1"
    folder_number = 1
    new_folder_path = base_dir / str(folder_number)

    # Increment the folder name until an available one is found
    while new_folder_path.exists():
        folder_number += 1
        new_folder_path = base_dir / str(folder_number)

    # Create the new experiment folder
    new_folder_path.mkdir()
    print(f"Created new experiment folder: {new_folder_path}")

    return new_folder_path

def download_image(url, timeout=None):
    """
    Tries to download images from a given url
    :param url:
    :return: images, if images not avaiable None
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        if timeout:
            response = requests.get(url, timeout=timeout)
        else:
            response = requests.get(url, headers=headers)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        return None

def collate_fn(batch):
    # Unpack the batch into three separate lists: one for the lists of items, one for labels, and one for paths
    images, labels, paths = zip(*batch)

    # Since lists is already a list of lists, we don't need to do much else here

    return list(images), list(labels), list(paths)

# Usage in DataLoader
# Assuming your dataset is called MyDataset
# dataloader = DataLoader(MyDataset(...), batch_size=32, collate_fn=collate_fn)

def load_feature(file):
    return np.load(file)


def remove_padding(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image to make the white padding black
    inverted_gray = cv2.bitwise_not(gray)

    # Apply a binary threshold to create a mask
    _, binary_mask = cv2.threshold(inverted_gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours of the non-zero areas
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop the image using the bounding box
    cropped_image = image[y:y + h, x:x + w]

    # Save the cropped image
    cv2.imwrite(output_path, cropped_image)


def load_features_in_parallel(files, num_workers):
    with mp.Pool(processes=num_workers) as pool:
        # Use imap_unordered for parallel processing with a progress bar
        results = list(tqdm(pool.imap(load_feature, files), total=len(files)))
    return results
def txt_to_string(file_path):
    """
    Converts the content of a text file into a string.

    Parameters:
    file_path (str): The path to the text file.

    Returns:
    str: The content of the file as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def keep_after_substring(example_string, sub_string):
    # Find the index of the substring
    index = example_string.find(sub_string)

    if index != -1:
        # Return everything after the substring
        return example_string[index + len(sub_string):]
    else:
        # Return the original string if substring is not found
        return example_string

def write_to_file(file_path, content):
    """
    Writes a string to a text file. If the file already exists, it will be overwritten.

    Args:
    - file_path (str or Path): The path to the text file where content will be written.
    - content (str): The string content to write to the file.
    """
    file = Path(file_path)
    try:
        # Open the file in write mode and write the content
        file.write_text(content)
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")

def _get_label(file):
    if "0" in file.stem[-1]:
        return 0
    elif "1" in file.stem[-1]:
        return 1
    else:
        raise Exception("Error with reading the label from the image path")

def _get_class(file_name, classes):
    for class_name in classes:
        if class_name in file_name:
            return class_name
    else:
        raise Exception("Something went wrong with splitting the file name into a class")


def evaluate_prediction(prediction):
    if contains_exact_no(prediction):
        return 0
    if contains_exact_yes(prediction):
        return 1
    if "2" in prediction:
        return 0
    if "1" in prediction:
        return 1
    else:
        return "reject"

def test_if_model_works(model, result_folder):
    prompt = "Write a caption for the provided image"
    response = requests.get("https://d1jyxxz9imt9yb.cloudfront.net/medialib/4659/image/p1300x1300/AdobeStock_331202050_538219_reduced.webp")
    img_path = result_folder / "test_image.jpg"
    if response.status_code == 200:
        with open(img_path, 'wb') as file:
            file.write(response.content)
    answer = model.predict([Image.open(img_path)], prompt)
    write_to_file(result_folder / "test_answer.txt", answer[0])
def contains_exact_no(text):
    """
    Checks if the string contains the exact word 'no', without allowing matches with words like 'nobody', 'norm', etc.

    Args:
    - text (str): The input string to check.

    Returns:
    - bool: True if the word 'no' is found as a whole word, False otherwise.
    """
    # Regular expression to match the exact word 'no'
    return bool(re.search(r'\bno\b', text.lower()))


def contains_exact_yes(text):
    """
    Checks if the string contains the exact word 'yes', without allowing matches with words like 'yesterday', 'yesman', etc.

    Args:
    - text (str): The input string to check.

    Returns:
    - bool: True if the word 'yes' is found as a whole word, False otherwise.
    """
    # Regular expression to match the exact word 'yes'
    return bool(re.search(r'\byes\b', text.lower()))

def assign_confusion_label(prediction, label):
    # False Positive with Reject
    if prediction == "reject" and label == 0:
        return "fp"
    # False Negative with Reject
    if prediction == "reject" and label == 1:
        return "fn"
    # False Positive
    if prediction == 1 and label == 0:
        return "fp"
    # False Negative
    if prediction == 0 and label == 1:
        return "fn"
    # True Positive
    if prediction == 1 and label == 1:
        return "tp"
    # True Negative
    if prediction == 0 and label == 0:
        return "tn"


def calculate_metrics_from_confusion_matrix(tp, fp, fn, tn):
    """
    Calculates various performance metrics from a confusion matrix.

    Args:
    - tp (int): True Positives
    - fp (int): False Positives
    - fn (int): False Negatives
    - tn (int): True Negatives

    Returns:
    - dict: A dictionary containing the calculated metrics.
    """
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Return metrics as a dictionary
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "specificity": specificity
    }

def calculate_recall(tp, fn):
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return recall

def format_evaluation_output(metrics, class_name="All", rejections=None, counts=None):
    # Create a list to hold the formatted strings
    output_lines = []

    # Header
    output_lines.append(f"\n{'='*40}")
    output_lines.append(f"{f'Evaluation Metrics {class_name}: ':^40}")
    output_lines.append(f"{'='*40}\n")

    # Metrics
    for metric, value in metrics.items():
        output_lines.append(f"{metric.capitalize():<15}: {value:.4f}")
    if counts:
        # Confusion Matrix Section
        output_lines.append(f"\n{'-'*40}")
        output_lines.append(f"{'Confusion Matrix':^40}")
        output_lines.append(f"{'-'*40}\n")

        # Counts
        for count, value in counts.items():
            output_lines.append(f"{count.upper():<15}: {value}")

    if rejections:
        # Epoch
        output_lines.append(f"\n{'-'*40}")
        output_lines.append(f"{'Number of Rejections':<15}: {rejections}")
        output_lines.append(f"{'='*40}\n")

    # Join all lines into a single string
    return '\n'.join(output_lines)

def format_recall_results(recall_dict):
    """
    Formats the recall per class into a nicely structured string.

    Parameters:
    recall_dict (dict): A dictionary with class names as keys and recall values as values.

    Returns:
    str: A formatted string showing the recall per class.
    """
    # Create a list to hold the formatted strings
    output_lines = []

    # Header
    output_lines.append(f"\n{'='*40}")
    output_lines.append(f"{'Recall Per Class':^40}")
    output_lines.append(f"{'='*40}\n")

    # Format each class and its recall value
    for class_name, recall_value in recall_dict.items():
        output_lines.append(f"{class_name.replace('_', ' ').capitalize():<20}: {recall_value:.2f}")

    # Join all lines into a single string
    return '\n'.join(output_lines)








