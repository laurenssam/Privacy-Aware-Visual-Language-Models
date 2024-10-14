import sys
from pathlib import Path

main_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(main_dir))
import argparse

from helpers import load_pickle, create_experiment_folder
from visualizations import create_heatmap

METRICS = ["accuracy", "f1_score", "recall", "precision", "specificity"]
PRIVATE_CLASSES = ['passport', 'face', 'tattoo', 'debit_card', 'license_plate', 'nudity', 'private_chat', 'fingerprint']


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run an experiment with a given model and dataset.")
    parser.add_argument('--dataset_name', type=str, default="privbench",
                        help="Name of the dataset to be used in the experiment.")
    parser.add_argument('--experiment', type=Path, default="results",
                        help="Folder where the experiment output will be saved.")
    parser.add_argument('--comparison', type=Path, default="comparisons",
                        help="Folder where the experiment output will be saved.")
    return parser.parse_args()

# Example usage:
def compare_experiments(model_names, experiment_ids, dataset_name, experiment_folder, output_folder):
    experiments_folder = experiment_folder / dataset_name
    output_folder = create_experiment_folder(output_folder)
    scores_dict = {}
    scores_dict_class = {}
    for idx, model in enumerate(model_names):
        scores = load_pickle(experiments_folder / Path(model) / Path(str(experiment_ids[idx])) / "metrics.pickle")
        scores_dict[model] = scores

        scores_per_class = load_pickle(experiments_folder / Path(model) / Path(str(experiment_ids[idx])) / "metrics_per_class.pickle")
        scores_dict_class[model] = scores_per_class


    table_scores = [[scores_dict[model]["metrics"][metric] for model in model_names] for metric in METRICS]
    create_heatmap(model_names, table_scores , METRICS, output_folder / "scores_table.jpg")

    for metric in METRICS:
        all_scores = [scores_dict[model]['metrics'][metric] for model in model_names]
        table_scores = [all_scores] + [[scores_dict_class[model]['metrics'][class_name][metric] for model in model_names] for class_name in PRIVATE_CLASSES]

        create_heatmap(model_names, table_scores , ["all"] + PRIVATE_CLASSES, output_folder / f"class_analysis_{metric}.jpg")






if __name__ == "__main__":
    args = parse_arguments()

    # model_names = ["coagent", "cogvlm", "fuyu", "chatgpt", "blip", "moellava",  "sharegpt", "tinyllava", "otter"]
    # experiment_ids = ['1'] * len(model_names)
    # compare_experiments(model_names, experiment_ids, args.dataset_name, args.experiment, args.comparison)

    model_names = [ "blip", "coagent", "cogvlm", "fuyu", "llava", "moellava", "otter", "sharegpt", "tinyllava"]
    experiment_ids = ['1'] * len(model_names)
    compare_experiments(model_names + ["tinyllava"], experiment_ids + ['2'], args.dataset_name, args.experiment, args.comparison)