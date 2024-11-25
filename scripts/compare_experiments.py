import sys
from pathlib import Path
import numpy as np

main_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(main_dir))
import argparse

from helpers import load_pickle, create_experiment_folder
from visualizations import create_heatmap, create_spider_plot

METRICS = ["f1_score", "precision", "recall", "accuracy"]
PRIVATE_CLASSES = list(
    sorted(
        [
            "passport",
            "face",
            "tattoo",
            "debit_card",
            "license_plate",
            "nudity",
            "private_chat",
            "fingerprint",
        ]
    )
)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run an experiment with a given model and dataset."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="privbench",
        help="Name of the dataset to be used in the experiment.",
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default="results",
        help="Folder where the experiment output will be saved.",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default="comparisons",
        help="Folder where the experiment output will be saved.",
    )
    return parser.parse_args()


# Example usage:
def compare_experiments(
    model_names,
    experiment_ids,
    dataset_name,
    experiment_folder,
    output_folder,
    experiment_names=None,
):
    experiments_folder = experiment_folder / dataset_name
    output_folder = create_experiment_folder(output_folder)
    scores_dict = {}
    scores_dict_class = {}

    for idx, model in enumerate(model_names):
        scores = load_pickle(
            experiments_folder
            / Path(model)
            / Path(str(experiment_ids[idx]))
            / "metrics.pickle"
        )
        scores_dict[
            model + "-" + experiment_names[idx] if experiment_names else model
        ] = scores

        if dataset_name == "privbench":
            scores_per_class = load_pickle(
                experiments_folder
                / Path(model)
                / Path(str(experiment_ids[idx]))
                / "metrics_per_class.pickle"
            )
            scores_dict_class[
                model + "-" + experiment_names[idx] if experiment_names else model
            ] = scores_per_class

    model_names = list(scores_dict.keys())
    table_scores = [
        [scores_dict[model]["metrics"][metric] for model in model_names]
        for metric in METRICS
    ]
    create_heatmap(
        model_names, table_scores, METRICS, output_folder / "scores_table.jpg"
    )

    if dataset_name == "privbench":
        for metric in METRICS:
            all_scores = [
                scores_dict[model]["metrics"][metric] for model in model_names
            ]
            table_scores = [all_scores] + [
                [
                    scores_dict_class[model]["metrics"][class_name][metric]
                    for model in model_names
                ]
                for class_name in PRIVATE_CLASSES
            ]

            create_heatmap(
                model_names,
                table_scores,
                ["all"] + PRIVATE_CLASSES,
                output_folder / f"class_analysis_{metric}.jpg",
            )
            create_spider_plot(
                PRIVATE_CLASSES,
                model_names,
                np.array(
                    [
                        [
                            scores_dict_class[model]["metrics"][class_name][metric]
                            for model in model_names
                        ]
                        for class_name in PRIVATE_CLASSES
                    ]
                ).T,
                output_folder / f"spider_plot_{metric}.jpg",
            )


if __name__ == "__main__":
    args = parse_arguments()

    # model_names = ["coagent", "cogvlm", "fuyu", "chatgpt", "blip", "moellava",  "sharegpt", "tinyllava", "otter"]
    # experiment_ids = ['1'] * len(model_names)
    # compare_experiments(model_names, experiment_ids, "privbench", args.experiment, args.comparison)
    #
    model_names = [
        "blip",
        "coagent",
        "chatgpt",
        "cogvlm",
        "fuyu",
        "llava",
        "moellava",
        "otter",
        "sharegpt",
        "tinyllava",
        "tinyllava",
    ]
    exerperiment_names = [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "zero-shot",
        "privacy-tuned",
    ]

    experiment_ids = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "41", "2"]
    compare_experiments(
        model_names,
        experiment_ids,
        "privbench",
        args.experiment,
        args.comparison,
        exerperiment_names,
    )
    #
    # model_names = ["tinyllava"] * 6
    # exerperiment_names = ["zero-shot", "16-epoch", "20-epoch", "24-epoch", "28-epoch", "32-epoch"]
    # experiment_ids = ['41', '1', '2', '3', '4', '5']
    # compare_experiments(model_names, experiment_ids, "privbench", args.experiment,
    #                     args.comparison, exerperiment_names)
    #
    # model_names = ["tinyllava"] * 21
    # exerperiment_names = ['00', "005", "01", "015", "02", "025", "03", "035", "04", "045", "05", "055", "06", "065", "07", "075", "08", "085", "09", "095", "10"]
    # experiment_ids = ['41', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '4']
    # compare_experiments(model_names, experiment_ids, "privbench", args.experiment,
    #                     args.comparison, exerperiment_names)
    # model_names = ["tinyllava"] * 9
    # exerperiment_names = ["zero-shot", "debit", "face", "fingerprint", "license", "nudity", "passport", "private_chat", "tattoo"]
    # experiment_ids = ['41', '25', '26', '27', '28', '29', '30', '31', '32']
    # compare_experiments(model_names, experiment_ids, "privbench", args.experiment,
    #                     args.comparison, exerperiment_names)
    # # #
    # model_names = ["tinyllava"] * 9
    # exerperiment_names = ["zero-shot", "debit", "face", "fingerprint", "license", "nudity", "passport", "private_chat", "tattoo"]
    # experiment_ids = ['41', '33', '34', '35', '36', '37', '38', '39', '40']
    # compare_experiments(model_names, experiment_ids, "privbench", args.experiment,
    #                     args.comparison, exerperiment_names)
    #
    # model_names = ["blip", "chatgpt", "coagent", "cogvlm", "fuyu", "llava", "moellava", "otter", "sharegpt",
    #                "tinyllava", "tinyllava"]
    # exerperiment_names = ['', '', '', '', '', '', '', '', '', "zero-shot", "privacy-tuned"]
    #
    experiment_ids = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "3"]
    compare_experiments(
        model_names,
        experiment_ids,
        "privalert",
        args.experiment,
        args.comparison,
        exerperiment_names,
    )

    model_names = [
        "blip",
        "chatgpt",
        "coagent",
        "cogvlm",
        "fuyu",
        "llava",
        "moellava",
        "otter",
        "sharegpt",
        "tinyllava",
        "tinyllava",
    ]
    exerperiment_names = [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "zero-shot",
        "privacy-tuned",
    ]

    experiment_ids = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "3"]
    compare_experiments(
        model_names,
        experiment_ids,
        "bivpriv",
        args.experiment,
        args.comparison,
        exerperiment_names,
    )
    #
    # model_names = 4 * ['chatgpt'] + 4 * ["sharegpt"]
    # exerperiment_names = ['en', 'de', 'cn', 'ru'] * 2
    #
    # experiment_ids = ['1', '2', '3', '4', '1', '2', '3', '4']
    # compare_experiments(model_names, experiment_ids, "privbench", args.experiment, args.comparison, exerperiment_names)
    #
    model_names = [
        "blip",
        "chatgpt",
        "coagent",
        "cogvlm",
        "fuyu",
        "llava",
        "moellava",
        "otter",
        "sharegpt",
        "tinyllava",
        "tinyllava",
    ]
    exerperiment_names = [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "zero-shot",
        "privacy-tuned",
    ]

    experiment_ids = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "2"]
    compare_experiments(
        model_names,
        experiment_ids,
        "privbench_hard",
        args.experiment,
        args.comparison,
        exerperiment_names,
    )

    ### PROMPT WITH CLASSES
    # model_names = ["blip", "chatgpt", "coagent", "cogvlm", "fuyu", "llava", "moellava", "otter", "sharegpt",
    #                "tinyllava", "tinyllava"]
    # exerperiment_names = ['', '', '', '', '', '', '', '', '', "zero-shot", "privacy-tuned"]
    #
    # experiment_ids = ['2', '5', '2', '2', '2', '2', '2', '3', '5', '42', '43']
    # compare_experiments(model_names, experiment_ids, "privbench", args.experiment, args.comparison,
    #                     exerperiment_names)

    # #### SPIDER PLOT ####
    # model_names = ["tinyllava", "chatgpt", "llava"]
    # experiment_ids = ['4', '1', '1']
    # compare_experiments(model_names, experiment_ids, "privbench", args.experiment,
    #                     args.comparison)
