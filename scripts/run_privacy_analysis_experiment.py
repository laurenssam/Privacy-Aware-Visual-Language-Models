import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from googletrans import Translator
import numpy as np
from copy import deepcopy
from PIL import Image

main_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(main_dir))

from helpers import create_experiment_folder, write_to_file, _get_label, evaluate_prediction, assign_confusion_label, \
    calculate_metrics_from_confusion_matrix, txt_to_string, format_evaluation_output, test_if_model_works, save_pickle, \
    _get_class, calculate_recall, format_recall_results, file_to_list, in_split_file
from models.init_model import init_model
from prompts.init_prompt import init_prompt
from data.init_data import init_data




class Experiment:
    def __init__(self, model_name, dataset_name, dataset_path, output_folder, prompt, temperature, max_new_tokens, translate, tl_model):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_folder = create_experiment_folder(output_folder / Path(dataset_name) / Path(model_name))
        self.split_folder = dataset_path.parent / "splits"
        self.predictions_folder = self.output_folder / "predictions"
        self.predictions_folder.mkdir(exist_ok=True, parents=True)
        self.prompt = prompt
        self.translate = translate
        self.dataloader = init_data(self.dataset_name, dataset_path)
        self.model = init_model(self.model_name, temperature, max_new_tokens, tl_model)
        test_if_model_works(self.model, self.output_folder)
        self.prompt = init_prompt(prompt, translate)
        write_to_file(self.output_folder / "prompt.txt", self.prompt)
        self.confusion_matrix = {"tp":0, "tn":0, "fp":0, "fn":0}
        self.rejections = 0


    def run(self):
        """Run the experiment."""
        print(f"Running experiment with model '{self.model_name}' on dataset '{self.dataset_name}'.")
        # Placeholder for actual experiment logic
        for idx, (img, label, img_path) in enumerate(tqdm(self.dataloader)):
            predictions = self.model.predict(img, self.prompt)
            for idy, prediction in enumerate(predictions):
                if self.translate and prediction:
                    translator = Translator()
                    prediction = translator.translate(prediction, src=self.translate, dest='en').text
                self.save_results(prediction, img_path[idy], label[idy])


    def save_results(self, prediction, img_path, label):
        """Save experiment results to the output folder."""
        result_path = self.predictions_folder / f"{img_path.stem}_{0 if label == 'public' else 1}.txt"
        Image.open(img_path).save(self.predictions_folder / img_path.name)
        write_to_file(result_path, prediction)

    def evaluate(self):
        for idx, prediction in enumerate(self.predictions_folder.iterdir()):
            if prediction.suffix != ".txt":
                continue
            elif self.model_name == "chatgpt" and "nudity" in prediction.stem:
                continue
            label = _get_label(prediction)
            prediction_score = evaluate_prediction(txt_to_string(prediction))
            if prediction == "reject":
                self.rejections += 1
            confusion_label = assign_confusion_label(prediction_score, label)
            self.confusion_matrix[confusion_label] += 1


        metrics = calculate_metrics_from_confusion_matrix(self.confusion_matrix['tp'], self.confusion_matrix['fp'], self.confusion_matrix['fn'], self.confusion_matrix['tn'])
        pretty_print = format_evaluation_output(metrics, "All Classes", self.rejections, self.confusion_matrix)
        write_to_file(self.output_folder / "metrics.txt", pretty_print)
        save_pickle({"metrics":metrics, "rejections":self.rejections, "confusion_matrix":self.confusion_matrix}, self.output_folder / "metrics.pickle")
        print(pretty_print)

    def evaluate_per_class(self):
        score_dict = {"precision": [], "recall": [], "f1_score": [], "accuracy": [], "specificity": []}
        scores_per_class =  {class_name: deepcopy(score_dict) for class_name in PRIVATE_CLASSES}
        for split_file in self.split_folder.glob("*.txt"):
            split_integers = file_to_list(split_file)
            per_split_class_confusion_matrix = {class_name: {"tp": 0, "fn": 0} for class_name in PRIVATE_CLASSES}
            per_split_negatives = {"tn": 0, "fp": 0}
            for idx, prediction_path in enumerate(self.predictions_folder.iterdir()):
                if prediction_path.suffix != ".txt":
                    continue
                label = _get_label(prediction_path)
                prediction = evaluate_prediction(txt_to_string(prediction_path))
                confusion_label = assign_confusion_label(prediction, label)
                if label == 0 and not in_split_file(split_integers, prediction_path):
                    continue
                elif label == 0:
                    per_split_negatives[confusion_label] += 1
                    continue
                else:
                    class_name = _get_class(prediction_path.stem, PRIVATE_CLASSES)
                    if prediction == "reject":
                        self.rejections += 1
                    per_split_class_confusion_matrix[class_name][confusion_label] += 1
            for class_name, confusion_matrix in per_split_class_confusion_matrix.items():
                metrics_per_class = calculate_metrics_from_confusion_matrix(confusion_matrix['tp'], per_split_negatives['fp'], confusion_matrix['fn'], per_split_negatives['tn'])
                for metric, score in metrics_per_class.items():
                    scores_per_class[class_name][metric].append(score)

        for class_name, scores_dict in scores_per_class.items():
            for metric, scores in scores_dict.items():
                scores_per_class[class_name][metric] = np.mean(scores)
            pretty_print = format_evaluation_output(scores_per_class[class_name], class_name)
            write_to_file(self.output_folder / f"metrics_{class_name}.txt", pretty_print)
            print(pretty_print)
        save_pickle({"metrics": scores_per_class}, self.output_folder / "metrics_per_class.pickle")





def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run an experiment with a given model and dataset.")
    parser.add_argument('--model_name', type=str, default="moellava", help="Name of the model to be used in the experiment.")
    parser.add_argument('--dataset_name', type=str, default="privbench",
                        help="Name of the dataset to be used in the experiment.")
    parser.add_argument('--dataset_path', type=Path, required=True,
                        help="Path of the dataset to be used in the experiment.")
    parser.add_argument('--output_folder', type=Path, default="results",
                        help="Folder where the experiment output will be saved.")
    parser.add_argument('--prompt', type=Path, default="prompts/standard_prompt.txt",
                        help="Path where the prompt is stored.")
    parser.add_argument('--temperature', type=float, default=0.,
                        help="Set the temperature for the Visual Language Model.")
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help="Max number of tokens the VLM can generate.")
    parser.add_argument('--translate', type=str, default=None,
                        help="Language to translate to")
    parser.add_argument('--tl_model', type=str, default=None,
                        help="finetuned tinyllava model")

    return parser.parse_args()

# Example usage:
if __name__ == "__main__":
    args = parse_arguments()
    experiment = Experiment(args.model_name, args.dataset_name, args.dataset_path, args.output_folder, args.prompt, args.temperature, args.max_new_tokens, args.translate, args.tl_model)
    experiment.run()
    experiment.evaluate()
    if args.dataset_name == "privbench":
        from data.privbench import PRIVATE_CLASSES
        experiment.evaluate_per_class()
    elif args.dataset_name == "bivpriv":
        from data.bivpriv import PRIVATE_CLASSES
        experiment.evaluate_per_class()

