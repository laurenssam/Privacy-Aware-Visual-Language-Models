import sys
import argparse
from pathlib import Path
main_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(main_dir))

from helpers import create_experiment_folder, write_to_file, _get_label, evaluate_prediction, assign_confusion_label, \
    calculate_metrics_from_confusion_matrix, txt_to_string, format_evaluation_output, test_if_model_works, save_pickle, \
    _get_class, calculate_recall, format_recall_results
from models.init_model import init_model
from prompts.init_prompt import init_prompt
from data.init_data import init_data
from data.privbench import PRIVATE_CLASSES

class Experiment:
    def __init__(self, model_name, dataset_name, dataset_path, output_folder, prompt, temperature, max_new_tokens):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_folder = create_experiment_folder(output_folder / Path(dataset_name) / Path(model_name))
        self.predictions_folder = self.output_folder / "predictions"
        self.predictions_folder.mkdir(exist_ok=True, parents=True)
        self.prompt = prompt
        self.dataloader = init_data(self.dataset_name, dataset_path)
        self.model = init_model(self.model_name, temperature, max_new_tokens)
        test_if_model_works(self.model, self.output_folder)
        self.prompt = init_prompt(prompt)
        write_to_file(self.output_folder / "prompt.txt", self.prompt)
        self.confusion_matrix = {"tp":0, "tn":0, "fp":0, "fn":0}
        self.class_confusion_matrix = {class_name: {"tp":0, "fn":0} for class_name in PRIVATE_CLASSES}
        self.rejections = 0


    def run(self):
        """Run the experiment."""
        print(f"Running experiment with model '{self.model_name}' on dataset '{self.dataset_name}'.")
        # Placeholder for actual experiment logic
        for idx, (img, label, img_path) in enumerate(self.dataloader):
            predictions = self.model.predict(img, self.prompt)
            for idy, prediction in enumerate(predictions):
                self.save_results(prediction, img_path[idy], label[idy])


    def save_results(self, prediction, img_path, label):
        """Save experiment results to the output folder."""
        result_path = self.predictions_folder / f"{img_path.stem}_{0 if label == 'public' else 1}.txt"
        write_to_file(result_path, prediction)

    def evaluate(self):
        for idx, prediction in enumerate(self.predictions_folder.iterdir()):
            label = _get_label(prediction)
            prediction = evaluate_prediction(txt_to_string(prediction))
            if prediction == "reject":
                self.rejections += 1
            confusion_label = assign_confusion_label(prediction, label)
            self.confusion_matrix[confusion_label] += 1
        metrics = calculate_metrics_from_confusion_matrix(self.confusion_matrix['tp'], self.confusion_matrix['fp'], self.confusion_matrix['fn'], self.confusion_matrix['tn'])
        pretty_print = format_evaluation_output(metrics, self.rejections, self.confusion_matrix)
        write_to_file(self.output_folder / "metrics.txt", pretty_print)
        save_pickle({"metrics":metrics, "rejections":self.rejections, "confusion_matrix":self.confusion_matrix}, self.output_folder / "metrics.pickle")
        print(pretty_print)

    def evaluate_per_class(self):
        negatives = {"tn":0, "fp":0}
        for idx, prediction_path in enumerate(self.predictions_folder.iterdir()):
            label = _get_label(prediction_path)
            prediction = evaluate_prediction(txt_to_string(prediction_path))
            confusion_label = assign_confusion_label(prediction, label)
            if label == 0:
                negatives[confusion_label] += 1
                continue
            class_name = _get_class(prediction_path)
            if prediction == "reject":
                self.rejections += 1
            self.class_confusion_matrix[class_name][confusion_label] += 1

        metrics_all_class = {}
        for class_name, confusion_matrix in self.class_confusion_matrix.items():
            metrics_per_class = calculate_metrics_from_confusion_matrix(confusion_matrix['tp'], negatives['fp'], confusion_matrix['fn'], negatives['tn'])
            metrics_all_class[class_name] = metrics_per_class
            pretty_print = format_evaluation_output(metrics_per_class, self.rejections, confusion_matrix | negatives)
            write_to_file(self.output_folder / f"metrics_{class_name}.txt", pretty_print)
            print(class_name)
            print(pretty_print)
        save_pickle({"metrics": metrics_all_class, "rejections": self.rejections, "confusion_matrix": self.class_confusion_matrix},
                        self.output_folder / "metrics_per_class.pickle")





def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run an experiment with a given model and dataset.")
    parser.add_argument('--model_name', type=str, default="moellava", help="Name of the model to be used in the experiment.")
    parser.add_argument('--dataset_name', type=str, default="privbench",
                        help="Name of the dataset to be used in the experiment.")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help="Path of the dataset to be used in the experiment.")
    parser.add_argument('--output_folder', type=Path, default="results",
                        help="Folder where the experiment output will be saved.")
    parser.add_argument('--prompt', type=Path, default="prompts/standard_prompt.txt",
                        help="Path where the prompt is stored.")
    parser.add_argument('--temperature', type=float, default=0.,
                        help="Set the temperature for the Visual Language Model.")
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help="Max number of tokens the VLM can generate.")

    return parser.parse_args()

# Example usage:
if __name__ == "__main__":
    args = parse_arguments()
    experiment = Experiment(args.model_name, args.dataset_name, args.dataset_path, args.output_folder, args.prompt, args.temperature, args.max_new_tokens)
    experiment.run()
    experiment.evaluate()
    experiment.evaluate_per_class()
