import os
import torch
os.environ['TRANSFORMERS_CACHE'] = '/var/scratch/lsamson/LLMS'
os.environ['TORCH_HOME'] = '/var/scratch/lsamson/TORCHHUB'

from lavis.models import load_model_and_preprocess


class InstructBLIP():
    def __init__(self, temperature, max_new_tokens):
        print("Loading InstructBLIP")
    #     # loads InstructBLIP model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

        model_type = "vicuna7b"
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type=model_type,
                                                             is_eval=True, device=self.device)


    def predict(self, list_of_imgs, prompt, no_context_answers=None):
        image = self.vis_processors["eval"](list_of_imgs[0]).unsqueeze(0).to(self.device)
        response = self.model.generate({"image": image, "prompt": prompt}, temperature=self.temperature, max_length=self.max_new_tokens)
        return response
