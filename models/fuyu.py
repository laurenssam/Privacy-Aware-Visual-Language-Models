from transformers import (
    FuyuProcessor,
    FuyuForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    FuyuImageProcessor,
)
import torch

from helpers import keep_after_substring


class Fuyu:
    def __init__(self, temperature, max_new_tokens):
        print("Loading Fuyu")
        # load model and processor
        model_id = "adept/fuyu-8b"
        self.device = "cuda:0"
        self.processor = FuyuProcessor.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.model = FuyuForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=self.device)
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        self.start_answer = "\x04"
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def predict(self, list_of_imgs, prompt, in_context_answers=None):

        inputs = self.processor(text=prompt, images=list_of_imgs[0]).to(self.device)
        generation_output = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, temperature=self.temperature
        )
        generation_text = self.processor.batch_decode(
            generation_output, skip_special_tokens=True
        )
        generation_text = [
            keep_after_substring(answer, self.start_answer)
            for answer in generation_text
        ]
        return generation_text
