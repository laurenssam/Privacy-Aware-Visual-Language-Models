import os
import torch
import transformers
from PIL import Image
import sys
# sys.path.append("/var/scratch/lsamson/Otter/src/otter_ai")
from otter_ai import OtterForConditionalGeneration
os.environ['TRANSFORMERS_CACHE'] = '/var/scratch/lsamson/LLMS'
os.environ['TORCH_HOME'] = '/var/scratch/lsamson/TORCHHUB'



class Otter():
    def __init__(self, temperature, max_new_tokens):
        # self.model_path = "luodian/OTTER-Image-MPT7B"
        self.model_path = "luodian/OTTER-9B-LA-InContext" # WHAT IS THIS?
        self.device = 'cuda:0'
        self.model = OtterForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, device_map='auto')
        self.model.text_tokenizer.padding_side = "left"
        self.tokenizer = self.model.text_tokenizer
        self.image_processor = transformers.CLIPImageProcessor()
        self.model.eval()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens


    def get_formatted_prompt(self, prompt: str, in_context_answers) -> str:
        if in_context_answers:
            return f"<image>User:{prompt} GPT: <answer>{in_context_answers[0]}<|endofchunk|><image>User:{prompt} GPT: <answer>{in_context_answers[1]}<|endofchunk|><image>User:{prompt} GPT: <answer>"

        return f"<image>User: {prompt} GPT:<answer>"

    def predict(self, imgs, prompt, in_context_answers=None):
        input_data = imgs

        # if isinstance(input_data, Image.Image):
        #     if input_data.size == (224, 224) and not any(input_data.getdata()):  # Check if image is blank 224x224 image
        #         vision_x = torch.zeros(1, 1, 1, 3, 224, 224, dtype=next(self.model.parameters()).dtype)
        #     else:
        vision_x = self.image_processor.preprocess(input_data, return_tensors="pt")["pixel_values"].unsqueeze(
            1).unsqueeze(0)
        # else:
        #     raise ValueError("Invalid input data. Expected PIL Image.")

        lang_x = self.model.text_tokenizer(
            [
                self.get_formatted_prompt(prompt, in_context_answers),
            ],
            return_tensors="pt",
        )
        model_dtype = next(self.model.parameters()).dtype

        vision_x = vision_x.to(dtype=model_dtype)
        lang_x_input_ids = lang_x["input_ids"]
        lang_x_attention_mask = lang_x["attention_mask"]
        bad_words_id = self.tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids

        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device),
            lang_x=lang_x_input_ids.to(self.model.device),
            attention_mask=lang_x_attention_mask.to(self.model.device),
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            bad_words_ids=bad_words_id,
            pad_token_id=self.tokenizer.eos_token_id

        )
        parsed_output = \
        self.model.text_tokenizer.decode(generated_text[0]).split("<answer>")[-1].lstrip().rstrip().split("<|endofchunk|>")[
            0].lstrip().rstrip().lstrip('"').rstrip('"')
        return [parsed_output]


