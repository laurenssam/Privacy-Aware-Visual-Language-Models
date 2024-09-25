import torch
import os
from pathlib import Path
import re

from tinyllava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from tinyllava.conversation import conv_templates, SeparatorStyle
from tinyllava.model.builder import load_pretrained_model
from tinyllava.utils import disable_torch_init
from tinyllava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria)


os.environ['TRANSFORMERS_CACHE'] = '/var/scratch/lsamson/LLMS'
os.environ['TORCH_HOME'] = '/var/scratch/lsamson/TORCHHUB'

class TinyLlaVa():
    def __init__(self, temperature, max_new_tokens, tl_model=None):
        print("Loading CoAgent")
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.model_base = "bczhou/TinyLLaVA-3.1B" if tl_model else None
        self.model_path = tl_model if tl_model else "bczhou/TinyLLaVA-3.1B"
        if tl_model and "2.0" in tl_model:
            self.model_base = "bczhou/TinyLLaVA-2.0B"
        print(f"Model Base: {self.model_base}")
        print(f"Model Path: {self.model_path}")

        disable_torch_init()
        # self.model_path = "/var/scratch/lsamson/LLMS/TinyLLaVa_Privacy_Aware/"

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=self.model_path,
            model_base=self.model_base,
            model_name=get_model_name_from_path(self.model_path))


    def predict(self, imgs, prompt, in_context_answers=None):
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in prompt:
            if self.model.config.mm_use_im_start_end:
                prompt = re.sub(IMAGE_PLACEHOLDER, image_token_se, prompt)
            else:
                prompt = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, prompt)
        else:
            if self.model.config.mm_use_im_start_end:
                prompt = image_token_se + "\n" + prompt
            else:
                prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt


        conv_mode = "phi"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        images_tensor = process_images(
            imgs,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        if len(imgs) > 1:
            input_ids = input_ids.repeat(len(imgs), 1)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        outputs = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        all_outputs = []
        for output in outputs:
            output = output.strip()
            if output.endswith(stop_str):
                output = output[: -len(stop_str)]
            output = output.strip()
            all_outputs.append(output)
        return all_outputs



