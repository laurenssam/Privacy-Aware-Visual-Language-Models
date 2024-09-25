from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava.constants import IMAGE_TOKEN_INDEX
import os
os.environ['TRANSFORMERS_CACHE'] = '/var/scratch/lsamson/LLMS'
os.environ['TORCH_HOME'] = '/var/scratch/lsamson/TORCHHUB'



class ShareGPT4V():
    def __init__(self, temperature, max_new_tokens):
        print("Loading ShareGPT4V")

        disable_torch_init()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model("4bit/ShareGPT4V-7B-5GB", None, "llava-v1.5-7b", False, False)

    def predict(self, imgs, prompt, in_context_answers=None):
        prompt = f"""A chat between a curious human and an artificial intelligence assistant. The assistant gives 
        helpful, detailed, and polite answers to the human's questions. USER: <image>\n{prompt} ASSISTANT:"""
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = self.image_processor.preprocess(imgs[0], return_tensors='pt')['pixel_values'].half().cuda()
        # output_ids = self.model.generate(input_ids, images=image_tensor, temperature=self.temperature, max_new_tokens=self.max_new_tokens)
        output_ids = self.model.generate(input_ids, images=image_tensor, temperature=self.temperature, max_new_tokens=self.max_new_tokens)
        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        return [outputs]


