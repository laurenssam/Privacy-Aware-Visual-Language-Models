import torch
import os
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


os.environ['TRANSFORMERS_CACHE'] = '/var/scratch/lsamson/LLMS'
os.environ['TORCH_HOME'] = '/var/scratch/lsamson/TORCHHUB'

class MoeLLava():
    def __init__(self, temperature=0, max_new_tokens=512):
        print("LOADING MOELLAVA")
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        disable_torch_init()
        self.model_path = 'LanguageBind/MoE-LLaVA-Phi2-2.7B-4e'  # LanguageBind/MoE-LLaVA-Qwen-1.8B-4e or LanguageBind/MoE-LLaVA-StableLM-1.6B-4e
        self.device = 'cuda:0'
        load_4bit, load_8bit = False, False  # FIXME: Deepspeed support 4bit or 8bit?
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, processor, context_len = load_pretrained_model(self.model_path, None, model_name, load_8bit,
                                                                             load_4bit, device=self.device)
        self.image_processor = processor['image']


    def predict(self, imgs, prompt, in_context_answers=None):
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv_mode = "phi"  # qwen or stablelm
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        image_tensor = self.image_processor.preprocess([imgs[0]], return_tensors='pt')['pixel_values'].to(self.device, dtype=torch.float16)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        return [outputs]



