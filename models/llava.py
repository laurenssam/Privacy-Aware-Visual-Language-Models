from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava.constants import IMAGE_TOKEN_INDEX
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
import re
import torch



class LlaVa():
    def __init__(self, temperature, max_new_tokens):
        print("Loading LlaVa")

        # model_path = "4bit/llava-v1.5-13b-3GB"
        model_path = "liuhaotian/llava-v1.5-13b"
        # model_path = "liuhaotian/llava-v1.6-vicuna-13b"

        disable_torch_init()

        self.model_name = get_model_name_from_path(model_path)
        # self.model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, self.model_name, load_4bit=True)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def prepare_prompt(self, prompt, image):
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in prompt:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, prompt)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, prompt)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + prompt
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"


        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        images_tensor = process_images(
            [image],
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)

        return prompt, images_tensor

    def predict(self, imgs, prompt, in_context_answers=None):
        # prompt = f"""###Human:<image> {prompt} ###Assistant:"""
        prompt, image_tensor = self.prepare_prompt(prompt, imgs[0])
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )
        outputs = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

        return [outputs]

