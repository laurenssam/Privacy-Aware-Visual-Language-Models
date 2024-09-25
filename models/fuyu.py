from transformers import FuyuProcessor, FuyuForCausalLM, BitsAndBytesConfig, AutoTokenizer, FuyuImageProcessor
import torch

from helpers import keep_after_substring


class Fuyu():
    def __init__(self, args):
        print("Loading Fuyu")
        # load model and processor
        # model_id = "adept/fuyu-8b"
        self.device = 'cuda:0'
        model_id = "ybelkada/fuyu-8b-sharded"
        # self.processor = FuyuProcessor.from_pretrained(model_id, torch_dtype=torch.bfloat16, cache_dir="/var/scratch/lsamson/LLMS",)
        # self.model = FuyuForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, cache_dir="/var/scratch/lsamson/LLMS", device_map=self.device)
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.dtype
        )
        self.model = FuyuForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.processor = FuyuProcessor(image_processor=FuyuImageProcessor(), tokenizer=self.tokenizer)
        self.start_answer = '\x04'
        self.args = args

    def predict(self, list_of_imgs, prompt, in_context_answers=None):

        inputs = self.processor(text=prompt, images=list_of_imgs[0]).to(self.device)
        generation_output = self.model.generate(**inputs, max_new_tokens=self.args.max_new_tokens, temperature=self.args.temperature)
        generation_text = self.processor.batch_decode(generation_output, skip_special_tokens=True)
        generation_text = [keep_after_substring(answer, self.start_answer) for answer in generation_text]
        return generation_text


