import os
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer

os.environ["TRANSFORMERS_CACHE"] = "/var/scratch/lsamson/LLMS"
os.environ["TORCH_HOME"] = "/var/scratch/lsamson/TORCHHUB"


class CoAgent:
    def __init__(self, temperature, max_new_tokens):
        print("Loading CoAgent")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_path = "THUDM/cogagent-chat-hf"
        # self.model_path = "THUDM/cogvlm-chat-hf"
        self.tokenizer_path = "lmsys/vicuna-7b-v1.5"
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.tokenizer = LlamaTokenizer.from_pretrained(self.tokenizer_path)
        self.torch_type = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_type,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
            trust_remote_code=True,
        ).eval()
        # add any transformers params here.
        self.gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": 0,
            "top_k": 1,
            "do_sample": False,
        }

    def predict(self, imgs, prompt, in_context_answers=None):
        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer, query=prompt, images=[imgs[0]]
        )
        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(self.device),
            "token_type_ids": input_by_model["token_type_ids"]
            .unsqueeze(0)
            .to(self.device),
            "attention_mask": input_by_model["attention_mask"]
            .unsqueeze(0)
            .to(self.device),
            "images": [
                [input_by_model["images"][0].to(self.device).to(self.torch_type)]
            ],
        }
        if "cross_images" in input_by_model and input_by_model["cross_images"]:
            inputs["cross_images"] = [
                [input_by_model["cross_images"][0].to(self.device).to(self.torch_type)]
            ]

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            response = self.tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]
        return [response]
