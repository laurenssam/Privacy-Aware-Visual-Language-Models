from models.moellava import MoeLLava
from models.chatgpt import ChatGPT
from models.coagent import CoAgent
from models.cogvlm import CogVLM
from models.instruct_blip import InstructBLIP
from models.llava import LlaVa
from models.otter import Otter
from models.sharegpt import ShareGPT4V
from models.tiny_llava import TinyLlaVa


def init_model(model_name, temperature, max_new_tokens):
    if model_name.lower() == "moellava":
        return MoeLLava(temperature, max_new_tokens)
    elif model_name.lower() == "chatgpt":
        return ChatGPT(temperature, max_new_tokens)
    elif model_name.lower() == "coagent":
        return CoAgent(temperature, max_new_tokens)
    elif model_name.lower() == "cogvlm":
        return CogVLM(temperature, max_new_tokens)
    elif model_name.lower() == "blip":
        return InstructBLIP(temperature, max_new_tokens)
    elif model_name.lower() == "llava":
        return LlaVa(temperature, max_new_tokens)
    elif model_name.lower() == "otter":
        return Otter(temperature, max_new_tokens)
    elif model_name.lower() == "sharegpt":
        return ShareGPT4V(temperature, max_new_tokens)
    elif model_name.lower() == "tinyllava":
        return TinyLlaVa(temperature, max_new_tokens)