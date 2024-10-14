

def init_model(model_name, temperature=0, max_new_tokens=512, tl_model=None):
    if model_name.lower() == "moellava":
        from models.moellava import MoeLLava
        return MoeLLava(temperature, max_new_tokens)
    elif model_name.lower() == "chatgpt":
        from models.chatgpt import ChatGPT
        return ChatGPT(temperature, max_new_tokens)
    elif model_name.lower() == "coagent":
        from models.coagent import CoAgent
        return CoAgent(temperature, max_new_tokens)
    elif model_name.lower() == "fuyu":
        from models.fuyu import Fuyu
        return Fuyu(temperature, max_new_tokens)
    elif model_name.lower() == "cogvlm":
        from models.cogvlm import CogVLM
        return CogVLM(temperature, max_new_tokens)
    elif model_name.lower() == "blip":
        from models.instruct_blip import InstructBLIP
        return InstructBLIP(temperature, max_new_tokens)
    elif model_name.lower() == "llava":
        from models.llava import LlaVa
        return LlaVa(temperature, max_new_tokens)
    elif model_name.lower() == "otter":
        from models.otter import Otter
        return Otter(temperature, max_new_tokens)
    elif model_name.lower() == "sharegpt":
        from models.sharegpt import ShareGPT4V
        return ShareGPT4V(temperature, max_new_tokens)
    elif model_name.lower() == "tinyllava":
        from models.tiny_llava import TinyLlaVa
        return TinyLlaVa(temperature, max_new_tokens, tl_model)