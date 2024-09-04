from models.moellava import MoeLLava


def init_model(model_name, temperature, max_new_tokens):
    if model_name.lower() == "moellava":
        return MoeLLava(temperature, max_new_tokens)