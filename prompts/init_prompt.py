from helpers import txt_to_string
from googletrans import Translator

def init_prompt(path_to_prompt, translate=None):
    prompt = txt_to_string(path_to_prompt)
    if translate:
        translator = Translator()
        prompt = translator.translate(prompt, src='en', dest=translate).text
    return prompt