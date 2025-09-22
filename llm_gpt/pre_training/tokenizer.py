import tiktoken
import torch

class Tokenizer:

    def __init__(self):
       self.tokenizer =  tiktoken.get_encoding("gpt2")


    def text_to_token(self,input_text):
        encoded_text = self.tokenizer.encode(input_text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded_text)
        return encoded_tensor

    def token_to_text(self,token_id):
        flat = token_id.squeeze(0)
        return self.tokenizer.decode(flat.tolist())