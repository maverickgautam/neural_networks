import tiktoken
import torch

class Tokenizer:

    def __init__(self):
       self.tokenizer =  tiktoken.get_encoding("gpt2")

    def text_to_token(self,input_text):
        encoded_text = self.tokenizer.encode_ordinary(input_text['text'])
        out_encoded_text = {'ids' : encoded_text, 'len' : len(encoded_text)}
        return out_encoded_text

    def token_to_text(self,token_id):
        flat = token_id.squeeze(0)
        return self.tokenizer.decode(flat.tolist())