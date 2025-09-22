import torch
from llm_gpt.gpt_model.chatgpt import GPT2, GPT2_CONFIG
from  llm_gpt.pre_training.tokenizer import Tokenizer
class GenerateToken:

    def generate_next_token_text(self, model, idx, max_new_token, context_size):

        for _ in range(max_new_token):


            idx_selected = idx[:,-context_size:]
            #idx_selected = idx[-context_size:]
            #idx_selected = idx


            with torch.no_grad():
                logits = model(idx_selected)

            print("logits received ", logits)



            # focus on only the last step of every batch
            logits = logits[:,-1,:]

            print("logits last row ", logits)



            probabilities = torch.softmax(logits, dim=-1)

            print("probabilities", probabilities.shape)


            # Select the index with highest probablity
            idx_next = torch.argmax(probabilities, dim=-1, keepdim=True)

            print(idx_next)
            print(idx)

            idx = torch.cat((idx, idx_next), dim=-1)



        return idx


if __name__ == '__main__':

    # Model
    model = GPT2(GPT2_CONFIG)

    # load the trained model parameters to generate the next token
    model.load_state_dict(torch.load('/Users/kunalgau/data_science/neural_networks/llm_gpt/model.pt'))
    model.eval()
    #print(model)

    #input to model
    prompt = "Who is Mr Gisburn ?"
    tokenizer = Tokenizer()
    encoded = tokenizer.text_to_token(prompt)
    encoded_tensor = encoded.unsqueeze(0)

    print("encoded : ", encoded)
    print("encoded tensor : ", encoded_tensor)

    idx = encoded_tensor
    max_new_token = 50
    context_size = GPT2_CONFIG["context_length"]

    generate = GenerateToken()
    out = generate.generate_next_token_text(model, encoded_tensor, max_new_token, context_size)
    decoded_text = tokenizer.token_to_text(out)
    #out = model(torch.randn(1, 3, 256, 256))
    print(decoded_text)
