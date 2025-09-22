from calendar import month

import torch
from llm_gpt.config.gpt_config import GPT2_CONFIG
from llm_gpt.generate_token.generate_token import GenerateToken
from llm_gpt.gpt_model.chatgpt import GPT2
from llm_gpt.pre_training.cross_entropy_loss import CrossEntropyLoss
from llm_gpt.pre_training.dataloader import GTPDataLoader


class GPT_Model_Train:

    def __init__(self):

        # instantiate the model

        torch.manual_seed(123)
        self.model = GPT2(GPT2_CONFIG)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load data from file,  split into training and validation set
        self.dataloader = GTPDataLoader()
        self.train_loader = None
        self.val_loader = None

        self.cross_entropy_loss = CrossEntropyLoss()
        self.train_losses = []
        self.val_losses = []

        self.generate_next_token = GenerateToken()

        self.track_token_seen = []
        self.eval_freq = 5
        self.eval_iter = 5
        self.token_seen = 0
        self.global_step = 0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0004, weight_decay=0.1)

    def evaluate_model(self):
        self.model.eval()

        with torch.no_grad():
            train_loss = self.cross_entropy_loss.cal_loss_loader(self.train_loader, self.model, self.device, num_batches=self.eval_freq)
            val_loss = self.cross_entropy_loss.cal_loss_loader(self.val_loader, self.model, self.device, num_batches=self.eval_freq)

        self.model.train()
        return train_loss, val_loss

    def generate_print_sample(self, device, prompt):
        self.model.eval()

        context_size = self.model.pos_embedding.weight.shape[0]
        encoded = self.dataloader.tokenizer.text_to_token(prompt).unsqueeze(0)
        
        with torch.no_grad():
            token_id = self.generate_next_token.generate_next_token_text(self.model, idx=encoded, max_new_token=10, context_size=context_size)

        decoded_text = self.dataloader.tokenizer.token_to_text(token_id)
        print(decoded_text.replace("\n", ""))
        self.model.train()

    def train_model(self, num_epochs):
        # self.model.to(device=GPT2_CONFIG['device'])
        self.model.train()

        self.train_loader, self.val_loader = self.dataloader.create_train_val_data(path_of_data_file="/Users/kunalgau/Downloads/the-verdict.txt")

        for epoch in range(num_epochs):
            self.model.train()

            for input_batch, target_batch in self.train_loader:
                self.optimizer.zero_grad()
                loss = self.cross_entropy_loss.cal_loss_batch(input_batch, target_batch, self.model, self.device)
                loss.backward()
                self.optimizer.step()
                self.token_seen += input_batch.numel()
                self.global_step += 1

                if self.global_step % self.eval_freq == 0:
                    train_loss, val_loss = self.evaluate_model()
                    self.train_losses.append(train_loss)
                    self.val_losses.append(val_loss)
                    self.track_token_seen.append(self.token_seen)

                    print(f" Epoch  {epoch + 1}   Step {self.global_step:06d} : "
                          f" Train Loss {train_loss:.3f} Val Loss {val_loss:.3f} ")

            self.generate_print_sample(self.device, prompt="The height of his glory")

        # after training save the model weights
        torch.save(self.model.state_dict(), '../model.pt')

        return self.train_losses, self.val_losses, self.track_token_seen


if __name__ == '__main__':
    train = GPT_Model_Train()
    train.train_model(15)
