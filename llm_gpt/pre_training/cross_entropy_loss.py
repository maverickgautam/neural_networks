import torch.nn as nn
import torch
from llm_gpt.gpt_model.chatgpt import GPT2


class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def cal_loss_batch(self, input_batch, target_batch, model:GPT2, device ):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)
        # first apply softmax, and then take corresponding problities from logit corresponding to digits present in Target
        # Watch out - it doesnot chooses the max probablity index : rather it looks at target_batch looks at the id, goes to the voculbalry index pointing to the digit
        loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
        return loss

    def cal_loss_loader(self, data_loader, model:GPT2, device, num_batches=None, input_batch=None):
        total_loss = 0
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        for i ,  ( input_batch, target_batch ) in enumerate(data_loader):
            if i < num_batches:
                loss = self.cal_loss_batch(input_batch,target_batch, model, device)
                total_loss += loss.item()
            else:
                break


        return total_loss / num_batches