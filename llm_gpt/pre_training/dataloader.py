#https://www.youtube.com/watch?v=zuj_NJNouAA&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&index=27

import os
from torch.utils.data import DataLoader, Dataset

from llm_gpt.gpt_model.chatgpt import GPT2_CONFIG
from llm_gpt.pre_training.tokenizer import Tokenizer

filepath = "/Users/kunalgau/Downloads/the-verdict.txt"




class File_Loader:

    def read_content(self,filepath):
        if  os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                text_data = file.read()
                return text_data






class GTPDataset(Dataset):

    def __init__(self, text, max_length, stride=128):
        self.input_ids = []
        self.target_ids = []
        self.tokenizer = Tokenizer()


        token_ids = self.tokenizer.text_to_token(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            output_chunk = token_ids[i +1 : i + max_length +1]
            self.input_ids.append(input_chunk)
            self.target_ids.append(output_chunk)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]



    def print_tensor(self):

        for i in range(len(self.input_ids)):
            print(self.input_ids[i], self.target_ids[i])



class GTPDataLoader:

    def __init__(self):
        self.tokenizer = Tokenizer()

    def create_dataloader(self, text, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):

        dataset = GTPDataset(text, max_length, stride)

        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                drop_last=drop_last,
                                num_workers=num_workers)

        return dataloader


    def create_train_val_data(self, path_of_data_file="/Users/kunalgau/Downloads/the-verdict.txt"):

        file_loader = File_Loader()
        file_content = file_loader.read_content(path_of_data_file)

        # gtp_dataset = GTPDataset(file_content, GPT2_CONFIG["context_length"], GPT2_CONFIG["context_length"])
        # gtp_dataset.print_tensor()

        # divide the data into training and validation
        train_ratio = 0.90
        split_idx = int(train_ratio * len(file_content))

        # divide the file input into training and validation and create tensor out of it.
        train_data = file_content[:split_idx]
        val_data = file_content[split_idx:]



        train_loader = self.create_dataloader(
                                    train_data,
                                    batch_size=2,
                                    max_length=GPT2_CONFIG["context_length"],
                                    stride=GPT2_CONFIG["context_length"],
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=0 )

        val_loader = self.create_dataloader(
                                    val_data,
                                    batch_size=2,
                                    max_length=GPT2_CONFIG["context_length"],
                                    stride=GPT2_CONFIG["context_length"],
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=0 )

        print("Train Loader")
        for x, y in train_loader:
            print(x.shape, y.shape)

        print("Val Loader")
        for x, y in val_loader:
            print(x.shape, y.shape)

        return train_loader, val_loader











if __name__ == "__main__":


    file_loader = File_Loader()
    file_content = file_loader.read_content("/Users/kunalgau/Downloads/the-verdict.txt")

    #gtp_dataset = GTPDataset(file_content, GPT2_CONFIG["context_length"], GPT2_CONFIG["context_length"])
    #gtp_dataset.print_tensor()

    # divide the data into training and validation
    train_ratio = 0.90
    split_idx = int(train_ratio * len(file_content))

    # divide the file input into training and validation and create tensor out of it.
    train_data =  file_content[:split_idx]
    val_data = file_content[split_idx:]

    dataloader = GTPDataLoader()

    train_loader = dataloader.create_dataloader(train_data,
                                 batch_size=2,
                                 max_length= GPT2_CONFIG["context_length"],
                                 stride= GPT2_CONFIG["context_length"],
                                 shuffle=True,
                                 drop_last=True,
                                 num_workers=0
                                 )

    val_loader = dataloader.create_dataloader(val_data,
                                              batch_size=2,
                                              max_length=GPT2_CONFIG["context_length"],
                                                stride=GPT2_CONFIG["context_length"],
                                              shuffle=True,
                                              drop_last=True,
                                              num_workers=0)


    print("Train Loader")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("Val Loader")
    for x, y in val_loader:
        print(x.shape, y.shape)


    #print(len(tokenizer.text_to_token(file_content)))





