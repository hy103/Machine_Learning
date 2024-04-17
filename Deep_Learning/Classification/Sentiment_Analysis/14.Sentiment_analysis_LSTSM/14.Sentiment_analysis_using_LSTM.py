
from utils import prepare_dataset
import torch

class Reviewdataset(torch.utils.data.Dataset):
    def __init__(self, dataset, word2indx):
        self.data = dataset
        self.word2indx = word2indx

    def text2index(self, x):
        unk_id = self.word2indx['<UNK>']
        input_ids = []
        for word in x.split():
            input_ids.append(self.word2indx.get(word, unk_id))
        return input_ids    


    def __getitem__(self, index):
        sample = self.data[index]
        text , label = sample.split('\t')
        label = int(label)

        input_ids = self.text2index(text)

    

    def __len__(self):
        return len(self.data)



def main():
    data_dir = "./data"
    dataset, word2indx = prepare_dataset(data_dir)

    train_data = Reviewdataset(dataset["train"], word2indx)
    


if __name__ == '__main__':
    main()
    
