import torch

class Reviewdataset(torch.utils.data.Dataset):
    def __init__(self, dataset, word2indx, max_seq_length):
        self.data = dataset
        self.word2indx = word2indx
        self.max_seq_length = max_seq_length

    def text2index(self, x):
        unk_id = self.word2indx["<UNK>"]
        sos_id = self.word2indx["<SOS>"]
        eos_id = self.word2indx["<EOS>"]
        pad_id = self.word2indx["<PAD>"]
        input_ids, text = [sos_id], ["<SOS>"]
        for word in x.split():
            token_id = self.word2indx.get(word, unk_id)
            input_ids.append(token_id)
            if token_id == unk_id:
                text.append(word)
            else:
                text.append(word)
            

        if len(input_ids)>= self.max_seq_length:
            text = text[:self.max_seq_length]
            text.append("<EOS>")
            input_ids = input_ids[:self.max_seq_length]
            input_ids.append(eos_id)
            text_len = len(input_ids)
        else:
            to_add = self.max_seq_length - len(input_ids)
            text.append("<EOS>")
            input_ids.append(eos_id)
            text_len = len(input_ids)

            text.extend(["PAD"]*to_add)
            input_ids.extend([pad_id]*to_add)
        return input_ids, " ".join(text), text_len


    def __getitem__(self, index):
        sample = self.data[index]
        text , label = sample.split('\t')
        label = int(label)

        input_ids, _text, text_length = self.text2index(text)

        return {
            "text": _text,
            "input_ids": torch.tensor(input_ids),
            "label": torch.tensor(label),
            "length": text_length,
        }
    

    def __len__(self):
        return len(self.data)

def make_dataloader(data, word2indx, max_seq_length, batch_size):

    data_set = Reviewdataset(data, word2indx, max_seq_length)
    return torch.utils.data.DataLoader(data_set, batch_size)