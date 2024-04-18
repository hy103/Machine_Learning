import os, re, unicodedata
import numpy as np

def unicode_to_ascii(x):
     return "".join(
          c for c in unicodedata.normalize("NFD", x) if unicodedata.category(c)!= "Mn"
     )
     
def normalize_text(lines):
     x = unicode_to_ascii(lines.lower().strip())
     x = re.sub(r"([.!?])", r"\1", x)
     x = re.sub(r"[^a-zA-Z.!?]+", r" ", x)
     #print(x)
     return x

def reading_file(file_path):
    with open(file_path, 'r') as f:
        text = f.readlines()
        return text

def create_vocab(dataset, min_freq):
    wordcount, word2index ={}, {}
    word2index = {'<SOS>' :0, '<EOS>' : 1, '<PAD>' : 2, '<UNK>': 3}
    for line in dataset:
         sample = line.split("\t")[0]
         for word in sample.split():
            if word not in wordcount:
                   wordcount[word] =1
            else :
                 wordcount[word] +=1
                   
    for word, count in wordcount.items():
         if wordcount[word]>min_freq:
              word2index[word] = len(word2index)
    return word2index
                   

def prepare_dataset(data_dir):
    dataset = []
    pos_file_path = os.path.join(data_dir, "TrainingDataPositive.txt")
    pos_lines = reading_file(pos_file_path)   
    for line in pos_lines:
        line = normalize_text(line)
        sample = f'{line}\t1'
        dataset.append(sample)
    neg_file_path = os.path.join(data_dir, "TrainingDataNegative.txt")
    neg_lines = reading_file(neg_file_path)
    for line in neg_lines:
            line = normalize_text(line)
            sample = f'{line}\t0'
            dataset.append(sample)

    np.random.shuffle(dataset)
    word2index = create_vocab(dataset, 10)
    n_train = int(len(dataset)*0.8)
    n_val = int(len(dataset)*0.1)

    dataset = {'train': dataset[:n_train],
               'val' : dataset[n_train :n_train+n_val],
               'test': dataset[-n_val:]}
    
    return dataset, word2index
