import torch 
from model import LSTMclassifier
import pickle

def load_cp(ckp_pth):
    word2indx = pickle.load(open('data/word2indx.pkl', "rb"))
    model = LSTMclassifier(len(word2indx), 300, 128, 2, 2)
    model.load_state_dict(torch.load(ckp_pth))
    return model