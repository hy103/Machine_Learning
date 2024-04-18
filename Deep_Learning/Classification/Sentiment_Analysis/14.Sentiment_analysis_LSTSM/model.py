import torch

class LSTMclassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layer, n_classes):
        super(LSTMclassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        #self.hidden_dim = hidden_dim
        self.embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, n_layer)
        self.hidden= torch.nn.Linear(hidden_dim, n_classes)


    def forward(self, x):
        embeddings = self.embedding_layer(x["input_ids"])
        output, _ = self.lstm(embeddings)
        output = output[:, -1]
        output= self.hidden(output)
        return output