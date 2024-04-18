from utils import prepare_dataset
from dataloader import make_dataloader
import numpy as np
import torch
from tqdm import tqdm
import os
import pickle
from inference import LSTMclassifier

def train(dloader, model, criterion, optimizer):
    model.train()
    losses, acc = [], []
    for batch in tqdm(dloader):
        y = batch["label"]
        logits = model(batch)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds = torch.argmax(logits, -1)
        acc.append((preds == y).float().mean().item())

    print(
        f"Train Loss: {np.array(losses).mean():.4f} | Train Accuracy: {np.array(acc).mean():.4f}"
    )


@torch.no_grad()
def test(dloader, model, criterion):
    model.eval()
    losses, acc = [], []
    for batch in dloader:
        y = batch["label"]
        logits = model(batch)
        loss = criterion(logits, y)
        losses.append(loss.item())
        preds = torch.argmax(logits, -1)
        acc.append((preds == y).float().mean().item())

    print(f"Loss: {np.array(losses).mean():.4f} | Accuracy: {np.array(acc).mean():.4f}")


def save_cp(model):
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    torch.save(model.state_dict(), "checkpoints/lstm_model.pt")


def main():
    data_dir = "./data"
    dataset, word2indx = prepare_dataset(data_dir)
    with open("data/word2indx.pkl", 'wb')as f:
        pickle.dump(word2indx, f)
    max_seq_length = 30
    
    train_loader = make_dataloader(dataset["train"], word2indx, max_seq_length, 10)
    val_loader = make_dataloader(dataset["val"], word2indx, max_seq_length, 10)
    test_loader = make_dataloader(dataset["test"], word2indx, max_seq_length, 10)

    model = LSTMclassifier(len(word2indx), 100, 64, 2, 2)
    criterion = torch.nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


    for epoch in range(10):
        print(f"===Epoch {epoch}===")
        train(train_loader, model, criterion, optimizer)
        print("Validating...")
        test(val_loader, model, criterion)
        print("Testing...")
        test(test_loader, model, criterion)

        
    save_cp(model)


    


if __name__ == '__main__':
    main()
    
