import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vectors
from dataloader import Dataloader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import numpy as np

from tqdm import tqdm
# import multiprocessing

# Define hyperparameters
INPUT_DIM = 33643  # Replace with the actual size of your vocabulary
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 12  # Replace with the actual number of POS tags in your dataset
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, tags = self.data[idx]
        return torch.tensor(sentence), torch.tensor(tags)

# Custom collate function for padding sequences in batches
def custom_collate(batch):
    texts, tags = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True)
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=12)
    return padded_texts, padded_tags

# Define the RNN model
class RNNTagger(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        output, _ = self.rnn(embedded)
        predictions = self.fc(output)
        return predictions


# Training loop
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in tqdm(iterator):
        texts, tags = batch
        texts, tags = texts.to(device), tags.to(device)
        optimizer.zero_grad()
        predictions = model(texts)
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        loss = criterion(predictions, tags)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Validation loop
def evaluate(model, iterator, criterion):

    conf_matrix = np.zeros((13, 13), dtype=np.int32)

    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            texts, tags = batch
            texts, tags = texts.to(device), tags.to(device)
            predictions = model(texts)
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            loss = criterion(predictions, tags)
            epoch_loss += loss.item()

            predictions = predictions.argmax(dim=-1).cpu().numpy()
            tags = tags.cpu().numpy()

            for i in range(len(predictions)):
                conf_matrix[tags[i], predictions[i]] += 1
    
    return epoch_loss / len(iterator), conf_matrix


def worker(data, epochs=10):
    """
    Train and evaluate an RNN model for sequence tagging.

    Args:
        data (list): The input data for training and evaluation.
        epochs (int, optional): The number of training epochs. Defaults to 10.

    Returns:
        None
    """
    train_data, valid_data, all_tags = vectorize(data)
    # Create instances of the custom dataset
    train_dataset = CustomDataset(train_data)
    valid_dataset = CustomDataset(valid_data)

    # Instantiate the model
    model = RNNTagger(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM + 1, N_LAYERS, BIDIRECTIONAL, DROPOUT)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Move the model and criterion to the device
    model = model.to(device)
    criterion = criterion.to(device)

    # Define DataLoaders
    BATCH_SIZE = 64  # Adjust based on your dataset and available memory
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate)

    # Training the model
    N_EPOCHS = epochs
    conf_matrix = [[]]
    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion)
        valid_loss, conf_matrix = evaluate(model, valid_loader, criterion)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Val. Loss: {valid_loss:.3f}')

    conf_matrix[12][12] = 0
    accuracy = np.sum([conf_matrix[i][i] for i in range(12)]) / np.sum(conf_matrix)
    print(f'Accuracy: {accuracy:.3f}')

    ConfusionMatrixDisplay(conf_matrix).plot()
    plt.show()

def vectorize(data):
    train, test = data
    train_data, valid_data = [], []

    index = 0
    all_tags = {}
    for d in train:
        for tag in d.tags[2:]:
            if tag not in all_tags:
                all_tags[tag] = index
                index += 1

        if len(all_tags.keys()) == 12:
            break

    all_tokens = {}
    index = 0
    for d in train:
        for token in d.tokens[2:]:
            if token not in all_tokens:
                all_tokens[token] = index
                index += 1
    
    for d in test:
        for token in d.tokens[2:]:
            if token not in all_tokens:
                all_tokens[token] = index
                index += 1
    
    # print(len(all_tags), len(all_tokens))
    for d in train:
        sentence = [all_tokens[token] for token in d.tokens[2:]]
        tags = [all_tags[tag] for tag in d.tags[2:]]
        train_data.append((sentence, tags))
    
    for d in test:
        sentence = [all_tokens[token] for token in d.tokens[2:]]
        tags = [all_tags[tag] for tag in d.tags[2:]]
        valid_data.append((sentence, tags))

    return train_data, valid_data, all_tags


if __name__ == '__main__':
    # Sample data (replace this with your own dataset loading logic)

    data = Dataloader()
    data.load("Brown_train.txt", 80)

    for data in data.n_fold(5):
        worker(data)
        break
