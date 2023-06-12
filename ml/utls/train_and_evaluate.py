import torch
import torch.nn as nn
from torch.nn.modules import loss
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import accuracy_score, precision_score, recall_score


def train_and_evaluate_model(
    model: nn.Module, 
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss_fn: loss._Loss, 
    optimizer: optim.Optimizer, 
    epochs: int = 10
) -> None:

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Evaluation phase
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                outputs = model(inputs).squeeze()
                y_true.extend(labels.tolist())
                y_pred.extend((outputs > 0.5).long().tolist())

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        print(f'Epoch: {epoch+1}/{epochs}, Loss: {running_loss / len(train_dataloader)}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')
