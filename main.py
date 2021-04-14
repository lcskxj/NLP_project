from collections import defaultdict
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
from utils.preprocessing import trim, get_datasets, create_data_loader
from model import CNNClassifier, RNNClassifier
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

BATCH_SIZE = 4

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print("Using GPU on {}".format(DEVICE))
else:
    DEVICE = torch.device("cpu")
    print("Using CPU only.")


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(trained_model, data_loader, loss_fn, device, n_examples):
    trained_model = trained_model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = trained_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def train(total_epoch=10):

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * total_epoch

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(total_epoch):

        print(f'Epoch {epoch + 1}/{total_epoch}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            DEVICE,
            scheduler,
            len(df_train)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            DEVICE,
            len(df_val)
        )

        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')

    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv("./data/balance_data.csv")
    trim(df)
    df_train, df_val, df_test = get_datasets(df)
    train_data_loader = create_data_loader(df_train, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, BATCH_SIZE)
    # model = CNNClassifier(2)
    model = RNNClassifier(2)
    model.to(DEVICE)
    train()
