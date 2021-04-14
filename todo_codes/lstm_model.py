import re
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
from torch import nn
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

BATCH_SIZE = 8
# pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
pattern = r'\.|\?|!'


# plot setting
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

# random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# device
device = torch.device("cuda:1,2,3" if torch.cuda.is_available() else "cpu")

# load data
# df = pd.read_csv("data/test.csv")
print("Loading Data......")
df = pd.read_csv("data/arxiv.cs.ai_2007-2017.csv")
class_names = ['reject', 'accept']

# show the distribution of data
sns.countplot(x="Label", data=df)
plt.xlabel('review score')
plt.show()
plt.savefig("data_distribution.png")

df_temp = pd.DataFrame(columns=('Abstract', 'Label'))


count = 0
for index, row in df.iterrows():
    if row['Label'] == 1:
        df_temp = df_temp.append(row, ignore_index=True)
    elif count < 500:
        df_temp = df_temp.append(row, ignore_index=True)
        count += 1

sns.countplot(x="Label", data=df_temp)
plt.xlabel('review score')
plt.show()
plt.savefig("data_distribution.png")


df = df_temp

# pretrained bert model
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

# load a pre-trained [BertTokenizer]
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# data processing ---- get the max length
token_lens = []
sequence_lens = []
for i, txt in enumerate(df.Abstract):
    # result_list = txt.split(".")
    result_list = re.split(pattern, txt)
    for r in result_list:
        tokens = tokenizer.encode(r, truncation=True, max_length=1024)
        token_lens.append(len(tokens))
    sequence_lens.append(len(result_list))
    df.loc[i, 'Abstract'] = result_list

# plot the distribution
print(max(token_lens))
sns.displot(token_lens)
plt.xlim([0, 140])
plt.xlabel('Token count')
plt.show()
plt.savefig("data_length_distribution.png")

print(max(sequence_lens))
sns.displot(sequence_lens)
plt.xlim([0, 30])
plt.xlabel('Sequence count')
plt.show()
plt.savefig("sequence_length_distribution.png")

# set the max length
MAX_LEN = 186

# data processing
df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
print("The size of training data, validation data and test data:")
print(df_train.shape, df_val.shape, df_test.shape)


# create a PyTorch dataset
class GenDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = self.reviews[item]
        target = self.targets[item]
        for i, r in enumerate(review):
            encoding = self.tokenizer.encode_plus(r, add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False,
            padding='max_length', return_attention_mask=True, return_tensors='pt')
            if i == 0:
                e = encoding['input_ids']
                a = encoding['attention_mask']
            else:
                e = torch.cat((e, encoding['input_ids']), 0)
                a = torch.cat((a, encoding['attention_mask']), 0)

        while e.shape[0] < 30:
            temp = torch.zeros((1, self.max_len), dtype=torch.long)
            e = torch.cat((e, temp), 0)
            a = torch.cat((a, temp), 0)
        # print("ok")
        return {
            'review_text': review,
            'input_ids': e,
            'attention_mask': a,
            'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GenDataset(
    reviews=df.Abstract.to_numpy(),
    targets=df.Label.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len)

    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0)


train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)


# TODO add lstm layer
# creat a classifier using bert model
class RateClassifier(nn.Module):
    def __init__(self, n_classes):
        super(RateClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.lstm = nn.LSTM(MAX_LEN, 2, bidirectional=True)

    def forward(self, input_ids, attention_mask):
        # TODO input_ids:batch_size, 30, 186

        # batch_size = input_ids.shape[0]
        # input_ids = input_ids.permute(1, 0, 2).float()
        # input_ids, _ = self.lstm(input_ids)
        # input_ids = input_ids.reshape(batch_size, -1).long()
        # attention_mask = attention_mask.permute(1, 0, 2).float()
        # attention_mask, _ = self.lstm(attention_mask)
        # attention_mask = attention_mask.reshape(batch_size, -1).long()

        x = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = x['pooler_output']
        output = self.drop(pooled_output)
        return self.out(output)


print("Build model......")
model = RateClassifier(len(class_names))
model = model.to(device)

# training setting
EPOCHS = 10
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)


# training process
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler,n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

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


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


# training
print("Start training......")
history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))
    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(df_val))

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'bert_origin.bin')
        best_accuracy = val_acc

# plot training vs validation accuracy
plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')

plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1])
plt.show()
plt.savefig("data_accuracy.png")

# test best model
model = RateClassifier(len(class_names))
model.load_state_dict(torch.load('bert_origin.bin'))
model = model.to(device)
test_acc, _ = eval_model(model, test_data_loader, loss_fn, device, len(df_test))
print("The accuracy of best model:", test_acc.item())
