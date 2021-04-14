import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
MAX_LENGTH = 15  # number of sentences
MAX_SENTENCE_LENGTH = 192  # number of words in one sentence


class CNNClassifier(nn.Module):
    def __init__(self, n_classes, out_channels=100, filter_width=None):
        super(CNNClassifier, self).__init__()
        if filter_width is None:
            filter_width = [2, 3, 4]
        else:
            assert len(filter_width) == 3
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.conv1 = nn.Conv2d(1, out_channels, kernel_size=(filter_width[0], self.bert.config.hidden_size))
        self.conv2 = nn.Conv2d(1, out_channels, kernel_size=(filter_width[1], self.bert.config.hidden_size))
        self.conv3 = nn.Conv2d(1, out_channels, kernel_size=(filter_width[2], self.bert.config.hidden_size))
        self.out = nn.Linear(3 * out_channels, n_classes)

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.view((-1, MAX_SENTENCE_LENGTH))
        attention_mask = attention_mask.view((-1, MAX_SENTENCE_LENGTH))
        x = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = x['pooler_output']
        output = self.drop(pooled_output)
        output = output.view((-1, MAX_LENGTH, self.bert.config.hidden_size))
        output.unsqueeze_(1)
        conv1 = self.conv1(output).squeeze_(dim=-1)
        conv2 = self.conv2(output).squeeze_(dim=-1)
        conv3 = self.conv3(output).squeeze_(dim=-1)

        pooled1 = F.max_pool1d(F.relu(conv1), conv1.shape[-1])
        pooled2 = F.max_pool1d(F.relu(conv2), conv2.shape[-1])
        pooled3 = F.max_pool1d(F.relu(conv3), conv3.shape[-1])

        cat = self.drop(torch.cat((pooled1, pooled2, pooled3), dim=1)).squeeze(dim=-1)

        return self.out(cat)


class RNNClassifier(nn.Module):
    def __init__(self, n_classes, hidden_dim=32):
        super(RNNClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim, bidirectional=False, batch_first=True)
        self.out = nn.Linear(hidden_dim, n_classes)

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        input_ids = input_ids.reshape((-1, MAX_SENTENCE_LENGTH))
        attention_mask = attention_mask.reshape((-1, MAX_SENTENCE_LENGTH))
        x = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = x['pooler_output']
        output = self.drop(pooled_output)
        output = output.reshape((batch_size, -1, self.bert.config.hidden_size))
        output, _ = self.lstm(output)
        output = self.out(output[:, -1, :])
        return output
