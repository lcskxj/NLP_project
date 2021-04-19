import re

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from text_scoring import get_score

MAX_LENGTH = 15
MAX_TOKEN_LENGTH = 512
MAX_SENTENCE_LENGTH = 192

RANDOM_SEED = 689009

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


def trim(df):
    for i, txt in enumerate(df.Abstract):
        tokens = tokenizer.encode(txt, truncation=True, max_length=1024)
        if len(tokens) > MAX_TOKEN_LENGTH:
            df = df.drop(i)


def split(df):
    splited_sentences = []
    for txt in df.Abstract:
        parts = re.split(r'[\.?!]\s', txt)
        parts = [x for x in parts if len(x) > 0]
        splited_sentences.append(parts[:min(MAX_LENGTH, len(parts))])
    return splited_sentences


def get_datasets(df):
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
    return df_train, df_val, df_test


class AbstractReviewDataset(Dataset):
    def __init__(self, abstracts, targets, scores):
        self.abstract = abstracts
        self.targets = targets
        self.tokenizer = tokenizer
        self.scores = scores

    def __len__(self):
        return len(self.abstract)

    def __getitem__(self, item):
        abstract = self.abstract[item]
        target = self.targets[item]
        score = self.scores[item]
        input_ids = []
        attention_masks = []
        for txt in abstract:
            encoding = self.tokenizer.encode_plus(
                txt,
                add_special_tokens=True,
                max_length=MAX_SENTENCE_LENGTH,
                return_token_type_ids=False,
                padding='max_length',
                # pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )
            input_ids.append(encoding['input_ids'])
            attention_masks.append(encoding['attention_mask'])
        offset = MAX_LENGTH - len(input_ids)
        for i in range(offset):
            input_ids.append(torch.zeros(input_ids[0].shape, dtype=input_ids[0].dtype))
            attention_masks.append(torch.zeros(attention_masks[0].shape, dtype=attention_masks[0].dtype))

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return {
            'score': torch.tensor(score),
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(df, batch_size):
    splited = split(df)
    scores = []
    for txt in df.Abstract:
        scores.append(get_score(txt))
    ds = AbstractReviewDataset(
        abstracts=splited,
        targets=df.Label.to_numpy(),
        scores=scores
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0
    )
