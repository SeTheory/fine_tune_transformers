import random

import pandas as pd
import torchtext
from torch.utils.data import DataLoader
from torchtext import datasets
from transformers import AutoTokenizer, BertTokenizer, BertModel
import torch
import re

from bert.codes.datasets.bert_dataset import BertDataset
from bert.codes.models.modeling_glycebert import GlyceBertModel


class DataProcessor:
    def __init__(self, rate, pad_size=256, seed=123):
        self.rate = rate
        self.seed = int(seed)
        self.tokenizer = BertDataset('./bert/ChineseBERT-base/')
        self.pad_size = pad_size

    def extract_data(self):
        df = pd.read_csv('./data/dessert.csv')
        contents = list(map(lambda x: ' '.join(re.split(r'[\s]+', x.strip())).strip() + '\n', df['comment'].to_list()))
        labels = list(map(lambda x: str(x) + '\n', df['label'].to_list()))
        print(pd.DataFrame([len(content) for content in contents]).describe(percentiles=[0.8, 0.9, 0.93, 0.95, 0.98]))
        print(contents[:5])
        with open('./data/contents.list', 'w') as fw:
            fw.writelines(contents)
        with open('./data/labels.list', 'w') as fw:
            fw.writelines(labels)

    def split_data(self):
        data_result = {}
        with open('./data/contents.list', 'r') as fr:
            contents = list(map(lambda x: x.strip(), fr.readlines()))
        with open('./data/labels.list', 'r') as fr:
            labels = list(map(lambda x: x.strip(), fr.readlines()))

        all_data = list(zip(contents, labels))
        random.seed(self.seed)
        random.shuffle(all_data)
        contents, labels = zip(*all_data)

        data_count = len(contents)

        data_result['train'] = [
            contents[:int(self.rate * data_count)],
            labels[:int(self.rate * data_count)]
        ]
        data_result['val'] = [
            contents[int(self.rate * data_count): int((1 + self.rate) / 2 * data_count)],
            labels[:int(self.rate * data_count): int((1 + self.rate) / 2 * data_count)]
        ]
        data_result['test'] = [
            contents[int((1 + self.rate) / 2 * data_count):],
            labels[int((1 + self.rate) / 2 * data_count):]
        ]
        torch.save(data_result, './data/dealt_data_{}'.format(self.seed))

    def load_data(self):
        all_data = torch.load('./data/dealt_data_{}'.format(self.seed))
        train_contents, train_labels = all_data['train']
        val_contents, val_labels = all_data['val']
        test_contents, test_labels = all_data['test']
        return train_contents, train_labels, \
               val_contents, val_labels, \
               test_contents, test_labels

    def get_dataloader(self, batch_size):
        train_contents, train_labels, \
        val_contents, val_labels, \
        test_contents, test_labels = self.load_data()
        # self.tokenizer = BertDataset('./bert/ChineseBERT-base/')
        self.text_pipeline = lambda x: self.get_dealt_text(x, self.pad_size)

        train_dataloader = DataLoader(list(zip(train_contents, train_labels)), batch_size=batch_size,
                                      shuffle=True, collate_fn=self.collate_batch)
        val_dataloader = DataLoader(list(zip(val_contents, val_labels)), batch_size=batch_size,
                                    shuffle=True, collate_fn=self.collate_batch)
        test_dataloader = DataLoader(list(zip(test_contents, test_labels)), batch_size=batch_size,
                                     shuffle=True, collate_fn=self.collate_batch)

        self.dataloaders = [train_dataloader, val_dataloader, test_dataloader]

        return train_dataloader, val_dataloader, test_dataloader

    def collate_batch(self, batch):
        content_list, pinyin_list, label_list = [], [], []
        for (_content, _label) in batch:
            label_list.append(int(_label.strip()))
            processed_content, processed_pinyin = self.text_pipeline(_content)
            content_list.append(processed_content)
            pinyin_list.append(processed_pinyin)
        return torch.cat(content_list, dim=0).long(), torch.cat(pinyin_list, dim=0).long(), torch.tensor(label_list, dtype=torch.int8)

    def get_dealt_text(self, text, pad_seq):
        tokens, pinyins = self.tokenizer.tokenize_sentence(text[:500])
        pinyins = pinyins.view(len(tokens), 8)
        if len(tokens) <= pad_seq:
            tokens = torch.cat((tokens, torch.tensor([0]*(pad_seq - len(tokens)), dtype=torch.long)))
            pinyins = torch.cat((pinyins, torch.tensor([[0]*8]*(pad_seq - len(pinyins)), dtype=torch.long)), dim=0)
        else:
            tokens = torch.cat((tokens[:pad_seq-1], torch.tensor([102])))
            pinyins = torch.cat((pinyins[:pad_seq-1], torch.tensor([[0]*8])), dim=0)
        return tokens.unsqueeze(dim=0), pinyins.unsqueeze(dim=0)




if __name__ == '__main__':
    dataProcessor = DataProcessor(0.7, 256)
    # dataProcessor.extract_data()
    # dataProcessor.split_data()
    # dataProcessor.get_dataloader(32)
    # test = dataProcessor.dataloaders[2]
    # for idx, (content, pinyin, label) in enumerate(test):
    #     if idx == 1:
    #         print(content.shape)
    #         print(pinyin.shape)
    #         print(label.shape)

    # tokenizer = BertDataset('./bert/ChineseBERT-base/')
    # chinese_bert = GlyceBertModel.from_pretrained('./bert/ChineseBERT-base/')
    #
    # input_ids, pinyin_ids = tokenizer.tokenize_sentence('我喜欢猫[PAD][PAD]')
    # length = input_ids.shape[0]
    # input_ids = input_ids.view(1, length)
    # pinyin_ids = pinyin_ids.view(1, length, 8)
    # output_hidden = chinese_bert.forward(input_ids, pinyin_ids)
    # print(output_hidden[0].shape)
    # print(output_hidden[1].shape)

    string = '我爱学习'
    print(dataProcessor.get_dealt_text(string, 256))
