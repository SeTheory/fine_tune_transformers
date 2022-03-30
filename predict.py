import torch

from data_processor import DataProcessor
from model import ChineseBERT
import pandas as pd

path = './checkpoints/ChineseBERT_3.pkl'
dataProcessor = DataProcessor(0.7)
tokenizer = dataProcessor.tokenizer

model = ChineseBERT(2).load_model(path)
model.eval()

df = pd.read_csv('./data/predicted_data.csv')
contents = df['评论内容'].to_list()
contents = [dataProcessor.get_dealt_text(content, 256) for content in contents]
with torch.no_grad():
    results = [model(content[0], content[1]).numpy().tolist() for content in contents]
df['result'] = results
df.to_csv('predicted_result.csv')