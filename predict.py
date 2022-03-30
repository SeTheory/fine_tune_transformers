import torch

from data_processor import DataProcessor
from model import ChineseBERT
import pandas as pd

path = ''
dataProcessor = DataProcessor(0.7)
tokenizer = dataProcessor.tokenizer
model = ChineseBERT(2).load_model(path)
model.eval()

df = pd.read_csv('')
contents = df['review'].to_list()
contents = [dataProcessor.get_dealt_text(content, 256) for content in contents]
with torch.no_grad():
    results = [model(content[0], content[1]) for content in contents]
df['result'] = results
df.to_csv('predicted_result.csv')