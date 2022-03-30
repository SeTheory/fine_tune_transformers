from data_processor import DataProcessor
from model import ChineseBERT
import pandas as pd

path = ''
dataProcessor = DataProcessor(0.7)
tokenizer = dataProcessor.tokenizer
model = ChineseBERT(2).load_model(path)

df = pd.read_csv('')