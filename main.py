from model import ChineseBERT
from data_processor import DataProcessor
import torch

def main():
    dataProcessor = DataProcessor(0.7)
    dataProcessor.extract_data()
    dataProcessor.split_data()
    dataProcessor.get_dataloader(32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChineseBERT(2).to(device)
    final_results = model.train_epoch(dataProcessor.dataloaders, 5, 2e-5,
                                      record_path='./checkpoints/results/',
                                      save_path='./checkpoints/')


if __name__ == '__main__':
    main()
