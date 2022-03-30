import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import time

from bert.codes.models.modeling_glycebert import GlyceBertModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score


class ChineseBERT(nn.Module):
    def __init__(self, num_class, dropout_rate=0.3, model_path='./bert/ChineseBERT-base/', hidden_size=768):
        super(ChineseBERT, self).__init__()
        self.bert = GlyceBertModel.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.adaptive_lr = True
        self.seed = None
        self.model_name = 'ChineseBERT'
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.dn = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.hidden_size, self.num_class)

    def forward(self, contents, pinyins):
        output = self.bert(contents, pinyins)[1]
        out = self.fc(self.dn(output))
        return out

    def save_model(self, path):
        torch.save(self, path)
        print('Save successfully!')

    def load_model(self, path):
        model = torch.load(path, map_location=self.device)
        print('Load successfully!')
        return model

    def train_model(self, dataloader, epoch, criterion, optimizer, epoch_log):
        self.train()
        total_acc, total_count = 0, 0
        log_interval = 10
        loss_list = []
        start_time = time.time()
        all_predicted_result = []
        all_true_label = []

        for idx, (contents, pinyins, labels) in enumerate(dataloader):
            contents = contents.to(self.device)
            pinyins = pinyins.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            predicted_result = self(contents, pinyins)
            loss = criterion(predicted_result, labels.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5)  # 梯度裁剪，防止梯度消失或爆炸
            optimizer.step()

            all_predicted_result += F.softmax(predicted_result, dim=1).detach().cpu().numpy().tolist()
            all_true_label += labels.cpu().numpy().tolist()
            total_acc += (predicted_result.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
            loss_list.append(loss.item())
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| accuracy {:8.3f} | loss {:8.3f}'.format(epoch, idx, len(dataloader),
                                                                 total_acc / total_count, loss.item()))
                epoch_log.write('| epoch {:3d} | {:5d}/{:5d} batches '
                                '| accuracy {:8.3f} | loss {:8.3f}\n'.format(epoch, idx, len(dataloader),
                                                                             total_acc / total_count, loss.item()))
                total_acc, total_count = 0, 0
                start_time = time.time()
            # print('val-time:', time.time() - start_time)

        all_predicted_result = np.array(all_predicted_result)
        all_predicted_label = all_predicted_result.argmax(1)

        acc = accuracy_score(all_true_label, all_predicted_label)
        prec = precision_score(all_true_label, all_predicted_label)
        recall = recall_score(all_true_label, all_predicted_label)
        f1 = f1_score(all_true_label, all_predicted_label, average='binary')
        maf1 = f1_score(all_true_label, all_predicted_label, average='macro')
        # mif1 = f1_score(all_true_label, all_predicted_label, average='micro')s
        auc = roc_auc_score(all_true_label, all_predicted_result[:, 1])
        log_loss_value = log_loss(all_true_label, all_predicted_result)
        avg_loss = np.mean(loss_list)

        print(
            '-' * 59 + '\n' +
            '| average loss {:4.3f} | train accuracy {:8.3f} |\n'
            '| precision {:8.3f} | recall {:10.3f} |\n'
            '| macro-f1 {:9.3f} | normal-f1 {:7.3f} |\n'
            '| auc {:14.3f} | log_loss {:8.3f} |\n'.format(avg_loss, acc, prec, recall, maf1, f1, auc, log_loss_value)
        )
        epoch_log.write(
            '-' * 59 + '\n' +
            '| average loss {:4.3f} | train accuracy {:8.3f} |\n'
            '| precision {:8.3f} | recall {:10.3f} |\n'
            '| macro-f1 {:9.3f} | normal-f1 {:7.3f} |\n'
            '| auc {:14.3f} | log_loss {:8.3f} |\n'.format(avg_loss, acc, prec, recall, maf1, f1, auc, log_loss_value)
        )

        return [avg_loss, acc, prec, recall, maf1, f1, auc, log_loss_value]

    def train_epoch(self, dataloaders, epochs, lr=10e-4, criterion='CrossEntropyLoss', optimizer='ADAMW',
                    record_path=None, save_path=None):
        """
        整体多次训练模型，先更新参数再用验证集测试
        输入主要包括训练集和验证集，训练轮次，损失函数，优化方法，自动调整学习率方法
        """
        final_results = []
        train_dataloader, val_dataloader, test_dataloader = dataloaders
        total_accu = None
        val_accu_list = []
        # 损失函数设定
        if criterion == 'CrossEntropyLoss':
            criterion = torch.nn.CrossEntropyLoss()
        # 优化器设定
        if optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=1e-3)
        elif optimizer == 'ADAM':
            if self.adaptive_lr:
                optimizer = torch.optim.Adam(self.get_group_parameters(lr), lr=lr, weight_decay=0)
            else:
                optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-3)
        elif optimizer == 'SPARSE_ADAM':
            optimizer = torch.optim.SparseAdam(self.parameters(), lr=lr)
        elif optimizer == 'ADAMW':
            if self.adaptive_lr:
                optimizer = torch.optim.AdamW(self.get_group_parameters(lr), lr=lr)
            else:
                optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        if self.seed:
            print('{}_records_{}.csv'.format(self.model_name, self.seed))
            fw = open(record_path + '{}_records_{}.csv'.format(self.model_name, self.seed), 'w')
        else:
            fw = open(record_path + '{}_records.csv'.format(self.model_name), 'w')
        fw.write('epoch,'
                 'loss,accu_train,prec_train,recall_train,maf1_train,f1_train,auc_train,log_loss_train,'
                 'accu_val,prec_val,recall_val,maf1_val,f1_val,auc_val,log_loss_val,'
                 'accu_test,prec_test,recall_test,maf1_test,f1_test,auc_test,log_loss_test\n')
        for epoch in range(1, epochs + 1):
            epoch_log = open(record_path + '{}_epoch_{}.log'.format(self.model_name, epoch), 'w')
            epoch_start_time = time.time()
            # train(model, train_dataloader)
            train_results = self.train_model(train_dataloader, epoch, criterion, optimizer, epoch_log)
            val_results = self.evaluate(val_dataloader)
            acc, prec, recall, maf1, f1, auc, log_loss_value = val_results
            val_accu_list.append(round(acc, 3))
            if save_path:
                self.save_model(save_path + '{}_{}.pkl'.format(self.model_name, epoch))
            # print('  '.join([str(accu_val), str(prec_val), str(recall_val), str(maf1_val), str(mif1_val)]))
            # fw.write(', '.join([str(epoch), str(round(float(loss), 3)), str(round(acc, 3)), str(round(prec, 3)),
            #                     str(round(recall, 3)), str(round(maf1, 3)), str(round(mif1, 3))]) + '\n')
            if (len(set(val_accu_list[-5:])) == 1) & (len(val_accu_list) >= 8) & (prec != 0) & (recall != 0):
                fw.close()
                break
            else:
                total_accu = acc
            print('-' * 59)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid accuracy {:8.3f} |\n'
                  '| precision {:8.3f} | recall {:10.3f} |\n'
                  '| macro-f1 {:9.3f} | normal-f1 {:7.3f} |\n'
                  '| auc {:14.3f} | log_loss {:8.3f} |\n'.format(epoch,
                                                                 time.time() - epoch_start_time,
                                                                 acc, prec, recall, maf1, f1, auc, log_loss_value))
            epoch_log.write(
                '-' * 59 + '\n' +
                '| end of epoch {:3d} | time: {:5.2f}s | valid accuracy {:8.3f} |\n'
                '| precision {:8.3f} | recall {:10.3f} |\n'
                '| macro-f1 {:9.3f} | normal-f1 {:8.3f} |\n'
                '| auc {:14.3f} | log_loss {:8.3f} |\n'.format(epoch,
                                                               time.time() - epoch_start_time,
                                                               acc, prec, recall, maf1, f1, auc, log_loss_value)
            )
            print('-' * 59)

            test_results = self.test(test_dataloader, epoch_log=epoch_log)
            epoch_log.close()
            all_results = ['{:3d}'.format(epoch)] + ['{:.3f}'.format(val) for val in train_results] + \
                          ['{:.3f}'.format(val) for val in val_results] + ['{:.3f}'.format(val) for val in test_results]
            final_results.append(all_results)
            fw.write(','.join(all_results) + '\n')
        fw.close()
        return final_results

    def evaluate(self, dataloader, phase='train'):
        self.eval()
        all_predicted_result = []
        all_true_label = []

        with torch.no_grad():
            for idx, (contents, pinyins, labels) in enumerate(dataloader):
                contents = contents.to(self.device)
                pinyins = pinyins.to(self.device)
                labels = labels.to(self.device)

                predicted_result = F.softmax(self(contents, pinyins), dim=1).detach().cpu().numpy()
                true_label = labels.cpu().numpy().tolist()

                all_predicted_result += predicted_result.tolist()
                all_true_label += true_label

            all_predicted_result = np.array(all_predicted_result)
            all_predicted_label = all_predicted_result.argmax(1)

        if phase == 'benchmark':
            print(all_predicted_label)
            print(all_true_label)
        accuracy = accuracy_score(all_true_label, all_predicted_label)
        precision = precision_score(all_true_label, all_predicted_label)
        recall = recall_score(all_true_label, all_predicted_label)
        f1 = f1_score(all_true_label, all_predicted_label, average='binary')
        maf1 = f1_score(all_true_label, all_predicted_label, average='macro')
        # mif1 = f1_score(all_true_label, all_predicted_label, average='micro')
        auc = roc_auc_score(all_true_label, all_predicted_result[:, 1])
        log_loss_value = log_loss(all_true_label, all_predicted_result)
        return [accuracy, precision, recall, maf1, f1, auc, log_loss_value]

    def test(self, test_dataloader, phase='test', epoch_log=None):
        print('-' * 59)
        test_start_time = time.time()
        accu, prec, recall, maf1, f1, auc, log_loss_value = self.evaluate(test_dataloader, phase)
        print('| end of test | time: {:5.2f}s | test accuracy {:8.3f} |\n'
              '| precision {:8.3f} | recall {:10.3f} |\n'
              '| macro-f1 {:9.3f} | normal-f1 {:7.3f} |\n'
              '| auc {:14.3f} | log_loss {:8.3f} |\n'.format(
            time.time() - test_start_time,
            accu, prec, recall, maf1, f1, auc, log_loss_value))
        print('-' * 59)
        if epoch_log:
            epoch_log.write(
                '-' * 59 + '\n' +
                '| end of test | time: {:5.2f}s | test accuracy {:8.3f} |\n'
                '| precision {:8.3f} | recall {:10.3f} |\n'
                '| macro-f1 {:9.3f} | normal-f1 {:7.3f} |\n'
                '| auc {:14.3f} | log_loss {:8.3f} |\n'.format(
                    time.time() - test_start_time,
                    accu, prec, recall, maf1, f1, auc, log_loss_value)
                + '-' * 59
            )
        return [accu, prec, recall, maf1, f1, auc, log_loss_value]

    def get_group_parameters(self, lr):
        # 不分别设置weight_decay
        print('use adaptive lr')
        print('不分别设置weight_decay')
        bert_params = list(map(id, self.bert.parameters()))
        bert_params_list = [(n, p) for n, p in self.bert.named_parameters()]
        # for n, p in self.named_parameters():
        #     print(n)

        bert_dict = {}
        params_list = []
        # bert参数
        bert_dict[-1] = list(filter(lambda value: 'embeddings.' in value[0], bert_params_list))
        for i in range(12):
            bert_dict[i] = list(filter(lambda value: '.' + str(i) + '.' in value[0], bert_params_list))
        bert_dict[11].extend(list(filter(lambda value: 'pooler.' in value[0], bert_params_list)))

        for i in range(-1, 12):
            gamma = 0.95 ** (11 - i)
            # print(i, gamma)
            current_value = bert_dict[i]
            current_list = [
                {
                    'params': [value[1] for value in current_value],
                    'lr': lr * gamma
                }
            ]
            params_list.extend(current_list)

        # 除bert外参数
        normal_list = filter(lambda p: id(p) not in bert_params, self.parameters())
        params_list.extend([
            {
                'params': [value for value in normal_list],
                'lr': lr * 10
            }
        ])

        return params_list
