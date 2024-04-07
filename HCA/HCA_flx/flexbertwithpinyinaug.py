import numpy as np
import json, time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')
import os
import uuid
from pypinyin import pinyin, lazy_pinyin, Style
from Pinyin2Hanzi import DefaultHmmParams
from Pinyin2Hanzi import viterbi
from Pinyin2Hanzi import DefaultDagParams
from Pinyin2Hanzi import dag
from Pinyin2Hanzi import simplify_pinyin
import jieba
import random
BATCH_SIZE = 64 # 如果会出现OOM问题，减小它
dagparams = DefaultDagParams()
num_per_sample = 4
trainsamples = 72000
# 给定添加样本的概率
alpha = 0.9
#classnumber
classnumber = 10
dataset_path = "news_title_dataset.csv"
print("alpha is ",alpha)

class Bert_Model(nn.Module):
    def __init__(self, bert_path, classes=classnumber):
        super(Bert_Model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)  # 导入模型超参数
        self.bert = BertModel.from_pretrained(bert_path)  # 加载预训练模型权重
        self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]  # 池化后的输出 [bs, config.hidden_size]
        logit = self.fc(out_pool)  # [bs, classes]
        return logit

def pinyinaug(inputtext = None):
    title_pinyin = lazy_pinyin(inputtext, style=Style.NORMAL, errors='ignore')
    newtitlelist = []
    choice = random.random()
    #给定单个样本额外添加的样本数
    add = num_per_sample
    if choice < alpha:
        newtitlelist = [inputtext]
        return newtitlelist
    elif choice >= alpha:
        dagresult = dag(dagparams, title_pinyin, path_num=add)
        if len(dagresult) == 0:
            newtitlelist = [inputtext]
            return newtitlelist
        newtitlelist.append(inputtext)
        for i in range(add - 1):
            newtitlelist.append(''.join(dagresult[i].path))
        return newtitlelist

bert_path = "bert_model/"    # 该文件夹下存放三个文件（'vocab.txt', 'pytorch_model.bin', 'config.json'）
tokenizer = BertTokenizer.from_pretrained(bert_path)   # 初始化分词器
input_ids, input_masks, input_types, = [], [], []  # input char ids, segment type ids,  attention mask
labels = []  # 标签
maxlen = 30  # 取30即可覆盖99%
# allsamples = 20000
testsamples = 10000
validationsamples = 10000
# seed = 0
# torch.manual_seed(seed) # 为CPU设置随机种子
# torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
# torch.cuda.manual_seed_all(seed) # 为所有GPU设置随机种子

with open(dataset_path, encoding='utf-8') as f:
    for i, line in tqdm(enumerate(f)):
        title, y = line.strip().split('\t')
        #文本数据增强
        titlelist = pinyinaug(title)
        # print("new title is",title)
        # encode_plus会输出一个字典，分别为'input_ids', 'token_type_ids', 'attention_mask'对应的编码
        # 根据参数会短则补齐，长则切断
        if i < trainsamples:
            for title in titlelist:
                encode_dict = tokenizer.encode_plus(text=title, max_length=maxlen,
                                                    padding='max_length', truncation=True)
                input_ids.append(encode_dict['input_ids'])
                input_types.append(encode_dict['token_type_ids'])
                input_masks.append(encode_dict['attention_mask'])
                labels.append(int(y))

        if i >= trainsamples and i < trainsamples+validationsamples+testsamples:
            encode_dict = tokenizer.encode_plus(text=title, max_length=maxlen,
                                                padding='max_length', truncation=True)
            input_ids.append(encode_dict['input_ids'])
            input_types.append(encode_dict['token_type_ids'])
            input_masks.append(encode_dict['attention_mask'])
            labels.append(int(y))

        if i == trainsamples+validationsamples+testsamples:
            break

input_ids, input_types, input_masks = np.array(input_ids), np.array(input_types), np.array(input_masks)
labels = np.array(labels)
print(input_ids.shape, input_types.shape, input_masks.shape, labels.shape)

# 随机打乱索引
idxes = np.arange(input_ids.shape[0])
np.random.seed(2019)   # 固定种子
# np.random.shuffle(idxes)
print("展示索引大小和索引中的前10个", idxes.shape, idxes[:10])


# # 8:1:1 划分训练集、验证集、测试集
# input_ids_train, input_ids_valid, input_ids_test = input_ids[idxes[:80000]], input_ids[idxes[80000:90000]], input_ids[idxes[90000:]]
# input_masks_train, input_masks_valid, input_masks_test = input_masks[idxes[:80000]], input_masks[idxes[80000:90000]], input_masks[idxes[90000:]]
# input_types_train, input_types_valid, input_types_test = input_types[idxes[:80000]], input_types[idxes[80000:90000]], input_types[idxes[90000:]]
# y_train, y_valid, y_test = labels[idxes[:80000]], labels[idxes[80000:90000]], labels[idxes[90000:]]
#
# print(input_ids_train.shape, y_train.shape, input_ids_valid.shape, y_valid.shape,
#       input_ids_test.shape, y_test.shape)
trainnum = len(idxes) - testsamples- validationsamples
testnum = int(testsamples)
validnum = int(validationsamples)
#使用更少的数据做训练：
# 8:1:1 划分训练集、验证集、测试集
input_ids_train, input_ids_valid, input_ids_test = input_ids[idxes[:trainnum]], input_ids[idxes[trainnum:trainnum+testnum]], input_ids[idxes[trainnum+testnum:trainnum+testnum+validnum]]
input_masks_train, input_masks_valid, input_masks_test = input_masks[idxes[:trainnum]], input_masks[idxes[trainnum:trainnum+testnum]], input_masks[idxes[trainnum+testnum:trainnum+testnum+validnum]]
input_types_train, input_types_valid, input_types_test = input_types[idxes[:trainnum]], input_types[idxes[trainnum:trainnum+testnum]], input_types[idxes[trainnum+testnum:trainnum+testnum+validnum]]
y_train, y_valid, y_test = labels[idxes[:trainnum]], labels[idxes[trainnum:trainnum+testnum]], labels[idxes[trainnum+testnum:trainnum+testnum+validnum]]

print(input_ids_train.shape, y_train.shape, input_ids_valid.shape, y_valid.shape,
      input_ids_test.shape, y_test.shape)

# 训练集
train_data = TensorDataset(torch.LongTensor(input_ids_train),
                           torch.LongTensor(input_masks_train),
                           torch.LongTensor(input_types_train),
                           torch.LongTensor(y_train))
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

# 验证集
valid_data = TensorDataset(torch.LongTensor(input_ids_valid),
                          torch.LongTensor(input_masks_valid),
                          torch.LongTensor(input_types_valid),
                          torch.LongTensor(y_valid))
valid_sampler = SequentialSampler(valid_data)
valid_loader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)

# 测试集（是没有标签的）
test_data = TensorDataset(torch.LongTensor(input_ids_test),
                          torch.LongTensor(input_masks_test),
                          torch.LongTensor(input_types_test))
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

#实例化bert模型
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cuda'
EPOCHS = 3
model = Bert_Model(bert_path).to(DEVICE)

#定义迭代器
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4, eps=1e-4) #AdamW优化器
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader),
                                            num_training_steps=EPOCHS*len(train_loader))
#定义训练和评估函数
#评估模型性能，在验证集上
def evaluate(model, data_loader, device):
    model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for idx, (ids, att, tpe, y) in (enumerate(data_loader)):
            y_pred = model(ids.to(device), att.to(device), tpe.to(device))
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
            if y.squeeze().shape != torch.Size([]):
                val_true.extend(y.squeeze().cpu().numpy().tolist())
            else:
                val_true.append(y.squeeze().cpu().numpy())

    return accuracy_score(val_true, val_pred)  # 返回accuracy

# 测试集没有标签，需要预测提交
def predict(model, data_loader, device):
    model.eval()
    val_pred = []
    with torch.no_grad():
        for idx, (ids, att, tpe) in tqdm(enumerate(data_loader)):
            y_pred = model(ids.to(device), att.to(device), tpe.to(device))
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
    return val_pred


def train_and_eval(model, train_loader, valid_loader, optimizer, scheduler, device, epoch):
    best_acc = 0.0
    patience = 0
    criterion = nn.CrossEntropyLoss()
    for i in range(epoch):
        """训练模型"""
        start = time.time()
        model.train()
        print("***** Running training epoch {} *****".format(i + 1))
        train_loss_sum = 0.0
        for idx, (ids, att, tpe, y) in enumerate(train_loader):
            ids, att, tpe, y = ids.to(device), att.to(device), tpe.to(device), y.to(device)
            y_pred = model(ids, att, tpe)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # 学习率变化
            train_loss_sum += loss.item()
            # if True:
            if (idx + 1) % (len(train_loader) // 5) == 0:  # 只打印五次结果
                print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}".format(i + 1, idx + 1, len(train_loader), train_loss_sum / (idx + 1), time.time() - start))
                # print("Learning rate = {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
            # break
        """验证模型"""
        model.eval()
        acc = evaluate(model, valid_loader, device)  # 验证模型的性能
        ## 保存最优模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "flexbest_bert_model.pth")
        print("current acc is {:.4f}, best acc is {:.4f}".format(acc, best_acc))
        print("time costed = {}s \n".format(round(time.time() - start, 5)))


train_and_eval(model, train_loader, valid_loader, optimizer, scheduler, DEVICE, EPOCHS)
model.load_state_dict(torch.load("flexbest_bert_model.pth"))
pred_test = predict(model, test_loader, DEVICE)
print("\n Test Accuracy = {} \n".format(accuracy_score(y_test, pred_test)))
print(classification_report(y_test, pred_test, digits=4))