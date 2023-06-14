# xzd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from tqdm import tqdm
import os
import time
from transformers import BertTokenizer
from transformers import logging

#graphic
import matplotlib.pyplot as plt
import sklearn.metrics as metrics	
from sklearn.metrics import roc_curve,auc
import numpy as np 
#graphic

import warnings

warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

# 设置transformers模块的日志等级，减少不必要的警告，对训练过程无影响
logging.set_verbosity_error()

#graphic
colors = ['blue', 'cyan', 'green', 'indigo', 'magenta', 'red', 'darkorange', 'yellow']
accs, recs, precs, f1s = [], [], [], [] #储存指标
fprs, tprs, roc_aucs =[], [], []
savepath = r"/home/xiazhiduo/pytorch/bert-sst2/xzd/ROC/"
epochs = []
threshold = 0.5 # 分类阈值
#graphic

# 超参数
gpu_num = "2"
batch_size = 16
num_epoch = 4  # 训练轮次
train_path = "/home/xiazhiduo/pytorch/bert-sst2/train.tsv"  # 数据所在地址
test_path = "/home/xiazhiduo/pytorch/bert-sst2/dev.tsv"
learning_rate = 2e-5  # 优化器的学习率
device = torch.device(("cuda:" + gpu_num) if torch.cuda.is_available() else "cpu")
pretrained_model_name = 'bert-base-uncased'

# 定义模型
class BertSST2Model(nn.Module):
    def __init__(self, num_labels, character, pretrained_name = 'bert-base-uncased'):
        super().__init__()
        self.num_labels = num_labels
        self.character = character
        self.bert = BertModel.from_pretrained(pretrained_name,
                                              output_hidden_states = True,
                                              return_dict = True)
    
        self.dropout = nn.Dropout(p = 0.1, inplace = False)
        self.classifier = nn.Linear(768, num_labels)
        
    def forward(self, inputs):
        input_ids, token_type_ids, attention_mask = inputs['input_ids'], inputs[
            'token_type_ids'], inputs['attention_mask']

        output = self.bert(input_ids = input_ids,
                           token_type_ids = token_type_ids,
                           attention_mask = attention_mask)
        
        last_hidden_state = output.last_hidden_state    
        hidden_states = output.hidden_states
        pooler_output = output.pooler_output        # CLS的pooler_output
        
        if self.character == 1:             # pooler_output
            dropped = self.dropout(pooler_output)
            logits = self.classifier(dropped)
            return logits
        
        if self.character == 2:             # last_hidden_state取平均
            dropped = self.dropout(last_hidden_state)
            logits = self.classifier(torch.mean(dropped, dim = 1))
            return logits
        if self.character == 3:             # last_hidden_state取CLS
            dropped = self.dropout(last_hidden_state)
            logits = self.classifier(last_hidden_state[:,0])
            return logits
        if self.character == 4:             # last_hidden_state取最大值
            dropped = self.dropout(last_hidden_state)
            logits = self.classifier(last_hidden_state.max(1)[0])
            return logits
        if self.character == 5:             # 取后四层平均值，层内取平均
            zero = torch.zeros(len(inputs['input_ids']), 768)
            zero = zero.to(device)
            last_four = hidden_states[9:]
            for _ in last_four:
                dropped = self.dropout(_)
                item = torch.mean(dropped, dim = 1)
                zero += item
            zero = zero / len(last_four)
            logits = self.classifier(zero)
            return logits
        if self.character == 6:             # 取后四层平均值，层内取最大
            zero = torch.zeros(len(inputs['input_ids']), 768)
            zero = zero.to(device)
            last_four = hidden_states[9:]
            for _ in last_four:
                dropped = self.dropout(_)
                item = dropped.max(1)[0]
                zero += item
            zero = zero / len(last_four)
            logits = self.classifier(zero)
            return logits

def save_pretrained(model, path):
    # 保存模型，先利用os模块创建文件夹，后利用torch.save()写入模型文件
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, 'model.pth'))

def load_sentence_polarity(data_path, train_ratio=0.8):
    all_data = []
    with open(data_path, 'r', encoding="utf8") as file:
        for sample in file.readlines():
            # polar指情感的类别，只有两种：
            #   ——0：positive
            #   ——1：negative
            # sent指对应的句子
            sent, polar = sample.strip().split("\t")
            all_data.append((sent, polar))

    return all_data

#graphic
def Modelanlysis(y_true, y_score, path, epoch, color, threshold=0.5):
    fpr,tpr, thresholds = roc_curve(y_true, y_score, pos_label=0)
    roc_auc = auc(fpr, tpr)
    # 计算相关指标
    y_pre = []
    for i in y_score:
        if i >= threshold:
            y_pre.append(0)
        elif i < threshold:
            y_pre.append(1)
    prec = metrics.precision_score(y_true, y_pre)	#精确率
    rec = metrics.recall_score(y_true, y_pre) #召回率
    f1 = metrics.f1_score(y_true, y_pre) # F1 score
    acc = metrics.accuracy_score(y_true, y_pre)	# 准确率
    return acc, rec, prec, f1, fpr, tpr, roc_auc

def myroc_curve(epochs, fprs, tprs, roc_aucs, save_path):
    plt.close()
    for a,b,c,d in zip(fprs, tprs ,roc_aucs, epochs):
        plt.plot(a, b,color=colors[d],  label='epoch={0} (area = {1:.4f})'.format(d, c), lw=1)
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate') 
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    plt.savefig(save_path+'ROC.jpg',dpi=800)

def acc_curve(epochs, accs, save_path, label):
    plt.close()
    plt.plot(epochs, accs, 'darkorange',  lw=1, marker='o', markerfacecolor='black', markersize=5)
    plt.ylim([-0.05, 1.05])
    plt.xlabel('epoch')
    plt.ylabel(label)
    my_x_ticks = np.arange(len(accs)) # 横坐标设置0,1,...,len(acc)-1,间隔为1
    plt.xticks(my_x_ticks)
    for a, b in zip(epochs, accs):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=5)
    plt.savefig(save_path+label,dpi=800)
#graphic

class BertDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return self.dataset[index]

# 回调函数
def coffate_fn(examples):
    inputs, targets = [], []
    for sent, polar in examples:
        inputs.append(sent)
        targets.append(int(polar))
    inputs = tokenizer(inputs,
                       padding=True,    # padding='max_length'训练效果极差
                       truncation=True,
                       return_tensors="pt",
                       max_length=512)
    targets = torch.tensor(targets)
    return inputs, targets

# 分层学习率递减
def get_group_parameters(model):
    params = [[] for _ in range(12)]
    other = []
    param_group = []
    beta = 0.95
    for n, p in model.named_parameters():  # n是每层的名称，p是每层的参数
        if 'layer' in n:
            striped = n[19:]
            index = striped.index('.')
            num = int(striped[:index])
            params[num].append(p)
        else:
            other.append(p)
    param_group.append({'params':other, 'lr':2e-5})
    lr = 2e-5
    for i in range(len(params), 0, -1):
        param_group.append({'params':params[i - 1], 'lr':lr})
        lr *= beta
    return param_group


# 数据预处理
# 获取训练、测试数据、分类类别总数
train_data = load_sentence_polarity(train_path)
test_data = load_sentence_polarity(test_path)
categories = {0, 1}

# 将训练数据和测试数据的列表封装成Dataset以供DataLoader加载
train_dataset = BertDataset(train_data)
test_dataset = BertDataset(test_data)

train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              collate_fn=coffate_fn,
                              shuffle=True)             # 经测试，shuffle对训练结果基本无影响
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             collate_fn=coffate_fn)


# 选择特征构建类型
print('1.pooler_output\n2.last_hidden_state取平均\n3.last_hidden_state取CLS\n4.last_hidden_state取最大值\n5.取后四层平均值，层内取平均\n6.取后四层平均值，层内取最大')
character = int(input('请选择特征类型:'))
# 模型初始化
model = BertSST2Model(len(categories), character, pretrained_model_name).to(device)

param_group = get_group_parameters(model)   # 得到分层递减的学习率
# optimizer = AdamW(model.parameters(),lr = learning_rate, eps = 1e-8)
optimizer = AdamW(param_group, learning_rate, eps = 1e-8)       # AdamW优化器

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)    # 分词器

criterion = nn.CrossEntropyLoss()  # 使用crossentropy作为二分类任务的损失函数

for epoch in range(1, num_epoch + 1):
    model.train()
    # 记录当前epoch的总loss
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
        inputs, targets = [x.to(device) for x in batch]
        # 清除现有的梯度
        optimizer.zero_grad()

        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        
        # 梯度反向传播
        loss.backward()

        # 根据反向传播的值更新模型的参数
        optimizer.step()

        total_loss += loss.item()
        
    # 测试过程
    # acc统计模型在测试数据上分类结果中的正确个数
    model.eval()
    #graphic
    acc = 0
    y_true = []
    y_score = []
    for batch in tqdm(test_dataloader, desc = "Testing"):
        inputs, targets = [x.to(device) for x in batch]
        y_true.append(targets.cpu().numpy())
        with torch.no_grad():    
            outputs = model(inputs)
            Softy = nn.Softmax(dim = 1)
            y_soft = Softy(outputs)
            y_score.append(y_soft[0][0].cpu().numpy())

    acc, rec, prec, f1 ,fpr, tpr, roc_auc= Modelanlysis(y_true, y_score, savepath, epoch+1, colors[epoch+1])# 阈值默认为0.5
    # 输出各项指标
    print(f"Acc: {acc :.4f}")
    print(f"Rec: {rec :.4f}")
    print(f"Prec: {prec :.4f}")
    print(f"F1: {f1 :.4f}")
    epochs.append(epoch)
    accs.append(acc)
    recs.append(rec)
    precs.append(prec)
    f1s.append(f1)
    fprs.append(fpr)
    tprs.append(tpr)
    roc_aucs.append(roc_auc)
    myroc_curve(epochs, fprs, tprs, roc_aucs, savepath)
    acc_curve(epochs, accs, savepath, 'acc')
    acc_curve(epochs, recs, savepath, 'rec')
    acc_curve(epochs, precs, savepath, 'prec')
    acc_curve(epochs, f1s, savepath, 'f1')
    #graphic

    