# 基于BERT实现简单的情感分类任务,并探究不同特征构造方式的效果
项目链接：
```
https://github.com/SummerSolsticeMuch/Bert-sst2
```
# 任务简介

**情感分类**是指根据文本所表达的含义和情感信息将文本划分成褒扬的或贬义的两种或几种类型，是对文本作者倾向性和观点、态度的划分，因此有时也称倾向性分析（opinion analysis）。

本文通过简单的情感二分类任务作为样例，展示如何利用预训练模型**BERT**进行简单的Finetune过程。
# 数据准备
此任务以演示BERT用法为主，数据集采用SST-2的子集，即在原本数据集基础上进行抽取得到的部分，总计10000条。
## SST-2数据集
SST数据集： 斯坦福大学发布的一个情感分析数据集，主要针对电影评论来做情感分类，因此SST属于单个句子的文本分类任务（其中SST-2是二分类，SST-5是五分类，SST-5的情感极性区分的更细致）

SST数据集地址：https://nlp.stanford.edu/sentiment/index.html

## 示例
0——positive
1——negative
sentiment polarity     | sentence
-------- | -----
1 |	this is the case of a pregnant premise being wasted by a...
0 |	is office work really as alienating as 'bartleby' so effectively...
0 |	horns and halos benefits from serendipity but also reminds...
1 |	heavy-handed exercise in time-vaulting literary pretension.
0 |	easily one of the best and most exciting movies of the year.
1 |	you . . . get a sense of good intentions derailed by a failure...
1 |	johnson has , in his first film , set himself a task he is not nearly up to.
## 数据加载
在这里并不体现参数调优的过程，只设置训练集和测试集，没有验证集。
```python
def load_sentence_polarity(data_path):
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

```
**定义Dataset和Dataloader为后续模型提供数据：**
```python
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
```
DataLoader主要有以下几个参数：
Args:

 - dataset (Dataset): dataset from which to load the data.
 - batch_size (int, optional): how many samples per batch to load(default: ``1``).
 - shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch (default: ``False``).
 - collate_fn : 传入一个处理数据的回调函数

DataLoader工作流程：
1. 先从dataset中取出batch_size个数据
2. 对每个batch，执行collate_fn传入的函数以改变成为适合模型的输入
3. 下个epoch取数据前先对当前的数据集进行shuffle，以防模型学会数据的顺序而导致过拟合

有关Dataset和Dataloader具体可参考文章：[Pytorch入门：DataLoader 和 Dataset](https://blog.csdn.net/zw__chen/article/details/82806900)
# 模型介绍
本文采用最简单的**BertModel**，预训练模型加载的是 **bert-base-uncased**，在此基础上外加**Linear**层进行线性映射达到二分类目的：
```python
from transformers import BertModel

# 通过继承nn.Module类自定义符合自己需求的模型
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
```

# 探究不同特征构造方式
```python
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
```

# Finetune过程
## 参数设定
# 超参数
gpu_num = "2"
batch_size = 16
num_epoch = 4  # 训练轮次
train_path = "/home/xiazhiduo/pytorch/bert-sst2/train.tsv"  # 数据所在地址
test_path = "/home/xiazhiduo/pytorch/bert-sst2/dev.tsv"
learning_rate = 2e-5  # 优化器的学习率
device = torch.device(("cuda:" + gpu_num) if torch.cuda.is_available() else "cpu")
pretrained_model_name = 'bert-base-uncased'
```
## 优化器和损失函数
~~~
optimizer = AdamW(model.parameters(), learning_rate)  #使用AdamW优化器
CE_loss = nn.CrossEntropyLoss()  # 使用crossentropy作为二分类任务的损失函数
~~~
## 训练
```python
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
```
## 测试
```python
# 测试过程
# acc统计模型在测试数据上分类结果中的正确个数
model.eval()
for batch in tqdm(test_dataloader, desc = "Testing"):
    inputs, targets = [x.to(device) for x in batch]
    y_true.append(targets.cpu().numpy())
    with torch.no_grad():    
        outputs = model(inputs)
        Softy = nn.Softmax(dim = 1)
        y_soft = Softy(outputs)
        y_score.append(y_soft[0][0].cpu().numpy())
```
## 运行结果
模型在数据集上的准确率由50%以下上升到93%左右，有明显提升。


