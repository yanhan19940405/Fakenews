import pandas as pd
import numpy as np
import torch
import random
import math
import jieba
import matplotlib.pyplot as plt
from random import shuffle
import keras
import torch.nn.functional as F
from torch.nn import Conv2d
import gensim
import pickle
from att import AttLayer
from torch.utils.data import TensorDataset,DataLoader,Dataset
def handle_csv(path):
    data=[]
    file = open(path, encoding="utf-8")
    line = file.readline()
    while line:
        data.append(line.replace("\n", "").replace("\ufeff",''))
        line = file.readline()
    file.close()
    data = data[1:]
    data_x = []
    for i in data:
        data_x.append(i.split(","))
    shuffle(data_x)
    Query1 = []
    Query2 = []
    label = []
    for a in data_x:
        if len(a)==3:
            Query1.append(a[0])
            Query2.append(a[1])
            label.append(int(a[2]))
        else:
            pass
    return Query1,Query2,label



class Text_data(Dataset):
    def __init__(self):
        self.trainData = traindata
        self.label=trainlabel

    def __len__(self):
        return len(self.trainData)

    def __getitem__(self, idx):

        data = self.trainData[idx]
        label=self.label[idx]

        return data,label

class Test_data(Dataset):
    def __init__(self):
        self.testData = testdata
        self.label = testlabel

    def __len__(self):
        return len(self.testData)

    def __getitem__(self, idx):
        data = self.testData[idx]
        label=self.label[idx]

        return data, label


class CNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size,out):
        super(CNN, self).__init__()
        # self.features=torch.nn.Sequential(torch.nn.Embedding(vocab_size, embedding_size),
        #                                   torch.nn.Conv1d(maxlen, maxlen, embedding_size) )
        # self.classifer=torch.nn.Sequential(torch.nn.Linear(maxlen, output_size))
        self.embed = torch.nn.Embedding(vocab_size, embedding_size)
        self.conv1d=torch.nn.Conv1d(in_channels=embedding_size,out_channels=100,kernel_size=3,padding=1,stride=1)
        self.pool=torch.nn.MaxPool1d(kernel_size=2,stride=2)
        self.linear = torch.nn.Linear(10000,output_size)
        self.att=AttLayer(out)

    def forward(self, text):
        embedded = self.embed(text)  # [seq_len, batch_size, embedding_size]
        embedded = self.att(embedded)
        embedded = embedded.transpose(1, 2)
        conv1d=self.conv1d(embedded)
        conv1d = conv1d.transpose(1, 2)
        pooling=self.pool(conv1d)
        pooling=pooling.view(pooling.size()[0],-1)
        # embedded = embedded.transpose(1,2) # [batch_size, seq_len, embedding_size]
        # pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        x=self.linear(pooling)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


# def binary_accuracy(preds, y):
#     rounded_preds = torch.round(torch.sigmoid(preds))
#
#     correct = (rounded_preds == y).float()
#     acc = correct.sum() / len(correct)
#     return acc
#
# def binary_accuracy(preds, y):#标签one-hot
#     rounded_preds = torch.round(torch.sigmoid(preds))
#     correct = (rounded_preds == y).int().to('cpu')
#     count=[]
#     for i in range(len(correct)):
#         if set(correct[i].numpy())=={1}:
#             count.append(1)
#         else:
#             count.append(0)
#
#     acc = sum(count) / len(count)
#     return acc

def binary_accuracy(preds, y):#标签one-hot
    preds = torch.round(torch.sigmoid(preds))
    preds = preds.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    TP, TN, FN, FP = 0, 0, 0, 0
    for i in range(len(y)):
        a = [i for i in y[i]]
        b = [i for i in preds[i]]
        if b == [0, 1] and a == [0, 1]:
            TP = TP + 1
        # TN    predict 和 label 同时为0
        elif b == [1, 0] and a == [1, 0]:
            TN = TN + 1
        # FN    predict 0 label 1
        elif b == [1, 0] and a == [0, 1]:
            FN = FN + 1
        # FP    predict 1 label 0
        elif b == [0, 1] and a == [1, 0]:
            FP = FP + 1
    # print(TN,TP,FP,FN)
    e = 0.000001
    p = TP / (TP + FP + e)
    recall = TP / (TP + FN + e)
    F1 = 2 * recall * p / (recall + p + e)
    acc = (TP + TN) / (TP + TN + FP + FN)
    return acc, F1, p, recall
def train(model, iter, optimizer, loss_fn,epoch):
    epoch_loss, epoch_acc, epoch_recall, epoch_F1 = 0., 0., 0., 0.
    # model.train()
    total_len = 0.
    for i,data in enumerate(iter):
        # 前向传播
        data[0]=data[0].to(device)
        data[1]=data[1].to(device)
        y_pred = model(data[0]).squeeze()
        # 计算loss
        loss = loss_fn(y_pred.float(), data[1].float())
        acc, F1, P, recall = binary_accuracy(y_pred.float(), data[1].float())


        # 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(data[1])
        epoch_acc += acc * len(data[1])
        epoch_recall += recall * len(data[1])
        epoch_F1 += F1 * len(data[1])
        total_len += len(data[0])
        train_loss = epoch_loss / total_len
        train_acc = epoch_acc / total_len
        train_recall = epoch_recall / total_len
        train_F1 = epoch_F1 / total_len
        print("Epoch:", epoch, "iter:", i, "Train Loss:", train_loss, "Train Acc:", train_acc, "Train_recall:",train_recall, "Train_F1:", train_F1, "train_len:", total_len, 'lr', optimizer.param_groups[0]['lr'])
    print("Epoch:", epoch, "epoch train Loss:", train_loss, "Epoch train Acc：", train_acc, "Epoch train recall",train_recall, "Epoch Valid F1", train_F1)
    print("_____________________________________第" + str(epoch + 1) + "轮训练集合处理完毕，准备模型验证____________________________________________________")
    return epoch_loss / total_len, epoch_acc / total_len,epoch_recall/total_len ,epoch_F1/total_len


def evaluate(model, iter, loss_fn,epoch):
    epoch_loss, epoch_acc, epoch_recall, epoch_F1 = 0., 0., 0., 0.
    model.eval()
    total_len = 0.
    for i, data in enumerate(iter):
        data[0] = data[0].to(device)
        data[1] = data[1].to(device)
        model = model.to(device)
        loss_fn = loss_fn.to(device)
        y_pred = model(data[0]).squeeze()
        loss = loss_fn(y_pred.float(), data[1].float())
        acc, F1, P, recall = binary_accuracy(y_pred.float(), data[1].float())

        epoch_loss += loss.item() * len(data[1])

        epoch_acc += acc * len(data[1])
        epoch_recall += recall * len(data[1])
        epoch_F1 += F1 * len(data[1])
        total_len += len(data[0])
        valid_loss = epoch_loss / total_len
        valid_acc = epoch_acc / total_len
        valid_recall = epoch_recall / total_len
        valid_F1 = epoch_F1 / total_len
        print("i:", i, "Valid Loss:", valid_loss, "Valid Acc：", valid_acc)
    print("Epoch:", epoch, "epoch Valid Loss:", valid_loss, "Epoch Valid Acc：", valid_acc,'Epoch valid recall:', valid_recall,'Epoch valid F1:',valid_F1)
    print("_____________________________________第" + str(epoch + 1) + "轮训练结束____________________________________________________")
    model.train()

    return epoch_loss / total_len, epoch_acc / total_len,epoch_recall/total_len , epoch_F1/total_len
def plot_pic(train,test,name,y_name,x_name,file_name):
    plt.plot(train)
    plt.plot(test)
    plt.title(name)
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(file_name)
    plt.close()
if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    # 固定随机种子
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    if use_cuda:
        torch.cuda.manual_seed(1)
    MAXLEN = 200
    _, Query2, label = handle_csv("./data/train.csv")
    Query2 = Query2[1:]
    label = label[1:]
    data_text = []
    vocab_list = []
    for i in Query2:#分词
        data_text.append(jieba.lcut(i))
    for a in data_text:#生成词典
        for b in a:
            vocab_list.append(b)
    vocab = {}
    for index, con in enumerate(list(set(vocab_list))):
        vocab[str(con)] = index + 2
    vocab["UNK"] = 1
    vocab["PAD"] = 0
    output = open('./dic/vocab.pkl', 'wb')
    pickle.dump(vocab, output)
    output.close()
    day = data_text
    for j in range(len(data_text)):#文本补长
        if len(data_text[j]) <= MAXLEN:
            data_text[j].extend(['PAD'] * (MAXLEN - len(data_text[j])))
        elif len(data_text[j]) > MAXLEN:
            data_text[j] = data_text[j][0:MAXLEN]
    s = data_text[111][0]
    for m in range(len(data_text)):#文本矩阵化
        for n in range(len(data_text[m])):
            data_text[m][n] = vocab[data_text[m][n]]

    for v in range(len(label)):
        if label[v]==1:
            label[v]=[0,1]
        elif label[v]==0:
            label[v]=[1,0]
    Query2 = data_text
    traindata = Query2[0:20000]
    testdata = Query2[20000:]
    trainlabel = label[0:20000]
    testlabel = label[20000:]
    trainlabel = torch.tensor([x for x in trainlabel])
    traindata = torch.tensor(traindata)
    testdata = torch.tensor(testdata)
    testlabel = torch.tensor([x for x in testlabel])
    traindata = Text_data()#训练集数据生成器
    # traindata=TensorDataset(torch.tensor(traindata),torch.tensor([int(x) for x in trainlabel]))
    trainloader = DataLoader(traindata, batch_size=32, drop_last=True, shuffle=True)
    validdata = Test_data()#验证集数据生成器
    validloader = DataLoader(validdata, batch_size=32)

    model = gensim.models.Word2Vec.load('./model/text_w2v.model')#词向量
    embedding_matrix = np.zeros((len(vocab), 256))
    for word, i2 in vocab.items():
        if word in model:
            embedding_matrix[i2] = np.asarray(model[word])
        elif word not in model:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i2] = np.random.uniform(-0.25, 0.25, 256)
    model = CNN(vocab_size=len(vocab), embedding_size=256, output_size=2,out=256)#模型构建
    # model.initialized_weights()
    pretrained_embedding = torch.from_numpy(embedding_matrix)
    model.embed.weight.data.copy_(pretrained_embedding)

    # 开始训练
    lr=0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    N_EPOCHS = 100
    best_valid_acc = 0.
    trainloss=[]
    val_loss=[]
    trainacc=[]
    val_acc=[]
    train_recall=[]
    val_recall=[]
    train_F1=[]
    val_F1=[]
    lr_list=[]
    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='min', patience=10,
                                                        cooldown=0, min_lr=0.00001, verbose=0,eps=10**-6)
    for epoch in range(N_EPOCHS):
        if epoch < 10:
            ga = lr/ 10
            optimizer.param_groups[0]['lr'] = (epoch+1) * ga
        else:
            pass
        train_loss, train_acc ,trainrecall,trainF1= train(model,trainloader, optimizer, loss_fn,epoch)
        valid_loss, valid_acc,validrecall, validF1= evaluate(model, validloader, loss_fn,epoch)
        trainloss.append(train_loss)
        val_loss.append(valid_loss)
        trainacc.append(train_acc)
        val_acc.append(valid_acc)
        train_F1.append(trainF1)
        val_F1.append(validF1)
        train_recall.append(trainrecall)
        val_recall.append(validrecall)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), "./model/wordavg-model.pth")
        scheduler_lr.step(train_loss)

    plot_pic(trainloss,val_loss,'model loss','y_loss','epoch','./img/att_loss_3.png')
    plot_pic(trainacc, val_acc, 'model acc', 'y_acc', 'epoch', './img/att_acc_3.png')
    plot_pic(train_F1, val_F1, 'model F1-Score', 'F1-score', 'epoch', './img/att_F1.png')
    plot_pic(train_recall, val_recall, 'model recall', 'Recall', 'epoch', './img/att_Recall.png')

print(1)