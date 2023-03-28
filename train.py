from torch import optim
import torch
import torch.nn as nn
from Substrate_classifier import SubstrateClassifier
import numpy as np
import sklearn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from Data_prepration import read_data
import sklearn.metrics
from sklearn.metrics import matthews_corrcoef





use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def train(num_epochs, model, loss_function, optimizer):

    all_loss_val = []
    all_mcc = []
    all_f1 = []
    all_acc = []
    all_rec = []
    all_pre = []
    loss_min = 100
    for epoch in range(1, num_epochs + 1):
        all_loss = list()

        for i in tqdm(range(len(train_set))):
            model.zero_grad()
            sample = train_set[i]
            pred = model(sample[0])
            gold = torch.tensor([sample[1]], dtype=torch.long).to(device)
            loss = loss_function(pred, gold)
            loss.backward()
            all_loss.append(loss.cpu().detach().numpy())
            optimizer.step()
        print("Avg loss: " + str(np.mean(all_loss)))
        with torch.no_grad():
            model.eval()
            all_gold = list()
            all_pred = list()
            optimizer.zero_grad()
            for sample in val_set:
              pred = model(sample[0])
              all_gold.append(sample[1])
              gold=torch.tensor([sample[1]], dtype=torch.long).to(device)
              loss=loss_function(pred, gold)
              all_loss_val.append(loss.cpu().detach().numpy())
              pred=np.argmax(pred.cpu().detach().numpy())
              all_pred.append(pred)

        all_mcc.append(matthews_corrcoef(all_gold, all_pred))
        all_rec.append(sklearn.metrics.recall_score(all_gold, all_pred, average='macro'))
        all_pre.append(sklearn.metrics.precision_score(all_gold, all_pred, average='macro'))
        all_f1.append(sklearn.metrics.f1_score(all_gold, all_pred, average='macro'))
        all_acc.append(sklearn.metrics.accuracy_score(all_gold, all_pred))



# torch.save(model.state_dict(), "/content/gdrive/My Drive/Prot_bert_bfd_toot_sc_"+str(epoch)+"_epoch")






test_set, train_set = read_data()
trainset_x = []
for item in list(zip(*train_set))[0]:
  trainset_x.append(item)

trainset_y = []
for item in list(zip(*train_set))[1]:
  trainset_y.append(item)

X_train, X_val, y_train, y_val = train_test_split(trainset_x, trainset_y, test_size=0.2, random_state=42)

train_set = []
for x, y in zip(X_train, y_train):
  train_set.append((x, y))
val_set = []
for x, y in zip(X_val, y_val):
  val_set.append((x, y))



model = SubstrateClassifier(11).cuda()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000005)
num_epochs = 8
train(num_epochs, model, loss_function, optimizer, train_set, val_set)