from torch import optim
import numpy as np
import argparse
from Bio import SeqIO
from tqdm import tqdm
import torch
import torch.nn as nn
from Data_prepration import read_data
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SubstrateClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.config = AutoConfig.from_pretrained("simaataei/TooT-BERT-SC")
        self.num_class = num_classes
        self.bert = AutoModel.from_pretrained("simaataei/TooT_BERT_SC")
        self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input):
        input = self.tokenizer(input, return_tensors="pt", truncation=True, max_length=1024)
        bert_rep = self.bert(input['input_ids'].cuda())
        cls_rep = bert_rep.last_hidden_state[0][0]
        class_scores = self.classifier(cls_rep)
        return F.log_softmax(class_scores.view(-1, self.num_class), dim=1)


model = SubstrateClassifier(11).to(device)
print("Reading training data")
train_set, test_set = read_data()
print("Training...")
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000005)
num_epochs = 8

for epoch in range(1, num_epochs + 1):
    all_loss = list()

    for i in tqdm(range(len(train_set))):
        model.zero_grad()
        sample = train_set[i]
        pred = model(sample[0])
        gold = torch.tensor([sample[1]], dtype=torch.long).cuda()
        loss = loss_function(pred, gold)
        loss.backward()
        all_loss.append(loss.cpu().detach().numpy())
        optimizer.step()
    print("Epoch: " + str(epoch))
    print("\nAvg loss: " + str(np.mean(all_loss)))




print("Testing...")
parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str, help='Input FASTA file')
parser.add_argument('output_file', type=str, help='Output txt file with predicted labels')

args = parser.parse_args()

with open(args.input_file, 'r') as f:
    records = list(SeqIO.parse(f, 'fasta'))
    sequences_ids = [(str(record.seq), str(record.id)) for record in records]


model.eval()
all_pred = list()
for sequence in sequences_ids:
    sequence = ' '.join(sequence[0])
    sequence = sequence.replace('U', 'X')
    sequence = sequence.replace('O', 'X')
    sequence = sequence.replace('B', 'X')
    sequence = sequence.replace('Z', 'X')
    pred = model(sequence)
    pred = np.argmax(pred.cpu().detach().numpy())
    all_pred.append(pred)

    print("\nSequence id: " + sequence[1] + " , Predicted class: "+ pred)
# write the output to the output file.
with open(args.output_file, 'w') as f:
    for prediction in all_pred:
        f.write(prediction + '\n')
