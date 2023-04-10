import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch import optim
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from optuna.integration import PyTorchLightningPruningCallback

class SubstrateClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate, learning_rate):
        super().__init__()
        self.config = AutoConfig.from_pretrained("Rostlab/prot_bert_bfd")
        self.num_class = num_classes
        self.dropout_rate = dropout_rate
        self.bert = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")
        self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.learning_rate = learning_rate

    def forward(self, input):
        input = self.tokenizer(input, return_tensors="pt", truncation=True, max_length=1024)
        bert_rep = self.bert(input['input_ids'].cuda())
        cls_rep = bert_rep.last_hidden_state[0][0]
        cls_rep = self.dropout(cls_rep)
        class_scores = self.classifier(cls_rep)
        return F.log_softmax(class_scores.view(-1, self.num_class), dim=1)

    def training_step(self, train_batch, batch_idx):
        input, label = train_batch
        output = self.forward(input)
        loss = F.cross_entropy(output, label)
        return loss

    def validation_step(self, val_batch, batch_idx):
        input, label = val_batch
        output = self.forward(input)
        val_loss = F.cross_entropy(output, label)
        preds = torch.argmax(output, axis=1)
        accuracy = (preds == label).float().mean()
        return {"val_loss": val_loss, "val_accuracy": accuracy}

    def test_step(self, test_batch, batch_idx):
        input, label = test_batch
        output = self.forward(input)
        loss = F.cross_entropy(output, label)
        preds = torch.argmax(output, axis=1)
        accuracy = (preds == label).float().mean()
        return {"test_loss": loss, "test_accuracy": accuracy}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def trainer(self, num_epochs, train_loader, val_loader, patience, trial=None):
        '''
        :param num_epochs: integer number of epochs for training
        :param train_loader: a PyTorch DataLoader object for training
        :param val_loader: a PyTorch DataLoader object for validation
        :param patience: integer number of epochs to wait before early stopping
        :param trial: optuna trial object, default None
        :return:
        '''
        callbacks = []
        if trial:
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_loss"))
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            progress_bar_refresh_rate=0,
            gpus=1,
            callbacks=callbacks,
            early_stop_callback=EarlyStopping(monitor="val_loss", patience=patience)
        )
        trainer.fit(self, train_loader, val_loader)



    def test(self, test_loader):
        self.eval()
        test_losses = []
        test_accuracies = []
        with torch.no_grad():
            for batch in test_loader:
                input, label = batch
                output = self.forward(input)
                loss = F.cross_entropy(output, label)
                preds = torch.argmax(output, axis=1)
                accuracy = (preds == label).float().mean()
                test_losses.append(loss.item())
                test_accuracies.append(accuracy.item())
        avg_test_loss = np.mean(test_losses)
        avg_test_accuracy = np.mean(test_accuracies)
        print("Test Loss: {:.4f}, Test Accuracy: {:.4f}".format(avg_test_loss, avg_test_accuracy))
        return avg_test_loss, avg_test_accuracy

