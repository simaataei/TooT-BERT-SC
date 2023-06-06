from Substrate_classifier import SubstrateClassifier
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Data_prepration import read_data, sc_dataset
import pytorch_lightning as pl

num_classes = 11
dropout_rate = 0.0
learning_rate = 0.000005

model = SubstrateClassifier(num_classes, dropout_rate, learning_rate)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000005)
num_epochs = 8



batch_size = 1
# Create train_loader

X,y = read_data()
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)
train_dataset = sc_dataset(X_train, y_train)
val_dataset = sc_dataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
trainer = pl.Trainer(max_epochs=num_epochs, progress_bar_refresh_rate=0)
trainer.fit(model, train_loader, val_loader)

