import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import sklearn
import matplotlib.pyplot as plt
import h5py
from lib.test import *

import mlflow
import h5py

from torch import optim

import random

from torch.utils.data import DataLoader


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

import argparse

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=21, padding=10)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=15, padding=7)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=11, padding=5)
        
        self.batchnorm32 = nn.BatchNorm1d(num_features=32)
        self.batchnorm64 = nn.BatchNorm1d(num_features=64)
        self.batchnorm128 = nn.BatchNorm1d(num_features=128)
        self.batchnorm512 = nn.BatchNorm1d(num_features=512)
        
        self.fc1 = nn.Linear(4736, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)
        
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)     
        
        self.dropout2d = nn.Dropout2d(p=0.5)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.dropout2d(self.conv1(x.unsqueeze(1)))
        x = F.relu(self.batchnorm32(x))
        x = self.maxpool(x)
        
        x = self.dropout2d(self.conv2(x))
        x = F.relu(self.batchnorm64(x))
        x = self.maxpool(x)
        
        x = self.dropout2d(self.conv3(x))
        x = F.relu(self.batchnorm128(x))
        x = self.maxpool(x)
        
        # Flatten input for fully connected layers
        x = x.view(x.shape[0], -1) 
        
        x = self.dropout(self.fc1(x))
        x = F.relu(self.batchnorm512(x))
        
        x = self.dropout(self.fc2(x))
        x = F.relu(self.batchnorm512(x))
        
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


def parse_args():
	"""Parse arguments from the command line"""
	tool_description = "Application Description TDB"

	parser = argparse.ArgumentParser(description=tool_description)

	parser.add_argument("epochs", metavar="epochs_to_train", help="Epochs for training network")
	parser.add_argument("lr", metavar="learning_rate", help='Learning rate for optimizer')
	parser.add_argument("device", metavar="train_device", help="Device for training network: (cuda, cpu)")

	args = parser.parse_args()

	return args
    
def split_trainset(train_val_data, train_val_labels, ratio):
    """Split train data into train and validation set
    with given ratio"""
    data = list(zip(train_val_data, train_val_labels))
    
    random.shuffle(data)
    
    train_val_data, train_val_labels = list(zip(*data))
	
    train_ratio = ratio
    
    trainsize = int(len(train_val_data) * train_ratio)
    
    trainset = train_val_data[:trainsize]
    trainlabels = train_val_labels[:trainsize]
    
    valset = train_val_data[trainsize:]
    valabels = train_val_labels[trainsize:]
    
    return (trainset, trainlabels), (valset, valabels)

def parallelize(model):
    """
    Wrap pytorch model in layer to run on multiple GPUs
    """
    device_ids = [i for i in range(torch.cuda.device_count())]
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model

def run(epochs, lr):
    """
    Train and test CNN with given parameters
    """    

    mixdata = h5py.File("../train/scsn_p_2000_2017_6sec_0.5r_pick_train_mix.hdf5", "r")
    testdata = h5py.File("../test/scsn_p_2000_2017_6sec_0.5r_pick_test_mix.hdf5", "r")

    batch_size = 500

    train_size = 1 * 10 ** 5
    train_ratio = 0.5
    test_size = 1 * 10 ** 5

    # Load test data
    train_val_data = mixdata["X"][:train_size]
    train_val_labels = mixdata["pwave"][:train_size]
	
    (trainset, trainlabels), (valset, val_labels) = split_trainset(train_val_data, train_val_labels, train_ratio)

    trainset = list(zip(trainset, trainlabels))

    valset = list(zip(valset, val_labels))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = CNN()

    model = parallelize(model)

    #lr = float(args.lr)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mlflow.set_tracking_uri("file:.\mlruns")
    mlflow.start_run()

    #epochs = int(args.epochs)

    train_losses = []
    val_losses = []
    
    min_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch, labels in trainloader:
            # ============================================
            #            TRAINING
            # ============================================
            batch, labels = batch.to(device), labels.to(device)
            # Clear gradients in optimizer
            optimizer.zero_grad()
            # Forward pass
            output = model.forward(batch)
            # Calculate loss
            loss = criterion(output, labels.type(torch.cuda.LongTensor).view(labels.shape, 1))
            train_loss += loss.item()
            # Backpropagation
            loss.backward()
            # Update weights
            optimizer.step()
        else:
            with torch.no_grad():
                model.eval()
                val_loss = 0

                for batch, labels in val_loader:
                    # ============================================
                    #            VALIDATION
                    # ============================================
                    batch, labels = batch.to(device), labels.to(device)
                    # Forward pass
                    ouput = model.forward(batch)
                    # Calculate loss
                    loss = criterion(output, labels.type(torch.cuda.LongTensor).view(labels.shape, 1))
                    val_loss += loss.item()
                    
        # Print epoch summary
        t_loss_avg = train_loss / len(trainloader)
        v_loss_avg = val_loss / len(val_loader)
        
        if v_loss_avg < min_val_loss:
            torch.save(model.state_dict(), "./artifacts/model.pth")
            mlflow.log_artifact("./artifacts/model.pth")
        
        mlflow.log_metric("train_loss", t_loss_avg)
        mlflow.log_metric("val_loss", v_loss_avg)
        
        train_losses.append(t_loss_avg)
        val_losses.append(v_loss_avg)
        


    test_path = "../test/scsn_p_2000_2017_6sec_0.5r_pick_test_mix.hdf5"

    y_true, y_pred, y_probs = test_model(model, test_path, test_size, device="cuda")

    report = classification_report(y_true, y_pred, target_names=["pwave", "noise"])
    report_dict = classification_report(y_true, y_pred, target_names=["pwave", "noise"], output_dict=True)
    accuracy = accuracy_score(y_true, y_pred)
    roc_score = roc_auc_score(y_true, y_probs)

    print(report)
    print("Accuracy: {:.4}%".format(accuracy * 100))
    print("ROC Score: ", roc_score)

    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("device", device)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("auc_roc_score", roc_score)

    for category in report_dict:
        for metric, value in report_dict[category].items():
            metric_name = category + "_" + metric
            mlflow.log_metric(metric_name, value)
            
    mlflow.end_run()

def main():
	"""Run hyperparameter search"""
	for epoch in range(10, 50, 10):
		for lr in range(1, 9, 2):
			print("\nParameters: ")
			print("\tEpochs: {}".format(epoch))
			print("\tLearning Rate: {}".format(lr * 10**-3))
			
			run(epoch, lr * 10**-3)

if __name__ == "__main__":
	main()
