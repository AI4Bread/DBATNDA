import torch
from bat import BatAugmenter
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
from utils import get_model, get_device, print_centered
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.metrics import f1_score
import time

#dataset1
def readfile1():
    x = torch.load(".../converted_data/dataset1/node_features.pt")
    x = x.to(torch.float)
    y=torch.load(".../converted_data/dataset1/labels.pt")
    #y = torch.load(".../converted_data/dataset1/labels_2.pt")
    y = y.long()
    edge_index = torch.load(".../converted_data/dataset1/edge_index.pt")
    train_mask = torch.load(".../converted_data/all/train_mask.pt")
    train_mask = train_mask.long()
    test_mask = torch.load(".../converted_data/all/test_mask.pt")
    test_mask = test_mask.long()

    return x,y,edge_index,train_mask,test_mask

#dataset2
def readfile2():
    x = torch.load(
        ".../converted_data/dataset2/node_features.pt")
    x = x.to(torch.float)
    y = torch.load(".../converted_data/dataset2/labels.pt")
    #y = torch.load(".../converted_data/dataset2/labels_2.pt")
    y = y.long()
    edge_index = torch.load(".../converted_data/dataset2/edge_index.pt")
    train_mask = torch.load(
        ".../converted_data/all/train_mask.pt")
    train_mask = train_mask.long()
    test_mask = torch.load(
        ".../converted_data/all/test_mask.pt")
    test_mask = test_mask.long()

    return x, y, edge_index, train_mask, test_mask


class GCNNodeClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNNodeClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return self.fc(x)


if __name__ == '__main__':

    x,y,edge_index,train_mask,test_mask=readfile1()
    # initialize
    input_dim = x.size(1)
    hidden_dim = 128
    output_dim = 2

    model = GCNNodeClassifier(input_dim, hidden_dim, output_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=0, test_mask=test_mask)
    data = data.to(device)
    augmenter = BatAugmenter(mode='bat1').init_with_data(data)  # Initialize with graph data

    epochs = 500
    for epoch in range(epochs):
        # Augmentation
        if (epoch + 1) % 1 == 0: #granularities
            x, edge_index, _ = augmenter.augment(model, data.x, data.edge_index)
            y, train_mask = augmenter.adapt_labels_and_train_mask(data.y, data.train_mask)
            # print('augmentation')
        else:
            x = data.x
            edge_index = data.edge_index
            y = data.y
            train_mask = data.train_mask
        # Original training code
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

    start_time = time.time()
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        # OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
    end_time = time.time()

    #evaluate metrics
    if output_dim == 6:

        probs = F.softmax(out, dim=1).cpu()
        predict_labels = probs[test_mask].argmax(axis=1)
        cm = confusion_matrix(data.y[test_mask].cpu().numpy(), predict_labels)

        num_classes = cm.shape[0]

        class_accuracies = []

        for i in range(num_classes):
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP
            class_accuracy = TP / (TP + FN) if (TP + FN) > 0 else 0
            class_accuracies.append(class_accuracy)

        # Balanced Accuracy
        balanced_accuracy = np.mean(class_accuracies)

        print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

        # Marco-F1
        f1 = f1_score(data.y[test_mask].cpu(), predict_labels.cpu(), average=None)
        macro_f1 = f1.mean()
        print(f"Macro-F1: {macro_f1:.4f}")

        'AUC'
        auc = roc_auc_score(data.y[test_mask].cpu().numpy(), probs[test_mask], multi_class='ovr')
        print(f'AUC: {auc:.4f}')

        'AUPR'
        classes, train_class_counts = y.unique(return_counts=True)
        y_test = label_binarize(data.y[test_mask].cpu(), classes=classes.cpu())
        aupr_scores = []
        for i in range(classes.shape[0]):
            aupr = average_precision_score(y_test[:, i], probs[test_mask, i])
            aupr_scores.append(aupr)
        average_aupr = np.mean(aupr_scores)
        print(f'AUPR: {average_aupr:.4f}')
    if output_dim == 2:
        probs = F.softmax(out, dim=1)[:, 1].cpu()

        # get balanced accuracy
        threshold = 0.5
        predict_labels = (probs[test_mask] >= threshold).to(torch.int64)
        balanced_accuracy = balanced_accuracy_score(data.y[test_mask].cpu(), predict_labels.cpu(), adjusted=False)
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

        # get Marco-F1
        f1 = f1_score(data.y[test_mask].cpu(), predict_labels.cpu(), average=None)
        macro_f1 = f1.mean()
        print(f"Macro-F1: {macro_f1:.4f}")

        'AUC'
        auc = roc_auc_score(data.y[test_mask].cpu().numpy(), probs[test_mask], multi_class='ovr')
        print(f'AUC: {auc:.4f}')

        'AUPR'
        aupr = average_precision_score(data.y[test_mask].cpu().numpy(), probs[test_mask])
        print(f'AUPR: {aupr:.4f}')

