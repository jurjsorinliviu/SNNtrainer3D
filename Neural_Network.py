import collections
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
# import itertools
import pandas as pd
import json
import pickle
import os
import random

num_steps = 100
# Define Network
class SNN(nn.Module):
    def __init__(self, num_inputs, hidden_layers, num_outputs, neuronType):
        super().__init__()
        self.neuron_type = neuronType
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self.hyperparameters = {}
        self.test_accuracy = None
    
    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def get_weight_list(self):
        # get fully connected weights (not biases)
        if len(self.hidden_layers) > 0:
            weights = []
            # get the input layer weights
            weights.append(self.fc1.weight.tolist())
            for layer in self.fcs:
                weights.append(layer.weight.tolist())
            # get the output layer weights
            weights.append(self.fc2.weight.tolist())
            return weights
        else:
            return [self.fc1.weight.tolist()]

    def save_model(self, path, pickle=False):
        if pickle:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        else:
            weights = self.state_dict()
            # convert to list
            for key in weights:
                w = weights[key].tolist()
                dtype = str(weights[key].dtype)
                weights[key] = {"data": w, "dtype": dtype}
            parameters = {
                "neuron_type": self.neuron_type,
                "num_inputs": self.num_inputs,
                "hidden_layers": self.hidden_layers,
                "num_outputs": self.num_outputs,
                "hyperparameters": self.hyperparameters
            }
            model_dict = {
                "weights": weights,
                "parameters": parameters
            }
            json.dump(model_dict, open(path, 'w'))
    
    @staticmethod
    def load_model(path, pickle=False):
        if pickle:
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            model_dict = json.load(open(path, 'r'))
            weights = collections.OrderedDict()
            for key in model_dict["weights"]:
                type_str = model_dict["weights"][key]["dtype"]
                class_name = type_str.split(".")[-1]
                # print(getattr(torch, class_name))
                w = torch.tensor(model_dict["weights"][key]["data"], dtype=getattr(torch, class_name))
                # w = torch.tensor(model_dict["weights"][key]["data"], dtype=getattr(torch, model_dict["weights"][key]["dtype"]))
                weights[key] = w
                # weights[key] = torch.tensor(model_dict["weights"][key]["data"])
            netClass = None
            if model_dict["parameters"]["neuron_type"] == "Leaky":
                netClass = LeakyNet
            elif model_dict["parameters"]["neuron_type"] == "Lapique":
                netClass = LapiqueNet
            elif model_dict["parameters"]["neuron_type"] == "Synaptic":
                netClass = SynapticNet
            elif model_dict["parameters"]["neuron_type"] == "Alpha":
                netClass = AlphaNet
            net = netClass(model_dict["parameters"]["num_inputs"], model_dict["parameters"]["hidden_layers"], model_dict["parameters"]["num_outputs"], **model_dict["parameters"]["hyperparameters"])
            net.load_state_dict(weights)
            return net

class LeakyNet(SNN):
    def __init__(self, num_inputs, hidden_layers, num_outputs, beta=0.95, **kwargs):
        super().__init__(num_inputs, hidden_layers, num_outputs, "Leaky")
        self.hyperparameters["beta"] = beta
        if len(hidden_layers) > 0:
            self.fc1 = nn.Linear(num_inputs, hidden_layers[0])
            self.lifs = nn.ModuleList([snn.Leaky(beta=beta) for _ in range(len(hidden_layers)+1)])
            self.fcs = nn.ModuleList([nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers)-1)])
            self.fc2 = nn.Linear(hidden_layers[-1], num_outputs)
        else:
            self.fc1 = nn.Linear(num_inputs, num_outputs)
            self.lifs = nn.ModuleList([snn.Leaky(beta=beta)])
            self.fcs = nn.ModuleList([])

    def forward(self, x):
        mems = [lif.init_leaky() for lif in self.lifs]
        spk_rec = []
        mems_rec = []

        
        for step in range(num_steps):
            if len(self.hidden_layers) > 0:
                cur = self.fc1(x)
                spk, mems[0] = self.lifs[0](cur, mems[0])
                for i in range(len(self.fcs)):
                    cur = self.fcs[i](spk)
                    spk, mems[i+1] = self.lifs[i+1](cur, mems[i+1])
                cur = self.fc2(spk)
                spk, mems[-1] = self.lifs[-1](cur, mems[-1])
                mems_rec.append(mems[-1].clone())
                spk_rec.append(spk)
            else:
                cur = self.fc1(x)
                spk, mems[0] = self.lifs[0](cur, mems[0])
                mems_rec.append(mems[0].clone())
                spk_rec.append(spk)
        return torch.stack(spk_rec, dim=0), torch.stack(mems_rec, dim=0)

class LapiqueNet(SNN):
    def __init__(self, num_inputs, hidden_layers, num_outputs, beta=0.95, **kwargs):
        super().__init__(num_inputs, hidden_layers, num_outputs, "Lapique")
        self.hyperparameters["beta"] = beta
        if len(hidden_layers) > 0:
            self.fc1 = nn.Linear(num_inputs, hidden_layers[0])
            self.lifs = nn.ModuleList([snn.Lapicque(beta=beta) for _ in range(len(hidden_layers)+1)])
            self.fcs = nn.ModuleList([nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers)-1)])
            self.fc2 = nn.Linear(hidden_layers[-1], num_outputs)
        else:
            self.fc1 = nn.Linear(num_inputs, num_outputs)
            self.lifs = nn.ModuleList([snn.Lapicque(beta=beta)])
            self.fcs = nn.ModuleList([])

    def forward(self, x):
        mems = [lif.reset_mem() for lif in self.lifs]
        spk_rec = []
        mems_rec = []

        
        for step in range(num_steps):
            if len(self.hidden_layers) > 0:
                cur = self.fc1(x)
                spk, mems[0] = self.lifs[0](cur, mems[0])
                for i in range(len(self.fcs)):
                    cur = self.fcs[i](spk)
                    spk, mems[i+1] = self.lifs[i+1](cur, mems[i+1])
                cur = self.fc2(spk)
                spk, mems[-1] = self.lifs[-1](cur, mems[-1])
                mems_rec.append(mems[-1].clone())
                spk_rec.append(spk)
            else:
                cur = self.fc1(x)
                spk, mems[0] = self.lifs[0](cur, mems[0])
                mems_rec.append(mems[0].clone())
                spk_rec.append(spk)
        return torch.stack(spk_rec, dim=0), torch.stack(mems_rec, dim=0)


class RealisticLapicqueNet(SNN):
    def __init__(self, num_inputs, hidden_layers, num_outputs, R=10, C=0.00015, time_step=0.001, **kwargs):
        super().__init__(num_inputs, hidden_layers, num_outputs, "RealisticLapicque")
        time_step = float(time_step)
        R = float(R)
        C = float(C)
        self.hyperparameters["R"] = R
        self.hyperparameters["C"] = C
        self.hyperparameters["time_step"] = time_step
        if len(hidden_layers) > 0:
            self.fc1 = nn.Linear(num_inputs, hidden_layers[0])
            self.lifs = nn.ModuleList([snn.Lapicque(R=R, C=C, time_step=time_step) for _ in range(len(hidden_layers)+1)])
            self.fcs = nn.ModuleList([nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers)-1)])
            self.fc2 = nn.Linear(hidden_layers[-1], num_outputs)
        else:
            self.fc1 = nn.Linear(num_inputs, num_outputs)
            self.lifs = nn.ModuleList([snn.Lapicque(R=R, C=C, time_step=time_step)])
            self.fcs = nn.ModuleList([])

    def forward(self, x):
        mems = [lif.reset_mem() for lif in self.lifs]
        spk_rec = []
        mems_rec = []

        
        for step in range(num_steps):
            if len(self.hidden_layers) > 0:
                cur = self.fc1(x)
                spk, mems[0] = self.lifs[0](cur, mems[0])
                for i in range(len(self.fcs)):
                    cur = self.fcs[i](spk)
                    spk, mems[i+1] = self.lifs[i+1](cur, mems[i+1])
                cur = self.fc2(spk)
                spk, mems[-1] = self.lifs[-1](cur, mems[-1])
                mems_rec.append(mems[-1].clone())
                spk_rec.append(spk)
            else:
                cur = self.fc1(x)
                spk, mems[0] = self.lifs[0](cur, mems[0])
                mems_rec.append(mems[0].clone())
                spk_rec.append(spk)
        return torch.stack(spk_rec, dim=0), torch.stack(mems_rec, dim=0)

class SynapticNet(SNN):
    def __init__(self, num_inputs, hidden_layers, num_outputs, alpha=0.95, beta=0.95, **kwargs):
        super().__init__(num_inputs, hidden_layers, num_outputs, "Synaptic")
        self.hyperparameters["alpha"] = alpha
        self.hyperparameters["beta"] = beta
        if len(hidden_layers) > 0:
            self.fc1 = nn.Linear(num_inputs, hidden_layers[0])
            self.lifs = nn.ModuleList([snn.Synaptic(alpha, beta) for _ in range(len(hidden_layers)+1)])
            self.fcs = nn.ModuleList([nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers)-1)])
            self.fc2 = nn.Linear(hidden_layers[-1], num_outputs)
        else:
            self.fc1 = nn.Linear(num_inputs, num_outputs)
            self.lifs = nn.ModuleList([snn.Synaptic(alpha, beta)])
            self.fcs = nn.ModuleList([])

    def forward(self, x):
        # cur1 = self.fc1(x)
        # spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
        # cur2 = self.fc2(spk1)
        # spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
        # return syn1, mem1, spk1, syn2, mem2, spk2
        synAndMems = [lif.init_synaptic() for lif in self.lifs]
        spk_rec = []
        syn_mems_rec = []
        # syn_rec = []

        
        for step in range(num_steps):
            if len(self.hidden_layers) > 0:
                cur = self.fc1(x)
                spk, syn, mem = self.lifs[0](cur, *synAndMems[0])
                synAndMems[0] = (syn, mem)
                for i in range(len(self.fcs)):
                    cur = self.fcs[i](spk)
                    spk, syn, mem = self.lifs[i+1](cur, *synAndMems[i+1])
                    synAndMems[i+1] = (syn, mem)
                cur = self.fc2(spk)
                spk, syn, mem = self.lifs[-1](cur, *synAndMems[-1])
                synAndMems[-1] = (syn, mem)
                syn_mems_rec.append(synAndMems[-1])
                spk_rec.append(spk)
            else:
                cur = self.fc1(x)
                spk, syn, mem = self.lifs[0](cur, *synAndMems[0])
                synAndMems[0] = (syn, mem)
                syn_mems_rec.append(synAndMems[0])
                spk_rec.append(spk)
        syn_rec = [syn_mems_rec[i][0] for i in range(len(syn_mems_rec))]
        mem_rec = [syn_mems_rec[i][1] for i in range(len(syn_mems_rec))]
        # print(len(syn_mems_rec))
        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
    
class AlphaNet(SNN):
    def __init__(self, num_inputs, hidden_layers, num_outputs, alpha=0.95, beta=0.95, **kwargs):
        super().__init__(num_inputs, hidden_layers, num_outputs, "Alpha")
        if alpha <= beta:
            print("Alpha must be greater than beta")
            print("Setting alpha to beta")
            alpha = beta + 10**-6
        self.hyperparameters["alpha"] = alpha
        self.hyperparameters["beta"] = beta
        if len(hidden_layers) > 0:
            self.fc1 = nn.Linear(num_inputs, hidden_layers[0])
            self.lifs = nn.ModuleList([snn.Alpha(alpha, beta) for _ in range(len(hidden_layers)+1)])
            self.fcs = nn.ModuleList([nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers)-1)])
            self.fc2 = nn.Linear(hidden_layers[-1], num_outputs)
        else:
            self.fc1 = nn.Linear(num_inputs, num_outputs)
            self.lifs = nn.ModuleList([snn.Alpha(alpha, beta)])
            self.fcs = nn.ModuleList([])

    def forward(self, x):
        synAndMems = [lif.init_alpha() for lif in self.lifs]
        spk_rec = []
        syn_mems_rec = []

        
        for step in range(num_steps):
            if len(self.hidden_layers) > 0:
                cur = self.fc1(x)
                spk, syn_exc, syn_inh, mem = self.lifs[0](cur, *synAndMems[0])
                synAndMems[0] = (syn_exc, syn_inh, mem)
                for i in range(len(self.fcs)):
                    cur = self.fcs[i](spk)
                    spk, syn_exc, syn_inh, mem = self.lifs[i+1](cur, *synAndMems[i+1])
                    synAndMems[i+1] = (syn_exc, syn_inh, mem)
                cur = self.fc2(spk)
                spk, syn_exc, syn_inh, mem = self.lifs[-1](cur, *synAndMems[-1])
                synAndMems[-1] = (syn_exc, syn_inh, mem)
                syn_mems_rec.append(synAndMems[-1])
                spk_rec.append(spk)
            else:
                cur = self.fc1(x)
                spk, syn_exc, syn_inh, mem = self.lifs[0](cur, *synAndMems[0])
                synAndMems[0] = (syn_exc, syn_inh, mem)
                syn_mems_rec.append(synAndMems[0])
                spk_rec.append(spk)
        syn_rec = [syn_mems_rec[i][0] for i in range(len(syn_mems_rec))]
        mem_rec = [syn_mems_rec[i][2] for i in range(len(syn_mems_rec))]
        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
    
def print_batch_accuracy(data, targets, net, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")
    return acc

def train_printer(
    data, targets, epoch,
    counter, iter_counter,
        loss_hist, test_loss_hist, test_data, test_targets, net):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    train_acc = print_batch_accuracy(data, targets, net, train=True)
    test_acc = print_batch_accuracy(test_data, test_targets, net, train=False)
    print("\n")
    return {
        "epoch": epoch,
        "iteration": counter,
        "train_loss": loss_hist[counter],
        "test_loss": test_loss_hist[counter],
        "train_acc": train_acc,
        "test_acc": test_acc
    }

# dataloader arguments
batch_size = 128
data_path='/tmp/data/mnist'

dtype = torch.float
# Load and transform mnist dataset
def loadMNIST():
    transform = transforms.Compose([
                # transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)


    # train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
    # return train_loader, test_loader

    return mnist_train, mnist_test


def train_network(net, mnist_train, mnist_test, flags, debug=False):
    # setSeed(flags['seed'])
    # TODO convert input to frequency for different training type (may be needed for STPD)
    train_loader = DataLoader(mnist_train, batch_size=flags['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=flags['batch_size'], shuffle=True, drop_last=True)
    epochs = flags['epochs']
    num_steps = flags['num_steps']
    batch_size = flags['batch_size']
    name = flags['network_name']
    # check if file named name already exists, if it does, add a random number to the end
    while os.path.exists(name):
        name = name + str(random.randint(0, 10))
    # else:
    # save flags dict as json in file
    with open(name, 'w') as f:
        json.dump(flags, f, indent=4)
        
    # Load the network onto CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    net.to(device)
    # net = Net(num_inputs, num_hidden, num_outputs, beta=beta).to(device)
    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=flags["learning_rate"], betas=(0.9, 0.999))

    num_epochs = epochs
    loss_hist = []
    test_loss_hist = []
    counter = 0
    train_data = pd.DataFrame(columns=["epoch", "iteration", "train_loss", "test_loss", "train_acc", "test_acc"])

    if debug:
        num_epochs = 1
    # Outer training loop
    for epoch in range(num_epochs):
        iter_counter = 0
        train_batch = iter(train_loader)

        # Minibatch training loop
        for data, targets in train_batch:
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            net.train()
            spk_rec, mem_rec = net(data.view(batch_size, -1))

            # print(len(mem_rec))
            # initialize the loss & sum over time
            loss_val = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += loss(mem_rec[step], targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            # Test set
            with torch.no_grad():
                net.eval()
                test_data, test_targets = next(iter(test_loader))
                test_data = test_data.to(device)
                test_targets = test_targets.to(device)

                # Test set forward pass
                test_spk, test_mem = net(test_data.view(batch_size, -1))

                # Test set loss
                test_loss = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    test_loss += loss(test_mem[step], test_targets)
                test_loss_hist.append(test_loss.item())

                # Print train/test loss/accuracy
                if counter % 50 == 0:
                    data_point = train_printer(
                        data, targets, epoch,
                        counter, iter_counter,
                        loss_hist, test_loss_hist,
                        test_data, test_targets, net)
                    train_data = train_data._append(data_point, ignore_index=True)
                counter += 1
                iter_counter +=1
            yield counter/(len(train_loader)*num_epochs)
            if debug:
                break

        # check if it's the last epoch
        if debug or epoch == num_epochs-1:
            counter -= 1
            iter_counter -= 1
            data_point = train_printer(
                data, targets, epoch,
                counter, iter_counter,
                loss_hist, test_loss_hist,
                test_data, test_targets, net)
            # check if this data point is already in the train data
            if len(train_data) == 0:
                train_data = train_data._append(data_point, ignore_index=True)
            elif train_data.iloc[-1]["iteration"] != data_point["iteration"]:
                train_data = train_data._append(data_point, ignore_index=True)

    # count true positives, true negatives, false positives, false negatives for each class over all the test data
    accs = []
    for _ in range(10):
        acc = 0
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            output, _ = net(data.view(batch_size, -1))
            _, idx = output.sum(dim=0).max(1)
            acc += np.sum((targets == idx).detach().cpu().numpy())
            if debug: break
        acc /= len(test_loader.dataset)
        net.test_accuracy = acc
        accs.append(acc)
    print(accs)
    TP = [0 for _ in range(10)]
    TN = [0 for _ in range(10)]
    FP = [0 for _ in range(10)]
    FN = [0 for _ in range(10)]
    # 10x10 confusion matrix
    confusion_matrix = np.zeros((10, 10))

    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)
        output, _ = net(data.view(batch_size, -1))
        _, idx = output.sum(dim=0).max(1)
        for i in range(10):
            for j in range(batch_size):
                if idx[j] == i and targets[j] == i:
                    TP[i] += 1
                elif idx[j] == i and targets[j] != i:
                    FP[i] += 1
                elif idx[j] != i and targets[j] == i:
                    FN[i] += 1
                else:
                    TN[i] += 1
        for i in range(batch_size):
            confusion_matrix[targets[i]][idx[i]] += 1

    # plot the confusion matrix
    plt.imshow(confusion_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(f"{name}_confusion_matrix.png")
    plt.close()

    precisions = [TP[i]/(TP[i]+FP[i]) if TP[i]+FP[i] != 0 else 0 for i in range(10)]
    recalls = [TP[i]/(TP[i]+FN[i]) if TP[i]+FN[i] != 0 else 0 for i in range(10) ]
    f1_scores = [2*precisions[i]*recalls[i]/(precisions[i]+recalls[i]) if precisions[i]+recalls[i] != 0 else 0 for i in range(10)]
    test_accuracy = (sum(TP)+sum(TN))/(sum(TP)+sum(TN)+sum(FP)+sum(FN))
    # save test data to file
    test_data = pd.DataFrame({
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Precision": precisions,
        "Recall": recalls,
        "F1 Score": f1_scores,
        "Test Accuracy": [test_accuracy]+[None]*9,
        "Test Loss": [test_loss_hist[-1]]+[None]*9
    })
    # test_data.to_csv(f"{name}_test_data.csv")

    # write a new file with accs
    # with open(f"{name}_test_accuracies.json", 'w') as f:
    #     json.dump({"accuracies": accs}, f, indent=4)

    # save train data to file
    # train_data.to_csv(f"{name}_train_data.csv")
    # with open(f"{name}_FLOPs.json", 'w') as f:
    #     json.dump({"FLOPs": flags["flops"]}, f, indent=4)

    # combine FLOPs, accuracies, train data and test data into a single file
    with open(f"{name}_summary.json", 'w') as f:
        json.dump({
            "FLOPs": flags["flops"],
            "accuracies": accs,
            "train_data": train_data.to_dict(),
            "statistics_data": test_data.to_dict()
        }, f, indent=4)
    # plot the train acc
    plt.plot(train_data['iteration'], train_data['train_acc'], label="Train Accuracy")
    plt.plot(train_data['iteration'], train_data['test_acc'], label="Test Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy during training")
    plt.legend()
    # save it to a file
    plt.savefig(f"{name}_accuracy_during_training.png")
    plt.close()
    # plot the train loss
    plt.plot(train_data['iteration'], train_data['train_loss'], label="Train Loss")
    plt.plot(train_data['iteration'], train_data['test_loss'], label="Test Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss during training")
    plt.legend()
    # save it to a file
    plt.savefig(f"{name}_loss_during_training.png")
    plt.close()



def train_network_xor(net, flags, debug=False):
    # setSeed(flags['seed'])
    # net = LapiqueNet(2, [2], 1, 10, 0.0015, 0.001)
    # train the network with xor
    x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    epochs = flags['epochs']
    num_steps = flags['num_steps']
    batch_size = 1
    name = flags['network_name']
    
    while os.path.exists(name):
        name = name + str(random.randint(0, 10))
    
    with open(name, 'w') as f:
        json.dump(flags, f, indent=4)
    # Load the network onto CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    net.to(device)
    # net = Net(num_inputs, num_hidden, num_outputs, beta=beta).to(device)
    loss = nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=flags["learning_rate"], betas=(0.9, 0.999))

    num_epochs = epochs
    loss_hist = []
    test_loss_hist = []
    counter = 0
    train_data = pd.DataFrame(columns=["epoch", "iteration", "train_loss", "test_loss", "train_acc", "test_acc"])

    if debug:
        num_epochs = 1
    iter_counter = 0
    # Outer training loop
    for epoch in range(num_epochs):
        iter_counter += 1
        counter += 1
        # sample a batch of from x,y
        train_batch = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y),
            batch_size=batch_size,
            shuffle=True
        )

        total_loss = torch.zeros((1), dtype=torch.float32, device=device)
        for data, targets in train_batch:
            # print(data, targets)
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            net.train()
            spk_rec, mem_rec = net(data.view(batch_size, -1))

            # print(len(mem_rec))
            # initialize the loss & sum over time
            loss_val = torch.zeros((1), dtype=torch.float32, device=device)
            for step in range(num_steps):
                loss_val += loss(mem_rec[step]/2, targets)
            total_loss += loss_val

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

        
        loss_hist.append(total_loss.item())

        if epoch % 10 == 0:
            net.eval()
            # compute accuracy
            errors = 0
            for xi, yi in zip(x, y):
                xi = xi.to(device)
                yi = yi.to(device)
                output, membranes = net(xi.view(1, -1))
                if output.mean() > 0.5 and yi == 0:
                    errors += 1
                elif output.mean() < 0.5 and yi == 1:
                    errors += 1
                # errors += torch.abs(yi - membranes.mean()/2).item()
                print(xi.numpy(), membranes.mean().detach().numpy())
            # print()
            print(f"Epoch {epoch} - Loss: {total_loss.item()} - Test Loss: {errors}")
            if errors == 0 and loss_val.item() < 1:
                break
            test_loss_hist.append(errors)
            # print(f"Epoch {epoch} - Loss: {loss_val.item()} - Test Loss: {errors}")
            train_data = train_data._append({
                "epoch": epoch,
                "iteration": iter_counter,
                "train_loss": total_loss.item(),
                "test_loss": 1,
                "train_acc": 0,
                "test_acc": (4-errors)/4
            }, ignore_index=True)
        yield counter/num_epochs
        
        if debug or (epoch == num_epochs-1 and not epoch % 10 == 0):
            counter -= 1
            iter_counter -= 1
            # update train data
            net.eval()
            # compute accuracy
            errors = 0
            for xi, yi in zip(x, y):
                xi = xi.to(device)
                yi = yi.to(device)
                output, membranes = net(xi.view(1, -1))
                if output.mean() > 0.5 and yi == 0:
                    errors += 1
                elif output.mean() < 0.5 and yi == 1:
                    errors += 1
                # errors += torch.abs(yi - membranes.mean()/2).item()
                print(xi.numpy(), membranes.mean().detach().numpy())
            # print()
            print(f"Epoch {epoch} - Loss: {total_loss.item()} - Test Loss: {errors}")
            if errors == 0 and total_loss.item() < 1:
                break
            test_loss_hist.append(errors)
            # print(f"Epoch {epoch} - Loss: {loss_val.item()} - Test Loss: {errors}")
            train_data = train_data._append({
                "epoch": epoch,
                "iteration": iter_counter,
                "train_loss": total_loss.item(),
                "test_loss": 1,
                "train_acc": 0,
                "test_acc": (4-errors)/4
            }, ignore_index=True)
    # count true positives, true negatives, false positives, false negatives
    accs = []
    for _ in range(10):
        errors = 0
        for xi, yi in zip(x, y):
            xi = xi.to(device)
            yi = yi.to(device)
            output, membranes = net(xi.view(1, -1))
            if output.mean() >= 0.5 and yi == 0:
                errors += 1
            elif output.mean() <= 0.5 and yi == 1:
                errors += 1
        net.test_accuracy = (4-errors)/4
        accs.append((4-errors)/4)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for xi, yi in zip(x, y):
        xi = xi.to(device)
        yi = yi.to(device)
        output, membranes = net(xi.view(1, -1))
        if output.mean() >= 0.5 and yi == 1:
            TP += 1
        elif output.mean() >= 0.5 and yi == 0:
            FP += 1
        elif output.mean() <= 0.5 and yi == 1:
            FN += 1
        elif output.mean() <= 0.5 and yi == 0:
            TN += 1
    # plot confusion matrix
    plt.imshow([[TP, FP], [FN, TN]], cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    # show ticks at 0 and 1 both x and y axes for predicted and actual values
    plt.xticks([0, 1], [0, 1])
    plt.yticks([0, 1], [0, 1])
    plt.title("Confusion Matrix")
    plt.savefig(f"{name}_confusion_matrix.png")
    plt.close()

    if TP == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1 = 2*(precision*recall)/(precision+recall)
    test_accuracy = sum(accs)/len(accs)
    test_data = {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Test Accuracy": test_accuracy,
        "Test Loss": test_loss_hist[-1]
    }
    with open(f"{name}_summary.json", "w") as f:
        json.dump({
            "FLOPs": flags['flops'],
            "accuracies": accs,
            "statistics_data": test_data,
            "train_data": train_data.to_dict()
        }, f, indent=4)
    
    # plot the train loss
    plt.plot(train_data['iteration'], train_data['train_loss'])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss during training")
    plt.legend()
    # save it to file
    plt.savefig(f"{name}_loss_during_training.png")
    plt.close()


if __name__ == "__main__":
    # net = Net(28*28, [100, 100, 100], 10, beta=0.95)
    # # get fully connected weights (not biases)
    # weights = []
    # # get the input layer weights
    # weights.append(net.fc1.weight.tolist())
    # for layer in net.fcs:
    #     weights.append(layer.weight.tolist())
    # # get the output layer weights
    # weights.append(net.fc2.weight.tolist())
    # for w in weights:
    #     print(w)
    # weights_path = "tmp_weights.pth"
    # net = Net.load_from_weights(weights_path)
    # print(net.test_accuracy)
    # net = LeakyNet(28*28, [], 10, beta=0.95, alpha=0.6)
    # # net.save_model("tmp_model.json")
    # # net = SynapticNet.load_model("tmp_model.json")
    # train_loader, test_loader = loadMNIST()
    # flags = {
    #     "mnist_loaded": True,
    #     "network_trained": False,
    #     "message": "",
    #     "num_steps": 25,
    #     "epochs": 1,
    #     "batch_size": 128,
    #     "learning_rate": 5e-4,
    #     "beta": 0.95,
    #     "alpha": 0.5,
    #     "render_nodes": False,
    #     "neuron_type": "Leaky",
    #     "weights": None,
    #     "network_name": "Net_0",
    #     "flops": 28*28*10
    # }
    # for progress in train_network(net, train_loader, test_loader, flags, debug=False):
    #     print(progress)
    #     pass

    # xor net
    # net = LapiqueNet(2, [2], 1, beta=0.95)
    flags = {
        "mnist_loaded": False,
        "network_trained": False,
        "message": "",
        "num_steps": 25,
        "epochs": 200,
        "batch_size": 128,
        "learning_rate": 5e-3,
        "beta": 0.95,
        "alpha": 0.5,
        "render_nodes": False,
        "neuron_type": "Leaky",
        "weights": None,
        "network_name": "Net_xor",
        "flops": 12,
        "R": 10,
        "C": 0.0015
    }
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    net = RealisticLapicqueNet(2, [2], 1, R=10, C=0.0015, time_step=0.001)
    # net = RealisticLapicqueNet(28*28, [], 10, R=10, C=0.0015, time_step=0.001)
    # net = LeakyNet(28*28, [], 10, beta=0.95, alpha=0.6)

    train_loader, test_loader = loadMNIST()
    # print(train_loader.dataset)
    # for progress in train_network(net, train_loader, test_loader, flags, debug=False):
    #     print(progress)
    #     pass
    
    for progress in train_network_xor(net, flags, debug=False):
        print(progress)
        pass
    net.save_model(f"{flags['network_name']}.json")

    # lif1 = snn.Lapicque(beta=0.95)
    # lif1.init_leaky()