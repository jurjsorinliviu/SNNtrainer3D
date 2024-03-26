import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

training_names = [
    "SNN-1L-10n-LIF",
    "SNN_1L_10n_Lapicque",
    "SNN-1L-10n-syn",
    "SNN-1L-10n-alfa",
]

# load the data from {name}_train_data.csv
train_data = {}
for name in training_names:
    train_data[name] = pd.read_csv(f"{name}_train_data.csv")

# plot the data
for name in training_names:
    plt.plot(train_data[name]['iteration'], train_data[name]['train_loss'], label=f"{name}")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training loss")
plt.savefig("training_loss.png")
plt.show()

for name in training_names:
    plt.plot(train_data[name]['iteration'], train_data[name]['test_loss'], label=f"{name}")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Test loss")
plt.savefig("test_loss.png")
plt.show()

for name in training_names:
    plt.plot(train_data[name]['iteration'], train_data[name]['train_acc'], label=f"{name}")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Training accuracy")
plt.savefig("training_acc.png")
plt.show()

for name in training_names:
    plt.plot(train_data[name]['iteration'], train_data[name]['test_acc'], label=f"{name}")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Test accuracy")
plt.savefig("test_acc.png")
plt.show()
