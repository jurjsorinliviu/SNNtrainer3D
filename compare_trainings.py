import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from scipy.stats import wilcoxon

training_names = [
    "Net_05",
    "Net_010"
]
# load the json from {name}_summary.json
summaries = {}
for name in training_names:
    with open(f"{name}_summary.json", "r") as f:
        summaries[name] = json.load(f)
    # print(f"{name} summary: {summaries[name]}")

# load the json from {name}_test_accuracies.json
test_accuracies = {}
for name in training_names:
    test_accuracies[name] = summaries[name]["accuracies"]

if len(training_names) == 2:
    # print test accuracies
    # for name in training_names:
    #     print(f"{name} test accuracies: {test_accuracies[name]}")
    # perform wilcoxon test over the test data accuracy (assuming only two models)
    wilcoxon_test = wilcoxon(test_accuracies[training_names[0]], test_accuracies[training_names[1]])
    # analyze the data
    print()
    print("Wilcoxon test results:")
    if wilcoxon_test.pvalue < 0.05:
        print(" The difference in accuracy is statistically significant, p-value: ", wilcoxon_test.pvalue)
    else:
        print(" The difference in accuracy is not statistically significant, p-value: ", wilcoxon_test.pvalue)
    print()
    # exit()

# load the data from {name}_train_data.csv
train_data = {}
for name in training_names:
    train_data[name] = summaries[name]["train_data"]
    # convert to pd.DataFrame
    train_data[name] = pd.DataFrame(train_data[name])
    # print(f"{name} train data: {train_data[name]}")
    # train_data[name] = pd.read_csv(f"{name}_train_data.csv")

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
