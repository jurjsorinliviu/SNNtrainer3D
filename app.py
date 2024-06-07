"""
app version 0.0.12
"""
from flask import Flask, render_template, request, jsonify
from Neural_Network import LeakyNet, loadMNIST, train_network, train_network_xor, LapiqueNet, SynapticNet, RealisticLapicqueNet
import threading
import numpy as np
import torch
import random

app = Flask(__name__)

mnist_train_dataset, mnist_test_dataset = None, None
# mnist_loaded = False

DATASET_INFORMATION = {
    "mnist": {
        "input_size": 28*28,
        "output_size": 10
    },
    "xor": {
        "input_size": 2,
        "output_size": 1
    }
}


# input_layer = 28*28
network_layers = [10]  # List to store network layers
# output_layer = 10

beta = 0.95

# neuralNetwork = LeakyNet(input_layer, network_layers, output_layer, beta)
neuralNetwork = None

percentageCompleted = 0
taskRunning = False
# network_trained = False

flags = {
    "mnist_loaded": False,
    "network_trained": False,
    "message": "",
    "num_steps": 25,
    "epochs": 1,
    "batch_size": 128,
    "learning_rate": 5e-3,
    "beta": 0.95,
    "alpha": 0.5,
    "render_nodes": False,
    "neuron_type": "Realistic Lapicque",
    "weights": None,
    "network_name": "Net",
    "flops": (DATASET_INFORMATION['xor']['input_size']*10 +10*DATASET_INFORMATION['xor']['output_size'])*2,
    "dataset": "xor",
    "R": 10,
    "C": 0.0015,
    "seed": 0,
    "time_step": 0.001
}


def setSeed(seed):
    # set the seed as numpy seed, torch seed and random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def update_flags(data):
    global flags
    print(data)
    flags['num_steps'] = int(data['num_steps'])
    flags['epochs'] = int(data['epochs'])
    flags['batch_size'] = int(data.get('batch_size', flags['batch_size']))
    flags['learning_rate'] = float(data['learning_rate'])
    flags['beta'] = float(data.get('beta', flags['beta']))
    flags['neuron_type'] = data['neuron_type']
    flags['network_name'] = data['network_name']
    flags['dataset'] = data['dataset']
    flags['R'] = float(data.get('R', flags['R']))
    flags['C'] = float(data.get('C', flags['C']))
    flags['seed'] = int(data['seed'])
    flags['time_step'] = float(data['time_step'])
    flags['alpha'] = float(data.get('alpha', flags['alpha']))
    if len(network_layers) > 0:
        flags['flops'] = 2*(DATASET_INFORMATION[flags['dataset']]['input_size']*network_layers[0] + sum([network_layers[i]*network_layers[i+1] for i in range(len(network_layers)-1)]) + network_layers[-1]*DATASET_INFORMATION[flags['dataset']]['output_size'])
    else:
        flags['flops'] = 2*(DATASET_INFORMATION[flags['dataset']]['input_size']*DATASET_INFORMATION[flags['dataset']]['output_size'])
    # latex
    """
    If $l_i$ is the size of layer $i$, the number of FLOPs of the model is:
    \begin{equation}
    \text{FLOPs} = 2\sum_{i=0}^{n}l_il_{i+1}
    \end{equation}
Where $l_0$ is the input size, and $l_{n+1}$ is the output size ($n$ being the number of hidden layers). This is valid for all fully connected layers.
    """
    if 'render_nodes' in data:
        flags['render_nodes'] = True
    else:
        flags['render_nodes'] = False
    flags['weights'] = None
    if neuralNetwork is not None:
        flags['weights'] = neuralNetwork.get_weight_list()

@app.route('/')
def index():
    return render_template('base.html', layers=network_layers, input_layer=DATASET_INFORMATION[flags['dataset']]['input_size'], output_layer=DATASET_INFORMATION[flags['dataset']]['output_size'], flags=flags, update_animation=True)


@app.route('/add_layer', methods=['POST'])
def add_layer():
    network_layers.append(1)  # Add a new layer with 1 node by default
    print(network_layers)
    return render_template('base.html', layers=network_layers, input_layer=DATASET_INFORMATION[flags['dataset']]['input_size'], output_layer=DATASET_INFORMATION[flags['dataset']]['output_size'], flags=flags, update_animation=True)


@app.route('/update_layers', methods=['POST'])
def update_layers():
    new_layers = request.form.getlist('layer')
    network_layers.clear()
    for layer in new_layers:
        network_layers.append(int(layer))
    print(network_layers)
    data = request.form.to_dict()
    update_flags(data)
    return render_template('base.html', layers=network_layers, input_layer=DATASET_INFORMATION[flags['dataset']]['input_size'], output_layer=DATASET_INFORMATION[flags['dataset']]['output_size'], flags=flags, update_animation=True)

# @app.route('/visualize')
# def visualize():
#     return render_template('visualize.html')

@app.route('/download_mnist', methods=['POST'])
def download_mnist():
    global mnist_train_dataset, mnist_test_dataset
    mnist_train_dataset, mnist_test_dataset = loadMNIST()
    return render_template('base.html', layers=network_layers, input_layer=DATASET_INFORMATION[flags['dataset']]['input_size'], output_layer=DATASET_INFORMATION[flags['dataset']]['output_size'], flags=flags, update_animation=False)

# @app.route('/train', methods=['POST'])
# def train():
#     print("Training")
#     train_network(input_layer, network_layers, DATASET_INFORMATION[flags['dataset']]['output_size'], mnist_train_dataset, mnist_test_dataset, 25, 128, 0.95)
#     return render_template('base.html', layers=network_layers, input_layer=input_layer, DATASET_INFORMATION[flags['dataset']]['output_size']=DATASET_INFORMATION[flags['dataset']]['output_size'], mnist_loaded=mnist_loaded, update_animation=False)

@app.route('/base')
def base():
    return render_template('base.html', input_layer=DATASET_INFORMATION[flags['dataset']]['input_size'], network_layers=network_layers, output_layer=DATASET_INFORMATION[flags['dataset']]['output_size'], flags=flags, update_animation=True)

@app.route('/update', methods=['POST'])
def update():
    print("Updating")
    data = request.get_json()  # Get the JSON data from the request body
    update_flags(data)
    return render_template('layers.html', layers=network_layers, input_layer=DATASET_INFORMATION[flags['dataset']]['input_size'], output_layer=DATASET_INFORMATION[flags['dataset']]['output_size'], flags=flags, update_animation=True)

@app.route('/progress')
def progress():
    global percentageCompleted, taskRunning
    # percentageCompleted += 1
    return jsonify(progress=percentageCompleted, task_running=taskRunning)

@app.route('/progress_MNIST')
def progress_MNIST():
    global percentageCompleted, taskRunning
    # percentageCompleted += 1
    return jsonify(progress=percentageCompleted, task_running=taskRunning)

@app.route('/start_task', methods=['POST'])
def start_task():
    data = request.get_json()  # Get the JSON data from the request body
    update_flags(data)
    setSeed(flags['seed'])
    global taskRunning, percentageCompleted, neuralNetwork
    # mnist_train_dataset, mnist_test_dataset = loadMNIST()
    if flags['neuron_type'] == "Lapique":
        print("Lapique")
        neuralNetwork = LapiqueNet(DATASET_INFORMATION[flags['dataset']]['input_size'], network_layers, DATASET_INFORMATION[flags['dataset']]['output_size'], **flags)
    elif flags['neuron_type'] == "Leaky":
        print("Leaky")
        neuralNetwork = LeakyNet(DATASET_INFORMATION[flags['dataset']]['input_size'], network_layers, DATASET_INFORMATION[flags['dataset']]['output_size'], **flags)
    elif flags['neuron_type'] == "Synaptic":
        print("Synaptic")
        neuralNetwork = SynapticNet(DATASET_INFORMATION[flags['dataset']]['input_size'], network_layers, DATASET_INFORMATION[flags['dataset']]['output_size'], **flags)
    elif flags['neuron_type'] == "Alpha":
        print("Alpha")
        neuralNetwork = SynapticNet(DATASET_INFORMATION[flags['dataset']]['input_size'], network_layers, DATASET_INFORMATION[flags['dataset']]['output_size'], **flags)
    elif flags['neuron_type'] == "Realistic Lapicque":
        print("Realistic Lapicque")
        neuralNetwork = RealisticLapicqueNet(DATASET_INFORMATION[flags['dataset']]['input_size'], network_layers, DATASET_INFORMATION[flags['dataset']]['output_size'], R=flags['R'], C=flags['C'], time_step=flags['time_step'])
    # neuralNetwork = SynapticNet(DATASET_INFORMATION[flags['dataset']]['input_size'], network_layers, DATASET_INFORMATION[flags['dataset']]['output_size'])
    def train_tmp_func():
        global taskRunning, percentageCompleted, flags
        if flags['dataset'] == "mnist":
            for progress in train_network(neuralNetwork, mnist_train_dataset, mnist_test_dataset, flags):
                percentageCompleted = progress*100
        elif flags['dataset'] == "xor":
            for progress in train_network_xor(neuralNetwork, flags):
                percentageCompleted = progress*100
        taskRunning = False
        flags['network_trained'] = True
        flags['message'] = f"Network trained! {100*neuralNetwork.test_accuracy:.2f}% accuracy on test set"

    # run task asynchronously
    task = threading.Thread(target=train_tmp_func)
    task.start()
    print("Task started")
    taskRunning = True
    return jsonify(progress=percentageCompleted, task_running=taskRunning)

@app.route('/download_mnist_task')
def download_mnist_task():
    global taskRunning, percentageCompleted
    def download_mnist_tmp_func():
        global taskRunning, percentageCompleted, mnist_train_dataset, mnist_test_dataset, flags
        mnist_train_dataset, mnist_test_dataset = loadMNIST()
        taskRunning = False
        flags['mnist_loaded'] = True
        flags['message'] = "MNIST dataset downloaded successfully"
    task = threading.Thread(target=download_mnist_tmp_func, daemon=True)
    task.start()
    print("Task started")
    taskRunning = True
    return jsonify(progress=percentageCompleted, task_running=taskRunning)

@app.route('/save_weights')
def save_weights():
    # prompt user for filename, open file dialog
    default_filename = f"{flags['network_name']}_model.json"
    neuralNetwork.save_model(default_filename)
    return {'filename': default_filename}
    # return send_file(default_filename, as_attachment=True)


@app.route('/get_weights')
def get_weights():
    return jsonify(weights=neuralNetwork.get_weight_list())

if __name__ == '__main__':
    app.run(debug=True)
