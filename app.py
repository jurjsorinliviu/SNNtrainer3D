from flask import Flask, render_template, request, jsonify
from Neural_Network import LeakyNet, loadMNIST, train_network, LapiqueNet, SynapticNet
import threading

app = Flask(__name__)

train_loader, test_loader = None, None
# mnist_loaded = False

input_layer = 28*28
network_layers = [10]  # List to store network layers
output_layer = 10

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
    "learning_rate": 5e-4,
    "beta": 0.95,
    "alpha": 0.5,
    "render_nodes": False,
    "neuron_type": "Leaky",
    "weights": None,
    "network_name": "Net",
    "flops": (input_layer*10 +10*output_layer)*2
}

def update_flags(data):
    global flags
    print(data)
    flags['num_steps'] = int(data['num_steps'])
    flags['epochs'] = int(data['epochs'])
    flags['batch_size'] = int(data['batch_size'])
    flags['learning_rate'] = float(data['learning_rate'])
    flags['beta'] = float(data['beta'])
    flags['neuron_type'] = data['neuron_type']
    flags['network_name'] = data['network_name']
    # flags['alpha'] = float(data['alpha'])
    if len(network_layers) > 0:
        flags['flops'] = 2*(input_layer*network_layers[0] + sum([network_layers[i]*network_layers[i+1] for i in range(len(network_layers)-1)]) + network_layers[-1]*output_layer)
    else:
        flags['flops'] = 2*(input_layer*output_layer)
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
    return render_template('base.html', layers=network_layers, input_layer=input_layer, output_layer=output_layer, flags=flags, update_animation=True)


@app.route('/add_layer', methods=['POST'])
def add_layer():
    network_layers.append(1)  # Add a new layer with 1 node by default
    print(network_layers)
    return render_template('base.html', layers=network_layers, input_layer=input_layer, output_layer=output_layer, flags=flags, update_animation=True)


@app.route('/update_layers', methods=['POST'])
def update_layers():
    new_layers = request.form.getlist('layer')
    network_layers.clear()
    for layer in new_layers:
        network_layers.append(int(layer))
    print(network_layers)
    data = request.form.to_dict()
    update_flags(data)
    return render_template('base.html', layers=network_layers, input_layer=input_layer, output_layer=output_layer, flags=flags, update_animation=True)

# @app.route('/visualize')
# def visualize():
#     return render_template('visualize.html')

@app.route('/download_mnist', methods=['POST'])
def download_mnist():
    global train_loader, test_loader
    train_loader, test_loader = loadMNIST()
    return render_template('base.html', layers=network_layers, input_layer=input_layer, output_layer=output_layer, flags=flags, update_animation=False)

# @app.route('/train', methods=['POST'])
# def train():
#     print("Training")
#     train_network(input_layer, network_layers, output_layer, train_loader, test_loader, 25, 128, 0.95)
#     return render_template('base.html', layers=network_layers, input_layer=input_layer, output_layer=output_layer, mnist_loaded=mnist_loaded, update_animation=False)

@app.route('/base')
def base():
    return render_template('base.html', input_layer=input_layer, network_layers=network_layers, output_layer=output_layer, flags=flags, update_animation=True)

@app.route('/update', methods=['POST'])
def update():
    print("Updating")
    data = request.get_json()  # Get the JSON data from the request body
    update_flags(data)
    return render_template('layers.html', layers=network_layers, input_layer=input_layer, output_layer=output_layer, flags=flags, update_animation=True)

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
    global taskRunning, percentageCompleted, neuralNetwork
    # train_loader, test_loader = loadMNIST()
    if flags['neuron_type'] == "Lapique":
        print("Lapique")
        neuralNetwork = LapiqueNet(input_layer, network_layers, output_layer, **flags)
    elif flags['neuron_type'] == "Leaky":
        print("Leaky")
        neuralNetwork = LeakyNet(input_layer, network_layers, output_layer, **flags)
    elif flags['neuron_type'] == "Synaptic":
        print("Synaptic")
        neuralNetwork = SynapticNet(input_layer, network_layers, output_layer, **flags)
    elif flags['neuron_type'] == "Alpha":
        print("Alpha")
        neuralNetwork = SynapticNet(input_layer, network_layers, output_layer, **flags)
    # neuralNetwork = SynapticNet(input_layer, network_layers, output_layer)
    def train_tmp_func():
        global taskRunning, percentageCompleted, flags
        for progress in train_network(neuralNetwork, train_loader, test_loader, flags):
            percentageCompleted = progress*100
        taskRunning = False
        flags['network_trained'] = True
        flags['message'] = f"Network trained! {neuralNetwork.test_accuracy:.2f}% accuracy on test set"

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
        global taskRunning, percentageCompleted, train_loader, test_loader, flags
        train_loader, test_loader = loadMNIST()
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
