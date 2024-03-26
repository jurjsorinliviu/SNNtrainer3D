import * as THREE from "/static/three.module.js";
import { OrbitControls } from "/static/OrbitControls.js";

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth*2/3, window.innerHeight*9/10);
// find element called 'visualization-div'
document.getElementById('visualization-div').appendChild(renderer.domElement);
// document.body.appendChild(renderer.domElement);


// Add white rectangles to represent layers
const addLayerRect = (group, width, height, position) => {
    const geometry = new THREE.PlaneGeometry(width, height);
    const material = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });
    const plane = new THREE.Mesh(geometry, material);
    plane.position.x = position;
    // plane.position.y = height/2 - nodeSpacing/2;
    plane.position.y = - nodeSpacing/2;
    group.add(plane);
//   rotate the plane to be vertical
    plane.rotation.y = Math.PI / 2;
};

// add white rectangles with 1 depth to represent layers
const addLayerRect3D = (group, width, height, position) => {
    const depth = 0.1;
    const geometry = new THREE.BoxGeometry(depth, height, width);
    const material = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });
    const plane = new THREE.Mesh(geometry, material);
    plane.position.x = position;
    plane.position.y = - nodeSpacing/2;
    group.add(plane);
};

const buildLayer = (layerSize, nodeSize, nodeSpacing, position, group, color, showNodes=true) => {
    // Initialize layer layer
    const currentGroup = new THREE.Group();
    var nodeGroup;
    if (showNodes) {
        nodeGroup = new THREE.Group();
        for (let i = 0; i < layerSize; i++) {
        const geometry = new THREE.SphereGeometry(nodeSize, 32, 32);
        const material = new THREE.MeshBasicMaterial({ color: color });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(position, (i-layerSize/2) * nodeSpacing, 0);
        nodeGroup.add(sphere);
        }
        currentGroup.add(nodeGroup);
    }
    // add layerRect
    addLayerRect3D(currentGroup, 5, layerSize*nodeSpacing, position);
    group.add(currentGroup);
    return nodeGroup;
}

const build2DLayer = (layerSize1, layerSize2, nodeSize, nodeSpacing, position, group, color) => {
    // Initialize layer layer
    const currentGroup = new THREE.Group();
    for (let i = 0; i < layerSize1; i++) {
    for (let j = 0; j < layerSize2; j++) {
        const geometry = new THREE.SphereGeometry(nodeSize, 32, 32);
        const material = new THREE.MeshBasicMaterial({ color: color });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(position, (i-layerSize1/2) * nodeSpacing, (j-layerSize2/2) * nodeSpacing+nodeSpacing/2);
        currentGroup.add(sphere);
    }
    }
    // add layerRect
    addLayerRect(currentGroup, layerSize1*nodeSpacing, layerSize2*nodeSpacing, position);
    group.add(currentGroup);
}

const makeConnectionBetweenNodes = (groupOfNodes1, groupOfNodes2, neuralNetworkGroup, weights=null) => {
    for (let node1 of groupOfNodes1.children) {
    for (let node2 of groupOfNodes2.children) {
        var material;
        // make a line of a random green or red color with opacity
        if (weights) {
            const node1index = groupOfNodes1.children.indexOf(node1);
            const node2index = groupOfNodes2.children.indexOf(node2);
            
            const weight = weights[node2index][node1index];
            const opacity = Math.abs(weight);
            material = new THREE.LineBasicMaterial({ color: weight > 0 ? 0x00ff00 : 0xff0000, transparent: true, opacity: opacity });
        } else {
            material = new THREE.LineBasicMaterial({ color: Math.random() > 0.5 ? 0x00ff00 : 0xff0000, transparent: true, opacity: 0.2 });
        }
        const points = [node1.position, node2.position];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const line = new THREE.Line(geometry, material);
        neuralNetworkGroup.add(line);
    }
    }
}

const controls = new OrbitControls( camera, renderer.domElement );

// const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.25;
controls.enableZoom = true;


// const inputLayer = 16*16;
// const hiddenLayers = [4, 3, 2, 10, 5];
// const outputLayer = 2;

const nodeSize = 0.1; // Size of nodes
const nodeSpacing = 0.5; // Spacing between nodes
const layerSpacing = 20; // Spacing between layers


function buildNeuralNetwork(inputLayer, hiddenLayers, outputLayer, weights=null, showNodes=true) {
    const neuralNetworkGroup = new THREE.Group()
    const scene = new THREE.Scene();
    scene.add(neuralNetworkGroup)

    var currentLayer = 0;

    // build a 2D layer of 4x4
    // build2DLayer(28, 28, nodeSize, nodeSpacing, -1*layerSpacing, neuralNetworkGroup, 0x00ff00);

    const inputNodes = buildLayer(inputLayer, nodeSize, nodeSpacing, currentLayer*layerSpacing, neuralNetworkGroup, 0x00ff00, showNodes);
    currentLayer += 1;

    var hiddenNodes = []
    for (let hiddenLayer of hiddenLayers) {
        const nodes = buildLayer(hiddenLayer, nodeSize, nodeSpacing, currentLayer*layerSpacing, neuralNetworkGroup, 0xff0000, showNodes);
        currentLayer += 1;
        hiddenNodes.push(nodes);
    }

    const outputNodes = buildLayer(outputLayer, nodeSize, nodeSpacing, currentLayer*layerSpacing, neuralNetworkGroup, 0x0000ff, showNodes);
    currentLayer += 1;

    // move neuralnetwork group to the left to center it
    neuralNetworkGroup.position.x = -currentLayer*layerSpacing/2

    // make connections between nodes
    if (showNodes){
        var weight = null;
        if (weights) {
            weight = weights[0];
        }
        makeConnectionBetweenNodes(inputNodes, hiddenNodes[0], neuralNetworkGroup, weight);
        for (let i = 0; i < hiddenNodes.length-1; i++) {

            if (weights) {
                weight = weights[i+1];
            }
            makeConnectionBetweenNodes(hiddenNodes[i], hiddenNodes[i+1], neuralNetworkGroup, weight);
        }
        if (weights) {
            weight = weights[weights.length-1];
        }
        makeConnectionBetweenNodes(hiddenNodes[hiddenNodes.length-1], outputNodes, neuralNetworkGroup, weight);
    }
    camera.position.z = 100;
    // camera.position.z = currentLayer*layerSpacing/2
    // camera.position.y = currentLayer*layerSpacing/2
    // camera.position.x = currentLayer*layerSpacing/2
    animate(scene);
}

const animate = function (scene) {
    requestAnimationFrame(function () {
        animate(scene);
    });

    controls.update(); // Update controls

    renderer.render(scene, camera);
};
export { buildNeuralNetwork}