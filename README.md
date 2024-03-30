# Synopsis

This is an exercise trying to implement the formula developed by [Javier Garcia](https://www.youtube.com/@Javier_Garcia) in his [Redes Neuronales - Fácil y desde cero](https://www.youtube.com/playlist?list=PLAnA8FVrBl8AWkZmbswwWiF8a_52dQ3JQ) series of videos.

# Usage documentation

```commandline
$ python .\main.py
usage: Usage:
======
.\main.py -l <layers array> -e <num epochs>
Example input:
========
 -l 3 4 3 2 --> means:                ► input layer with three neurons,
                ► first hidden layer with 4 neurons,
                ► second hidden layer with 3 neurons,
                ► Output layer with 2 neurons
 -e 5       --> means train the neural network 5 times

main.py: error: the following arguments are required: -l/--layers, -e/--epochs
```

# Example: train and test a matrix with 3 x values, 2 hidden layers with 4 neurons each and 2 y values

Train with 5000 epoch loops

```commandline

python.exe main.py -l 3 4 4 2 -e 5000

Testing inputs when the Neural Network still is not trained:

The output of the non-trained Neural Network for x=[-1.  -0.5  0. ] is [-0.40935189 -0.15996135]

Applying the Back Propagation algorithm to the weight matrix and keeps the weights matrix with less errors:

minError=0.026388417826122883, epoch=6

The output of the non-trained Neural Network for x=[-1.  -0.5  0. ] is [0.84984607 1.04056515]
```

