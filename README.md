**# Synopsis

In this exercise I've tried to implement the formulas explained by [Javier Garcia](https://www.youtube.com/@Javier_Garcia) in his [Redes Neuronales - Fácil y desde cero](https://www.youtube.com/playlist?list=PLAnA8FVrBl8AWkZmbswwWiF8a_52dQ3JQ) series of videos.

Thank you so much mr. Javier!

# Usage documentation

```commandline
$ python .\main.py
usage: Usage:
======
.\main.py -l <layers array> -e <num epochs>
Example input:
========
 -l 3 4 3 2 --> means:                
                ► input layer with three neurons,
                ► first hidden layer with 4 neurons,
                ► second hidden layer with 3 neurons,
                ► Output layer with 2 neurons
 -e 5       --> means train the neural network 5 times
 -r 0.01    --> means train the neural at a increased learning rate of 0.01

main.py: error: the following arguments are required: -l/--layers, -e/--epochs
```

# Example: train and test a matrix with 3 x values, 2 hidden layers with 4 neurons each and 2 y values

Train with 5000 epoch loops

```commandline

python.exe main.py -l 3 4 4 2 -e 5000 -r 0.02

Testing inputs when the Neural Network still is not trained:

The output of the non-trained Neural Network for x=[1 4 7] is [ 0. -1.]

TRAINING THE NEURAL NETWORK: Applying the Back Propagation algorithm to the weight matrix and keeps the weights matrix with less errors:

minError=26.874771295543436, epoch=115


Neural network trained!!!. It took 0:01:20.394639 seconds to train it.

Training the Neural Network with 1000 epoch gave a 59.41% of successful predictions with the test dataset

The output after having trained the Neural Network for x=[ 5 14  3] is [ 0. -1.]
```

