import argparse
import random
import sys
import numpy


def initializeWeights(layers):
    """
    Initializes a set of 2 dimensional arrays with values between 0 and 1 representing weights connecting neurons of two
    consecutive layers, and rest of cells with value -10000 representing they aren't used
    The number of arrays will be (len(layers) - 1) / 2
    The dimension of each layer will be (layer[i] x layer[i + 1]), being i from 0 to len(layers) -1

     layers[3, 4, 5, 1]
                   º
            º
     º 	    	   º
            º
     º w[0]   w[1] º w[2] º
            º
     º	           º
            º
                   º

    :param layers: unidimensional array of layers. Each layer will come with the number of neurons contained on it

    :return: initialized weight list of arrays
    """
    w = []
    for k in range(0, len(layers) - 1):
        wpartial = numpy.full((layers[k+1], layers[k]), -10000, dtype=float)
        for i in range(0, layers[k]):
            for j in range(0, layers[k + 1]):
                wpartial[j][i] = random.random()

        w.append(wpartial)
    return w

def initializeActivationThreshold(layers):
    """
    Initializes a 2 dimensional array with values between 0 and 1 representing activation thresholds for each neuron,
    and rest of cells with value -10000 representing they aren't used
    The dimension of the array will be of [len(layers) + max(layers) - 1]

     layers[3, 4, 5, 1]

                   u[3,1]
            u[2,1] º
     u[1,1] º      u[3,2]
     º 	    u[2,2] º
     u[1,2] º      u[3,3] y[1]
     º      u[2,3] º      º
     u[1,3] º      u[3,4]
     º	    u[2,4] º
            º      u[3,5]
                   º
    :param layers: unidimensional array of layers. Each layer will come with the number of neurons contained on it

    :return: initialized activation threshold array
    """
    u = numpy.full((max(layers), len(layers)), -10000, dtype=float)
    for i in range(0, len(layers)):
        for j in range(0, layers[i]):
            u[j][i] = random.random()
    return u


def normalizeMatrix(matrix):
    '''
    normalizes the self.extendedTrainingValuesMatrix with the formula

                     actualCellValue - max(matrix)
    normCellValue = -------------------------------
                       max(matrix) - min(matrix)

    :param matrix: numpy matrix to be normalized
    :return: the normalized matrix
    '''
    if len(matrix.shape).__eq__(1):
        normalizedMatrix = numpy.full((matrix.shape[0]), 0.0, dtype=float)
        maxTrainingValue = matrix.max()
        minTrainingValue = matrix.min()
        for x in range(0, matrix.shape[0]):
            actualCellValue = matrix[x]
            if actualCellValue.__eq__(-10000):
                normalizedMatrix[x] = 0
            else:
                normalizedMatrix[x] = (actualCellValue - maxTrainingValue) / \
                                      (maxTrainingValue - minTrainingValue)
    else:
        normalizedMatrix = numpy.full((matrix.shape[0], matrix.shape[1]), 0.0, dtype=float)
        maxTrainingValue = matrix.max()
        minTrainingValue = matrix.min()
        for x in range(0, matrix.shape[0]):
            for y in range(0, matrix.shape[1]):
                actualCellValue = matrix[x][y]
                if actualCellValue.__eq__(-10000):
                   normalizedMatrix[x] = 0
                else:
                   normalizedMatrix[x][y] = (actualCellValue - maxTrainingValue) / \
                                                           (maxTrainingValue - minTrainingValue)
    return normalizedMatrix

class NeuralNetwork():
    def __init__(self, layers):
        # Neural network layers and neurons per layer
        self.layers = layers
        # Path weights
        self.w = initializeWeights(layers)
        # Activation threshold
        self.u = initializeActivationThreshold(layers)
        # Array of activation function per neuron
        self.a = numpy.full((max(self.layers), len(self.layers)), -10000, dtype=float)
        # Extended training set of the neural network
        self.extendedTrainingValuesMatrix = numpy.array([])

    def neuronActivation(self, x):
        """
        :param x: is the user entry
        :param error: is the deviation error obtained from the previous epoch
        :initializes: self.a, 2 dimensional array where each cell is the result of applying the neuron activation function

    Initializes a 2 dimensional array with values obtained by the function that calculates the output of a neuron
    Array positions not corresponding to a neuron are set to -10000 value

      layers[3, 4, 5, 1]

      k=1       #layers         k=4
    *─────────────────────────────►
i=0 │                 a[2,1]
 #  │        a[1,1]  º
 n  │  x[1]   º       a[2,2]
 e  │  º 	    a[1,2]  º
 u  │  x[2]   º       a[2,3]   y[1]
 r  │  º      a[1,3]  º        º
 o  │  x[3]   º       a[2,4]
 n  │  º	    a[1,4]  º
i=5 │         º       a[2,5]
    ▼

    k: layer where is located the neuron for which we are calculating the activation function
    i: sequence number in layer k where is located the neuron for which we are calculating the activation function

      If k = 1:
      =========

         1
        a = x
         i   i
         
      If k > 1:
      =========
               ┌                                            ┐
         (k)   │  (k)      layers[k-1] ┌  (k-1)     (k-1) ┐ │
        a   = f│ u    + SUM            │ a      . w       │ │
         i     │  i        j=1         │  j         ji    │ │
               │                       └                  ┘ │
               └                                            ┘

      If k is maximum index of layer:
      ===============================

        len(layer)
       a           = y  
        i              i
        """

        for k in range(0, len(layers)):
            # If k = 0:
            # =========
            #    1
            #   a = x
            #    i   i
            if (k == 0):
                for i in range(0, self.layers[k]):
                    self.a[i][k] = x[i]

            # If k > 1:
            # =========
            #          ┌                                            ┐
            #    (k)   │  (k)      layers[k-1] ┌  (k-1)     (k-1) ┐ │
            #   a   = f│ u    + SUM            │ a      . w       │ │
            #    i     │  i        j=1         │  j         ji    │ │
            #          │                       └                  ┘ │
            #          └                                            ┘
            if (k > 0):
                for i in range(0, self.layers[k]):
                    sum = 0
                    for j in range(0, self.layers[k-1]):
                       sum = sum + (self.a[j][k-1] * self.w[k-1][i][j])
                    self.a[i][k] = self.u[i][k] + sum
            # If k is maximum index of layer:
            # ===============================
            #
            #   len(layer)
            #  a           = y
            #   i              i
            if k == len(layers)-1:
                for i in range(0, self.layers[k]):
                    sum = 0
                    for j in range(0, layers[k]):
                        sum = sum + (self.a[j][k - 1] * self.w[k - 1][j][i])
                    self.a[i][k] = self.u[i][k] + sum

        # self.a = self.a - error

        # Get latest layer
        y = self.a[:, -1]
        # Get only the number of neurons in the latest layer
        y = y[:layers[k]]

        # y is the result
        return y

    def buildExtendedTrainNeuralNetwork(self, trainingValuesMatrix):
        '''
        Fills the self.extendedTrainingValuesMatrix matrix with the values to train the Neural Network
          plus the yHat results obtainded applying  nn.neuronActivation(x=trainingSet) to every row of the
          trainingValuesMatrix parameter

        :param trainingValuesMatrix: Matrix of values to train the Neural Network
        '''
        numRows = trainingValuesMatrix.shape[0]
        trainingList = []
        for i in range(0, numRows):
            trainingSet = trainingValuesMatrix[i][:-layers[-1]]
            y = trainingValuesMatrix[i][-layers[-1]:]
            yHat = nn.neuronActivation(x=trainingSet)
            # print(f'x={trainingSet}, y={y}, yHat={yHat}')
            trainingList.append(numpy.concatenate((trainingSet, y, yHat)))
        self.extendedTrainingValuesMatrix = numpy.array(trainingList)

    def calculateTheError(self):
        '''
        Calculates the Neural Network result error using the below function

        ÿi ---> obtained result in neuron i
        yi ---> real result in neuron i

                    i=#_output_neurons ┌            2 ┐
        error = SUM                    │  (ÿi - yi)   │
                                       │ -----------  │
                    i=1                │      2       │
                                       └              ┘
        '''


        yHat = nn.extendedTrainingValuesMatrix[:, -2]
        y = nn.extendedTrainingValuesMatrix[:, -1]
        numResults = y.size
        error = 0
        for i in range(0, numResults - 1):
            error = error + (  numpy.float_power((yHat[i] - y[i]), 2) / 2)
        return error

    def adjustWeights(self, error):
        '''
        Applies the error correction to the weights' matrix (nn.w) and
         the activation threshold's matrix (nn.u) in this manner:

        ► if value of cell is less than 0, then we add the error
        ► if value of cell is greater than 0, then we subtract the error

        :param error: correction used to adjust the weights
        '''
        for k in range(0, len(nn.w) - 1):
            for i in range(0, len(nn.w[k]) - 1):
                for j in range(0, len(nn.w[k][i]) - 1):
                    if nn.w[k][i][j] < 0:
                        nn.w[k][i][j] = nn.w[k][i][j] + error
                    else:
                        nn.w[k][i][j] = nn.w[k][i][j] - error

        for i in range(0, nn.u.shape[0]):
            for i in range(0, nn.u.shape[1]):
                if nn.u[i][j] > -10000:
                    if nn.u[i][j] < 0:
                        nn.u[i][j] = nn.u[i][j] + error
                    else:
                        nn.u[i][j] = nn.u[i][j] - error

    def backPropagation(self, epochs):
        '''
        Back propagates the obtained error repeating the training proces during # epochs to obtain the optimal nn.w matrix and
        replaces nn.w matrix with the values that causes the lowest error

        :param epochs: number of training iterations
        '''
        listErrors = []
        listWeigths = []
        print()
        for i in range(0, epochs):
            error = nn.calculateTheError()
            listErrors.append(error)
            listWeigths.append(nn.w)
            # print(f'epoch={i}, error={error}')
            # print(nn.w)
            # Adjust the weights with the obtained error
            nn.adjustWeights(error=error)
            # And train the network once again
            nn.buildExtendedTrainNeuralNetwork(trainingValuesMatrix)
        # Update the weights matrix with the values that gives the lowest calculated error
        minError = min(listErrors)
        positionOfMinError = listErrors.index(minError)
        print(f'minError={minError}, epoch={positionOfMinError}\n')
        # print(f'Optimal weight matrix:\n{listWeigths[positionOfMinError]}')
        nn.w = listWeigths[positionOfMinError]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage=f'Usage:\n'
              '======\n' +
              sys.argv[0] + ' -l <layers array> -e <num epochs>'
              '\n'
              'Example input:\n'
              '========\n'
              ' -l 3 4 3 2 --> means:'
              '                ► input layer with three neurons,\n'
              '                ► first hidden layer with 4 neurons,\n'
              '                ► second hidden layer with 3 neurons,\n'
              '                ► Output layer with 2 neurons\n'
              ' -e 5       --> means train the neural network 5 times\n'
    )

    parser.add_argument('-l', '--layers', nargs='+', type=int, required=True, help='Layers')
    parser.add_argument('-e', '--epochs', type=int, required=True, help='Epochs')
    args = vars(parser.parse_args())


    layers = args.get('layers')
    epochs = args.get('epochs')
    nn = NeuralNetwork(layers=layers)

    #########################################################################
    # Testing inputs when the Neural Network still is not trained
    #########################################################################

    print('')
    print('Testing inputs when the Neural Network still is not trained:\n')
    # Values passed by the user to the neuron activation function
    x = normalizeMatrix(numpy.array([1, 4, 7]))

    yHat = nn.neuronActivation(x=x)

    print(f'The output of the non-trained Neural Network for x={x} is {yHat}')

    #########################################################################
    # Training the Neural Network
    #########################################################################

    # Initialize the matrix of training values
    '''
     x1  x2  x3 │ y1  y2
     ───────────┼────────
      1   2  3  │ 4   3
      4   5  6  │ 2   1
     23   1  5  │ 7   4
    '''
    listOfTrainingValues = []
    listOfTrainingValues.append([1, 2, 3, 4, 3])
    listOfTrainingValues.append([4, 5, 6, 2, 1])
    listOfTrainingValues.append([23, 1, 5, 7, 4])
    trainingValuesMatrix = normalizeMatrix(numpy.array(listOfTrainingValues))
    # print(f'Matrix of values that we will use to train the Neural Network:\n{trainingValuesMatrix}\n')

    # Enrich the initial matrix of training values with the calculated yHat results
    '''
     x1  x2  x3 │ y1  y2 │  yHat1   yHat2
     ───────────┼────────┼──────────────────
      1   2  3  │ 4   3  │  -2.44     7.50
      4   5  6  │ 2   1  │   7.26    15.12
     23   1  5  │ 7   4  │  25.51    28.40
    '''
    nn.buildExtendedTrainNeuralNetwork(trainingValuesMatrix=trainingValuesMatrix)
    # print(f'Matrix of values to train the Neural Network including a column with calculated yHat results:\n'
    #       f'{nn.extendedTrainingValuesMatrix}\n')

    ############################################################################################################
    # Applying the Back Propagation algorithm to the weight matrix and keeps the weights matrix with less errors
    ############################################################################################################
    print('')
    print('Applying the Back Propagation algorithm to the weight matrix and keeps the weights matrix with less errors:')
    nn.backPropagation(epochs)

    #########################################################################
    # Testing inputs after the Neural Network has been trained
    #########################################################################

    # Values passed by the user to the neuron activation function
    x = normalizeMatrix(numpy.array([1, 4, 7]))

    yHat = nn.neuronActivation(x=x)

    print(f'The output after having trained the Neural Network for x={x} is {yHat}')
