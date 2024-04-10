import numpy as np

def sigmoid(x : float) -> float:
        return 1/(1+np.exp(-x))


#* Parent class for all layers
class Layer:
    def __init__(self):
        ...

    def computeLayerActivation(self) -> None:
        ...

    def gradDesc() -> None:
        ...

    def setActivation(self, data : np.array):
        ...


#* Definition of layer types
#* InputLayer
class InputLayer(Layer):
    def __init__(self, inputShape : tuple[int, int]):
        self.layerIndex : int = 0
        self.output : np.array = np.ndarray(inputShape)

    def setActivation(self, data : np.array):
        self.output = data

    # Nothing has to be trained in the inputLayer
    def gradDesc() -> None:
        return


#* PerceptronLayer / Fully Connected Layer
class PerceptronLayer(Layer):
    def __init__(self, inputLayer : Layer, layerNum : int, layerSize: int) -> None:
        self.layerType : int = "Perceptron"
        self.layerIndex : int = layerNum
        self.layerSize : int = layerSize
        self.inputLayer : Layer = inputLayer
        inputRows, inputCols = self.inputLayer.output[0].shape
        # Output from prev layer has to be unrolled if not a perceptron layer
        if self.inputLayer.layerType != "Perceptron":
            inputRows, inputCols = self.inputLayer.output[0].shape
            self.matrix : np.array = np.ndarray([layerSize, inputLayer.layerSize*inputRows*inputCols])
        else:
            self.matrix : np.array = np.ndarray([layerSize, inputLayer.layerSize])
        self.bias : np.array = np.ndarray([layerSize, 1])
        self.output : np.array = np.ndarray([layerSize, 1])
        # Init of variables
        for i in range(self.layerSize):
            self.bias[i, 0] = (np.random.rand() - 0.5) * 2
            for j in range(inputLayer.layerSize):
                self.matrix[i, j] = (np.random.rand() - 0.5) * 2
    
    def computeLayerActivation(self) -> None:
        """
        Computes activation of every node in the layer.
        """
        # Laver billedoutput om til en 1 dimensionel array
        if type(self.inputLayer.output) == list:
            input = np.concatenate(self.inputLayer.output, axis=0)
            input = np.concatenate(input, axis=0) [:, np.newaxis]
        
        self.output = sigmoid(self.matrix @ input + self.bias)

    # TODO : Implement gradient descent
    def gradDesc() -> np.array:
        ...

    def setActivation(self, data : np.array):
        raise Exception("Not allowed to set activation!")

        

    

#* Convolution layer
# Has a filter which it convolves across the input, and thus produces a new matrix, which i smaller than the input
class ConvolutionLayer(Layer):
    def __init__(self, inputLayer : Layer, layerNum : int, layerSize : int, filterSize : int):
        # Init of layer variables
        self.layerType : int = "Convolutional"
        self.layerIndex : int = layerNum
        self.inputLayer : Layer = inputLayer
        self.layerSize : int = layerSize

        # TODO : Implementer måde hvorpå hvert filter i laget kan defineres
        # Init of filters
        self.filters : list[np.array] = []
        for _ in range(layerSize):
            self.filters.append(np.ndarray([filterSize, filterSize]))

        for k in range(len(self.filters)):
            for i in range(filterSize):
                for j in range(filterSize):
                    self.filters[k][i, j] = (np.random.rand() - 0.5) * 2

        # Sets output shape
        self.output : list[np.array] = []
        shapeOfOutput = [dim - (filterSize-1) for dim in self.inputLayer.output.shape]
        for _ in range(layerSize):
            self.output.append(np.ndarray([shapeOfOutput[0], shapeOfOutput[1]]))
        
    

    def computeLayerActivation(self) -> None:
        input = self.inputLayer.output
        inputRows, inputColumns = input.shape
        imageDecrease = self.filters[0].shape[0] - 1
        for k in range(len(self.filters)):
            for i in range(inputRows)[int(imageDecrease/2): int(-imageDecrease/2)]:
                for j in range(inputColumns)[int(imageDecrease/2): int(-imageDecrease)]:
                    self.output[k][i - int(imageDecrease/2), j - int(imageDecrease/2)] = np.vdot(self.filters[k], input[i-1:i+2, j-1:j+2])
            
            self.output[k] = sigmoid(self.output[k])
    


    # TODO : Implement Gradient descent function for convolutional layer
    def gradDesc() -> np.array:
        ...



    def setActivation(self, data : np.array):
        raise Exception("Not allowed to set activation!")





#* PoolingLayer
# Takes input from a convolutional layer, which is an image (matrix), and downscales it using either maxPooling or averagePooling
class PoolingLayer(Layer):
    def __init__(self, inputLayer : Layer, layerNum : int, layerSize : int, downScale : int):
        """
        downScale describes the factor that the input is compromized. If a 16x16 image is the input, and the downScale is 4, the output will be 4x4
        """
        self.layerType : int = "Pooling"
        self.layerIndex : int = layerNum
        self.inputLayer : Layer = inputLayer
        shapeOfOutput = [int(dim/downScale) for dim in self.inputLayer.output[0].shape]
        self.layerSize : int = layerSize
        self.output : list[np.array] = []
        for _ in range(layerSize):
            self.output.append(np.ndarray([shapeOfOutput[0], shapeOfOutput[1]]))
        self.downScale = downScale

    def computeLayerActivation(self) -> None:
        input = self.inputLayer.output[0]
        input = np.hstack([input, np.zeros([input.shape[0], self.downScale - input.shape[1] % self.downScale])])
        input = np.vstack([input, np.zeros([self.downScale - input.shape[0] % self.downScale, input.shape[1]])])
        for k in range(len(self.output)):
            for i in range(self.output[0].shape[1]):
                for j in range(self.output[0].shape[0]):
                    row = i*self.downScale
                    col = j*self.downScale
                    self.output[k][i, j] = np.max(input[row: row + self.downScale, col: col + self.downScale])
        
    # TODO: Implement Gradient descent function for pooling layer
    def gradDesc() -> np.array:
        ...

    def setActivation(self, data : np.array):
        raise Exception("Not allowed to set activation!")

class NeuralNetwork:
    def __init__(self, layerDescription : list[dict[str]]) -> None:
        """
        The first layer is the inputlayer, and the last layer will be the outputlayer.
        Example of layerDescription:
            layerEx = [
                {
                    "layerShape" : 300, 300               # parameter for deciding the amount of nodes in the layer
                    "layerType" : 'input'           # 1'st layer will always be interpreted as an inputlayer
                }, 
                {
                    "layerSize" : 10,               # parameter for deciding the amount of nodes in the layer
                    "layerType" : 'convolutional'   # Use either 'convolutional', 'pooling', or 'perceptron' to define the layer type.
                    "filterSize": 3                 # Tells the size of the filter (Must be uneven to work) #!
                }, 
                {
                    "layerSize" : 10,               # parameter for deciding the amount of nodes in the layer
                    "layerType" : 'pooling'         # Use either 'convolutional', 'pooling', or 'perceptron' to define the layer type.
                    "downScale" : 4                 # Tells how compromized the output should be, in this case 1/4'th of the input
                }, 
                {
                    "layerSize" : 10,               # parameter for deciding the amount of nodes in the layer
                    "layerType" : 'convolutional'   # Use either 'convolutional', 'pooling', or 'perceptron' to define the layer type.
                }, 
                    ...
            ]
        """
        self.amountOfLayers : int = len(layerDescription)
        self.layers : list[Layer] = []

        # Inits every layer described in layerDescription
        self.layers.append(InputLayer(inputShape=layerDescription[0]["inputShape"]))
        i : int = 1
        for layer in layerDescription[1:]:
            if layer['layerType'].lower() == "perceptron":
                self.layers.append(PerceptronLayer(inputLayer=self.layers[i-1], layerNum=i, layerSize=layer['layerSize']))
            elif layer['layerType'].lower() == "convolutional":
                self.layers.append(ConvolutionLayer(inputLayer=self.layers[i-1], layerNum=i, layerSize=layer['layerSize'], filterSize=layer['filterSize']))
            elif layer['layerType'].lower() == "pooling":
                self.layers.append(PoolingLayer(inputLayer=self.layers[i-1], layerNum=i, layerSize=layer['layerSize'], downScale=layer['downScale']))
            i += 1

    def runNetwork(self) -> None:
        for layer in self.layers[1:]:
            layer.computeLayerActivation()

    def getOutput(self) -> np.array:
        return self.layers[-1].output

    
    def setInput(self, input: np.array) -> None:
        """
        Make sure the network is setup correctly to take the correct amount of inputs.
        """
        if self.layers[0].output.shape != input.shape:
            print(self.layers[0].output.shape, input.shape)
            raise Exception("Input shape does not match the shape of the first layer.")
        self.layers[0].setActivation(input)


    # TODO : Implementer gradient descent
    def trainNetworkGradDec(self, trainingInput: np.array, expectedOutputMatrix: np.array, gradientStepSize : float = 0.01) -> None:
        """
        Given testInput and the expected outputs for the inputs, this function will "train" the network on the data.
        """

        gradient = np.ndarray(sum([len(layer.nodes) + len(layer.nodes) * len(layer.inputLayer.nodes) for layer in reversed(self.layers[1:])]))

        # Partition data to "get more" training data.
        NOPartions : int = 10
        partitionSize : int = 30
        for partition in range(NOPartions):
            # Randomly choose first datapoint for partition
            firstRandomElement : int = np.random.randint(0, len(trainingInput[0])-1)
            randomInputs : np.array = trainingInput[:, firstRandomElement] [:, np.newaxis]
            randomOutputs : np.array = expectedOutputMatrix[:, firstRandomElement] [:, np.newaxis]
            # Randomly select the rest of the elements
            for _ in range(partitionSize-1):
                nextRandomElement : int = np.random.randint(0, len(trainingInput[0])-1)
                randomInputs = np.hstack([randomInputs, trainingInput[:, nextRandomElement] [:, np.newaxis]])
                randomOutputs = np.hstack([randomOutputs, expectedOutputMatrix[:, nextRandomElement] [:, np.newaxis]])
            
            # Begin training on the partitions
            for inputIndex in range(len(randomInputs)):
                self.setInput(input=randomInputs[:, inputIndex])
                self.runNetwork()
                expectedOutput : np.array = randomOutputs[:, inputIndex]

                #! Implementer videre herfra.