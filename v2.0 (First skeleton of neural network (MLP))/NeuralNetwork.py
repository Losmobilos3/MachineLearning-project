import numpy as np



class Node:
    def __init__(self, layerNum : int, inputLayer : 'Layer', weights: np.array, bias: float) -> None:
        self.layerNum : int = layerNum
        self.inputLayer : Layer = inputLayer
        self.weights : np.array = weights
        self.activation : float = 0.0
        self.bias : float = bias


    def sigmoid(self, x : float):
        return 1/(1+np.exp(-x))


    def computeActivation(self) -> None:
        inputActivations = np.array([node.activation for node in self.inputLayer.nodes])
        sum = np.vdot(self.weights, inputActivations) + self.bias
        # Compress down between 0 and 1 (Runs sum through a sigmoid)
        self.activation = self.sigmoid(x = sum)


    def gradDecend(self, derivCvrtA : float = 1) -> np.array:
        if self.layerNum == 0:
            return np.ndarray(0), np.ndarray(0) # Returns empty array if we've reached the inputlayer
        
        # Derivative of activation a in relation to z (the sum)
        derivAvrtZ = self.activation * (1 - self.activation)

        # init of Gradient
        gradient = np.ndarray(1)

        # Derivative of costfunction in relation to the bias
        derivCvrtB = derivCvrtA * derivAvrtZ * 1
        gradient[0] = derivCvrtB

        # Calculation of derivatives of costfunction in relation to the weights
        for weightIndex in range(len(self.weights)):
            derivCvrtWn = derivCvrtA * derivAvrtZ * self.inputLayer.nodes[weightIndex].activation
            gradient = np.hstack([gradient, np.array([derivCvrtWn])])

        # Recursive step
        # Calculates the gradient with respect to each lower layer node
        lowerGradient : np.array = np.ndarray(0)
        passedLowerGradient : np.array = np.ndarray(0)
        for inputNodeIndex in range(len(self.inputLayer.nodes)):
            derivZvrtAn = self.weights[inputNodeIndex]
            if len(lowerGradient) == 0:
                passedGradient, passedLowerGradient = [derivCvrtA * derivAvrtZ * derivZvrtAn * gradientPart for gradientPart in self.inputLayer.nodes[inputNodeIndex].gradDecend()]
                lowerGradient = np.hstack([lowerGradient, passedGradient])
            else:
                passedGradient, passedLowerGradientAdd = [derivCvrtA * derivAvrtZ * derivZvrtAn * gradientPart for gradientPart in self.inputLayer.nodes[inputNodeIndex].gradDecend()]
                lowerGradient = np.hstack([lowerGradient, passedGradient])
                passedLowerGradient += passedLowerGradientAdd

        lowerGradient = np.hstack([lowerGradient, passedLowerGradient / len(self.inputLayer.nodes)])

        return gradient, lowerGradient







class Layer:
    def __init__(self, inputLayer : 'Layer', layerNum : int, layerSize: int, layerType: int = 0) -> None:
        self.layerType : int = layerType
        self.layerIndex : int = layerNum
        self.nodes : list[Node] = []
        self.inputLayer : Layer = inputLayer

        # Standard nodeType 
        for i in range(layerSize):
            if self.layerIndex == 0:
                self.nodes.append(Node(layerNum=self.layerIndex, inputLayer=inputLayer, weights=np.ndarray(0), bias=0))
                continue
            # Inits every node with their inputs and a random weight vector, with values between 0 and 1
            randomWeights : np.array = (np.random.rand(len(inputLayer.nodes)) - 0.5) * 2
            randomBias : float = (np.random.rand() - 0.5) * 2
            self.nodes.append(Node(layerNum=self.layerIndex, inputLayer=inputLayer, weights=randomWeights, bias=randomBias))

        # TODO: Add other layertypes like convolutional layers

    
    def computeLayerActivation(self) -> None:
        """
        Computes activation of every node in the layer.
        """
        for node in self.nodes:
            node.computeActivation()







class NeuralNetwork:
    def __init__(self, layerDescription : list[dict[str]]) -> None:
        """
        The first layer is the inputlayer, and the last layer will be the outputlayer.
        Example of layerDescription:
            layerEx = [
                {
                    "layerSize" : 10,               # parameter for deciding the amount of nodes in the layer
                    "layerType" : 0                 # Only a single layer type exists right now, so just use 0 (Convolutional layers will be added later)
                }, 
                {
                    "layerSize" : 10,               # parameter for deciding the amount of nodes in the layer
                    "layerType" : 0                 # Only a single layer type exists right now, so just use 0 (Convolutional layers will be added later)
                }, 
                    ...
            ]
        """
        self.amountOfLayers : int = len(layerDescription)
        self.layers : list[Layer] = []

        # Inits every layer described in layerDescription
        i : int = 0
        for layer in layerDescription:
            if i == 0: # Defines first layer as input layer
                self.layers.append(Layer(inputLayer=[], layerNum=i, layerSize=layer["layerSize"], layerType=layer["layerType"]))
            else:
                self.layers.append(Layer(inputLayer=self.layers[i-1], layerNum=i, layerSize=layer["layerSize"], layerType=layer["layerType"]))
            i += 1


    def runNetwork(self) -> None:
        for i in range(1, self.amountOfLayers):
            self.layers[i].computeLayerActivation()


    def getOutput(self) -> np.array:
        return np.array([node.activation for node in self.layers[self.amountOfLayers-1].nodes], dtype=float)

    
    def setInput(self, input: list[float]) -> None:
        """
        Make sure the network is setup correctly to take the correct amount of inputs.
        """
        if len(self.layers[0].nodes) != len(input):
            raise ValueError("Input parameter does not match the size of the input layer of the Neural Network.")

        i : int = 0
        for node in self.layers[0].nodes:
            node.activation = input[i]
            i += 1


    def trainNetworkGradDec(self, trainingInput: np.array, expectedOutputMatrix: np.array, iterationNum : int) -> None:
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
            for element in range(partitionSize-1):
                nextRandomElement : int = np.random.randint(0, len(trainingInput[0])-1)
                randomInputs = np.hstack([randomInputs, trainingInput[:, nextRandomElement] [:, np.newaxis]])
                randomOutputs = np.hstack([randomOutputs, expectedOutputMatrix[:, nextRandomElement] [:, np.newaxis]])

            for inputIndex in range(len(randomInputs)):
                self.setInput(input=randomInputs[:, inputIndex])
                self.runNetwork()
                expectedOutput : np.array = randomOutputs[:, inputIndex]

                gradientAdd : np.array = np.ndarray(0)
                passedLowerGradient : np.array = np.ndarray(0)

                # Train on the given data
                for outputNode in range(len(self.layers[-1].nodes)):
                    # C = activation - expectedValue
                    # Derivative of costFunction in relation to a (sigmoid of the sum)
                    if len(gradientAdd) == 0:
                        passedGradient, passedLowerGradient = [gradientPart for gradientPart in self.layers[-1].nodes[outputNode].gradDecend(derivCvrtA= 2*(self.layers[-1].nodes[outputNode].activation - expectedOutput[outputNode]))]
                        gradientAdd = np.hstack([gradientAdd, passedGradient])
                    else:
                        passedGradient, passedLowerGradientAdd = [gradientPart for gradientPart in self.layers[-1].nodes[outputNode].gradDecend(derivCvrtA= 2*(self.layers[-1].nodes[outputNode].activation - expectedOutput[outputNode]))]
                        gradientAdd = np.hstack([gradientAdd, passedGradient])
                        passedLowerGradient += passedLowerGradientAdd
                
                gradientAdd = np.hstack([gradientAdd, passedLowerGradient / len(self.layers[-1].nodes)])
                gradient += gradientAdd

        # The gradient is now calculated

        # Determine and print the length of the gradient
        lenOfGrad = np.linalg.norm(gradient)
        print("GradientLength: ", lenOfGrad)

        #! Determine other ways to downscale the effect of the gradient, when we get closer to a local minima
        negativeGradient = -gradient * 0.1 / (1 + 0.1 * iterationNum)

        # Applies the negative gradient to every weight and bias in the network
        i : int = 0
        for layerIndex in reversed(range(1, len(self.layers))):
            for nodeIndex, node in enumerate(self.layers[layerIndex].nodes):
                self.layers[layerIndex].nodes[nodeIndex].bias += negativeGradient[i]
                i += 1
                for weightIndex, weight in enumerate(node.weights):
                    self.layers[layerIndex].nodes[nodeIndex].weights[weightIndex] += negativeGradient[i]
                    i += 1
