from tkinter import *
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import draw_network
import embed_plot
import xor_neural
import math


##############################Constants##############################
WIDTH_IN_CELLS = 16
CELL_SIZE = 40
class Gate:
    def __init__(self, name, input, output):
        self.name = name
        self.input = input
        self.output = output
GATES = {"NOT": Gate("NOT", [0,1], [1,0]),
         "AND": Gate("AND", [[0,0],[0,1],[1,0],[1,1]], [0,0,0,1]),
         "OR": Gate("OR", [[0,0],[0,1],[1,0],[1,1]], [0,1,1,1]),
         "NAND": Gate("NAND", [[0,0],[0,1],[1,0],[1,1]], [1,1,1,0]),
         "XOR": Gate("XOR", [[0,0],[0,1],[1,0],[1,1]], [0,1,1,0]),
         "NOR": Gate("NOR", [[0,0],[0,1],[1,0],[1,1]], [1,0,0,0]),
         "XNOR": Gate("XNOR", [[0,0],[0,1],[1,0],[1,1]], [1,0,0,1])}


##############################Initialize Window and Main Canvas##############################
window = Tk()
window.resizable(False, False)
canvas = Canvas(window, 
                bg = "black", 
                highlightbackground = "blue",
                highlightthickness = 1,
                width = WIDTH_IN_CELLS * CELL_SIZE, 
                height = WIDTH_IN_CELLS * CELL_SIZE)
canvas.pack()
##############################Initialize Graph and Function Displays##############################
canvas.create_text((WIDTH_IN_CELLS/2)*CELL_SIZE, 0.5*CELL_SIZE, text = "Network")
canvas.create_text(((WIDTH_IN_CELLS/2)+5)*CELL_SIZE, 10.25*CELL_SIZE, text = "Output")
guessPlot = embed_plot.PlotObj(window, ((WIDTH_IN_CELLS/2)+2.5)*CELL_SIZE, 10.5*CELL_SIZE, 2, [], [])
canvas.create_text(((WIDTH_IN_CELLS/2)-3)*CELL_SIZE, 12.5*CELL_SIZE, text = "State")
stateLabel = canvas.create_text((0.5)*CELL_SIZE, 14*CELL_SIZE, text = "", anchor = "w")
##############################Dropdown Menu##############################
INPUT_OPTIONS = ["NOT", "AND", "OR", "NAND", "XOR", "NOR", "XNOR"]
currentGate = StringVar(window)
currentGate.set("AND")
#currentGate.trace("w", changeGate)
menu = OptionMenu(window, currentGate, *INPUT_OPTIONS)
menu.pack()
##############################Relu function (num)##############################
def relu(x):
    return max(0, x)
##############################Relu function (2D vector)##############################
def relu2DVector(vv):
    for v in vv: #for each vector in a 2D vector
        np.copyto(v, np.array(list(map(relu, v)))) #Apply relu(num) to each element and
        #copy new vector to original 2D vector
    return vv

##############################Calculates (MSE), Used as phi for Neural Network##############################
def meanSquaredError(output, expected):
    MSE = 0
    ##############################Calculate Error##############################
    errorVector = np.subtract(output, expected)
    ##############################Calculate Squared Error##############################
    errorVector = np.square(errorVector)
    ##############################Calculate Mean Squared Error##############################
    MSE = np.sum(errorVector) / len(output)
    return MSE
def linearizeInput(input):
    try:
        inputWidth = np.shape(input)[1]
    except IndexError:
            return input
    except  TypeError:
            return input
    ##############################Identity Function for Linearizing Input##############################
    id2 = np.full((inputWidth, inputWidth), 1)
    ##############################Linearized 'Sum' Vector##############################
    ##############################E.g. [[0,0][0,1][1,0][1,1]].[[1,1][1,1]]->[[0,0][1,1][1,1][2,2]]##############################
    return np.dot(input, id2)
##############################Display Current State of Neural Network##############################
def displayState(neural, output):
    ##############################X Axis of Network's Graph##############################
    linearized = linearizeInput(neural.input)
    ##############################Draw Current Neural Network every 1 second##############################
    if((time.time()) % 1 < 0.05):
        ##############################Update Output Layer Display##############################
        neural.layers[3] = np.around(output, 2)
        network.updateNeural(neural) #May not even need to update
        ##############################Draw Current Feed Function##############################
        stateText = "Relu(" + str(linearized) + " + " + \
            str(np.around(neural.params.biases, 2)) + ") * " + str(np.around(neural.params.weights, 2)) + " = " + str(np.around(output, 2))
        canvas.itemconfig(stateLabel, text = stateText)
    ##############################Draw Current Output on Graph##############################
    guessPlot.addLine(linearized, output)
    
    window.update()

##############################Object Containing Vectors for Weights and Biases##############################
class Params:
    ##############################Initialize Parameters##############################
    def __init__(self):
        self.weights = np.array([0.1, 0.1])
        self.biases = np.array([0.1, 0.1])
    ##############################Return Both Parameter Vectors as##############################
    ##############################One Vector##############################
    def paramVector(self):
        return np.concatenate([self.weights, self.biases])
    ##############################Revert Parameter Vector Into Tuple##############################
    ##############################Of Weight and Bias Vectors##############################
    def separateVector(self, paramVector):
        weights = np.copy(paramVector[0:len(self.weights)])
        biases = np.copy(paramVector[len(self.weights):len(paramVector)])
        return (weights, biases)
    ##############################Update Weight and Bias Vectors##############################
    def updateParams(self, paramVector):
        self.weights, self.biases = self.separateVector(paramVector)
    ##############################Print Param Object Weight and Bias Vectors##############################
    def printParams(self):
        print("weights: " + str(self.weights))
        print("biases: " + str(self.biases))
##############################Object Representing an Entire Neural Network##############################
class Neural:
    ##############################Constructor##############################
    def __init__(self, input, expected, lossFunction, numLayers):
        self.input = input #Input fed into neural network
        self.expected = expected #Expected Output used for training
        self.lossFunction = lossFunction #Loss function used to calculate phi
        self.params = Params() #Vectors containing weights and biases
        ##############################Amount by Which Weights and Biases are Altered##############################
        self.INC_AMOUNT = 0.05 #Precision with which network learns
        self.layers = [0 for i in range(numLayers)]
        self.layers[0] = np.around(input, 2)
    ##############################Produce Output Given Current Parameters##############################
    def feed(self, input):
        ##############################Linearize Input##############################
        linearized = linearizeInput(input)
        if((time.time()) % 1 < 0.05):
            self.layers[1] = np.around(linearized, 2)
        ##############################Add Bias##############################
        biasAdded = np.add(linearized, self.params.biases)
        ##############################Apply Activation Function##############################
        reluApplied = relu2DVector(biasAdded)
        ##############################Update Hidden Layer Display##############################
        if((time.time()) % 1 < 0.05):
            self.layers[2] = np.around(reluApplied, 2)
        ##############################Dot Product Weights##############################
        output = np.dot(reluApplied,  self.params.weights)
        ##############################Output Result##############################
        return output

    ##############################Adjust Parameter (Single Iteration) Using Loss Function##############################
    def train(self, incrementAmount):
        ##############################Convert Network's Weight and Bias Vectors##############################
        ##############################Into Single Vector##############################
        paramVector = self.params.paramVector()
        ##############################Create a Parameter Vector for Each##############################
        ##############################Test Adjustment##############################
        testVectors = []
        ##############################For Each Parameter, Create Test Parameter##############################
        ##############################Set with Parameter Increased and Decreased##############################
        ##############################By Passed Increment Amount##############################
        for i in range(len(paramVector)):
            ##############################Increase Parameter at i##############################
            testVector = np.copy(paramVector) #Vector of parameters to test with i increased
            testVector[i] = paramVector[i] + incrementAmount
            testVectors.append(testVector)
            ##############################Decrease Parameter at i##############################
            testVector = np.copy(paramVector) #Vector of parameters to test with i decreased
            testVector[i] = paramVector[i] - incrementAmount
            testVectors.append(testVector)
        ##############################Vector to Store Loss Function Outputs##############################
        ##############################For Parameter Sets Corresponding by Indices##############################
        lossVector = []
        ##############################Test Each Parameter Set##############################
        ##############################(Calculate Loss Function Output)##############################
        for testVector in testVectors:
            ##############################Adjust Neural Network Params to##############################
            ##############################Test Params##############################
            self.params.updateParams(testVector)
            ##############################Feed Neural Network##############################
            output = self.feed(self.input)
            ##############################Update Neural Network State Display##############################
            displayState(self, output)
            ##############################Calculate Loss Function Output##############################
            ##############################For Current Test State##############################
            loss = self.lossFunction(output, self.expected)
            ##############################Add Test Loss to Array##############################
            lossVector.append(loss)
        ##############################Back Propagation (Adjust Parameters##############################
        ##############################to Test Parameters Resulting in Optimal Loss)##############################
        optimalParameters = testVectors[np.argmin(lossVector)]
        self.params.updateParams(optimalParameters)
        ##############################Return Optimal Loss Value##############################
        phi = np.min(lossVector)
        return phi
            
PHI_TARGET = 0.001
##############################Initialize Input##############################
inputData = np.array([[0,0],[0,1],[1,0],[1,1]])
##############################Initialize Training Output##############################
expected = np.array([0,1,1,0])
##############################Create Neural Network Object##############################
neural = Neural(inputData, expected, meanSquaredError, 4)
network = draw_network.Network(canvas, neural, 4)
##############################Train Neural Network to Specified Phi Target##############################
while neural.train(neural.INC_AMOUNT) > PHI_TARGET:
    if((time.time()) % 1 < 0.1):
        neural.input = GATES[currentGate.get()].input
        neural.expected = GATES[currentGate.get()].output


##############################Keep window open after training algorithm completes##############################
while True:
    window.update()
