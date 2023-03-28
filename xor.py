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

##############################Constants##############################
WIDTH_IN_CELLS = 16
CELL_SIZE = 40
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
##############################Initialize Network Layer Displays##############################
inputLayer = draw_network.Layer(canvas, 2, [], "Input")
hiddenLayer = draw_network.Layer(canvas, 4, [], "Hidden")
outputLayer = draw_network.Layer(canvas, 6, [], "Output")
bestText = "Best Guess: "
bestGuessLabel = canvas.create_text(1*CELL_SIZE, 12*CELL_SIZE, text = bestText, anchor = "w")
guessPlot = embed_plot.PlotObj(window, 8*CELL_SIZE, 6*CELL_SIZE, 2, [], [])
##############################Relu function (num)##############################
def relu(x):
    return max(0, x)
##############################Relu function (2D vector)##############################
def relu2DVector(vv):
    for v in vv: #for each vector in a 2D vector
        np.copyto(v, np.array(list(map(relu, v)))) #Apply relu(num) to each element and
        #copy new vector to original 2D vector
    return vv
##########################################################################################
##############################Initialize Input and Expected Output Data##############################
##########################################################################################
#inputData = xor_neural.getInputVector() #Retrieve binary combination inputs generated in xor_neural.py

#outputVector = np.array([0, 1, 1, 1]) #Expected result: Retrieve xor distribution, given actual weights

#inputLayer.dataVector = inputData
#inputLayer.displayData() #Expected result: Retrieve xor distribution, given actual weights


##########################################################################################
##############################Neural Deep Learning Algorithm##############################
##########################################################################################


    


   
        ##########################################################################################
        ##############################Apply Loss Function (Least Mean Squared)##############################
        ##########################################################################################




##########################################################################################
##############################Calculate Phi (MSE) Using One Altered Weight or Bias##############################
##########################################################################################
def lossFunction(output, expected):
    MSE = 0
    ##########################################################################################
    ##############################Calculate Phi (Mean Squared Error)##############################
    ##########################################################################################
    ##############################Calculate Error##############################
    errorVector = np.subtract(output, expected)
    ##############################Calculate Squared Error##############################
    errorVector = np.square(errorVector)
    ##############################Calculate Mean Squared Error##############################
    MSE = np.sum(errorVector) / 4
    return MSE

def displayState(neural, output):
    ##############################Draw Current Neural Network##############################
    hiddenLayer.displayData()
    outputLayer.dataVector = np.around(output, 2)
    outputLayer.displayData()
    guessPlot.addLine(neural.sumVector, output)
    bestText = "Relu(" + str(neural.sumVector) + " + " + \
        str(neural.params.biases) + ") * " + str(neural.params.weights) + " = " + str(output)
    canvas.itemconfig(bestGuessLabel, text = bestText)
    window.update()



##########################################################################################
##############################Object Representing a (4,1)->(4, ) Neural Network##############################
##########################################################################################
class Params:
    def __init__(self):
        self.weights = np.array([0.0, 0.0])
        self.biases = np.array([0.0, 0.0])
    def paramVector(self):
        return np.concatenate([self.weights, self.biases])
    def separateVector(self, paramVector):
        weights = np.copy(paramVector[0:len(self.weights)])
        biases = np.copy(paramVector[len(self.weights):len(paramVector)])
        return (weights, biases)
    def updateParams(self, paramVector):
        self.weights, self.biases = self.separateVector(paramVector)
    def printParams(self):
        print("weights: " + str(self.weights))
        print("biases: " + str(self.biases))
class Neural:
    def __init__(self, input):
        self.input = input
        self.expected = expected
        self.bestGuess = np.full((4, ), 0)
        ##############################2x2 Identity Function for Linearizing Input##############################
        id2 = np.full((2, 2), 1)
        ##############################Linearized 'Sum' Vector##############################
        ##############################[[0,0][0,1][1,0][1,1]].[[1,1][1,1]]->[[0,0][1,1][1,1][2,2]]##############################
        self.params = Params()
        self.sumVector = np.dot(input, id2)
        ##############################Amount by Which Weights and Biases are Altered##############################
        self.INC_AMOUNT = 0.05
    

    def getOutput(self, input):
        ##########################################################################################
        ##############################Apply Learned Function (Using Guess Parameters)##############################
        ##########################################################################################
        ##############################Add Bias##############################
        biasAdded = np.add(input, self.params.biases)
        ##############################Apply Activation Function##############################
        reluApplied = relu2DVector(biasAdded)
        hiddenLayer.dataVector = reluApplied
        ##############################Dot Product Weights##############################
        output = np.dot(reluApplied,  self.params.weights)

        ##############################Output Guess##############################
        return output







    def optParams(self):
        paramVector = self.params.paramVector()

        testVectors = []
        for i in range(len(paramVector)):
            testVector = np.copy(paramVector)
            testVector[i] = paramVector[i] + self.INC_AMOUNT
            testVectors.append(testVector)
            testVector = np.copy(paramVector)
            testVector[i] = paramVector[i] - self.INC_AMOUNT
            testVectors.append(testVector)
        lossVector = []
        ##############################Test Each Parameter Set##############################
        for testVector in testVectors:
            self.params.updateParams(testVector)
            output = self.getOutput(self.input)
            #displayState(self, output)
            loss = lossFunction(output, expected)
            lossVector.append(loss)
        ##############################Back Propagation (Adjust Parameter Resulting Optimal Loss)##############################
        ##############################Determine Phi (Least MSE)##############################
        self.params.updateParams(testVectors[np.argmin(lossVector)])
        # self.bestGuess = self.testParams(self.sumVector, s)
        print("New Params: ")
        self.params.printParams()
        print("New Phi: " + str(np.min(lossVector)))
        return np.min(lossVector)
            
        
        
inputData = np.array([[0,0],[0,1],[1,0],[1,1]])
expected = np.array([0,1,1,0])
neural = Neural(inputData)
while neural.optParams() > 0.1:
    pass
#learn()
##############################Animation Loop##############################
while True:
    window.update()
