from tkinter import *
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import draw_network
import embed_plot

WIDTH_IN_CELLS = 16
CELL_SIZE = 40

window = Tk()
window.resizable(False, False)
canvas = Canvas(window, 
                bg = "black", 
                highlightbackground = "blue",
                highlightthickness = 1,
                width = WIDTH_IN_CELLS * CELL_SIZE, 
                height = WIDTH_IN_CELLS * CELL_SIZE)
canvas.pack()


##########################################################################################
##############################Manual Algorithm##############################
##########################################################################################
##############################Relu function (num)##############################
def relu(x):
    return max(0, x)
##############################Relu function (2D vector)##############################
def relu2DVector(vv):
    for v in vv: #for each vector in a 2D vector
        np.copyto(v, np.array(list(map(relu, v)))) #Apply relu(num) to each element and
        #copy new vector to original 2D vector
    return vv



##############################Input Data##############################
inputData = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
draw_network.displayData(canvas, 2, inputData, "Input Data")

##############################W##############################
weight2DVector = np.array([[1, 1], [1, 1]])
##############################c##############################
biasVector = np.array([0, -1])

##############################Hidden Vector h##############################
hiddenVector1 = np.dot(inputData, weight2DVector) #X * W
hiddenVector1 = np.add(hiddenVector1, biasVector) #(X * W) + c
hiddenVector1 = relu2DVector(hiddenVector1) #relu((X * W) + c)
draw_network.displayData(canvas, 4, hiddenVector1, "Hidden Vector")

##############################Output Data##############################
##############################w##############################
weightVector = np.array([1, -2])
outputVector = np.dot(hiddenVector1, weightVector)
draw_network.displayData(canvas, 6, outputVector, "Output Data")

##########################################################################################
##############################Naive Neural Learning Algorithm##############################
##########################################################################################
#embed_plot.embedPlot(window, 1*CELL_SIZE, 8*CELL_SIZE, 2, sumVector[:, 0], sumVector[:, 1])
def learnNaive(inputVector):
    weight1 = 0
    weight2 = 0
    bias = 0
    INC_AMOUNT = 0.1
    guessOutput = [0, 0, 0, 0]
    def testParams(weight1, weight2, bias):
        totalSE = 0
        MSE = 0
        for i in range(len(inputVector)):
            input1  = inputVector[i][0]
            input2 = inputVector[i][1]
            output = weight1 * input1 + weight2 * input2 + bias
            guessOutput[i] = output
            expected = outputVector[i]
            SE = (output - expected) **2
            totalSE = totalSE + SE
            MSE = totalSE / (i + 1)
        #print(str(weight1) + ", " + str(weight2) + ", " + str(bias) + ": " + str(MSE))
        return MSE
    leastMSE = 1
    while leastMSE > 0:
        leastMSE = 1
        incW1 = testParams(weight1 + INC_AMOUNT, weight2, bias)
        decW1  = testParams(weight1 - INC_AMOUNT, weight2, bias)
        incW2 = testParams(weight1, weight2 + INC_AMOUNT, bias)
        decW2  = testParams(weight1, weight2 - INC_AMOUNT, bias)
        incB = testParams(weight1, weight2, bias + INC_AMOUNT)
        decB  = testParams(weight1, weight2, bias - INC_AMOUNT)
        leastMSE = min(incW1, decW1, incW2, decW2, incB, decB)
        if leastMSE == incW1:
            weight1 = weight1 + INC_AMOUNT
        elif leastMSE == decW1:
            weight1 = weight1 - INC_AMOUNT
        elif leastMSE == incW2:
            weight2 = weight2 + INC_AMOUNT
        elif leastMSE == decW2:
            weight2 = weight2 - INC_AMOUNT
        elif leastMSE == incB:
            bias = bias + INC_AMOUNT
        elif leastMSE == decB:
            bias = bias - INC_AMOUNT
        print(str(weight1) + ", " + str(weight2) + ", " + str(bias) + ": " + str(leastMSE) + ", " + str(guessOutput))
#learnNaive(inputData)
##########################################################################################
##############################Neural Learning Algorithm##############################
##########################################################################################
bestGuess = np.full((4,), 0)
def learn(inputVector):
    global bestGuess
    canvas.create_text(10*CELL_SIZE, 1*CELL_SIZE, text = "Initial Distribution")
    embed_plot.embedPlot(window, 8*CELL_SIZE, 2*CELL_SIZE, 2, inputData[:, 0], inputData[:, 1])
    canvas.create_text(3*CELL_SIZE, 7*CELL_SIZE, text = "Hidden Layer 1 (Dot Product to Linearize)")
    id2 = np.full((2, 2), 1)
    ##############################Linearized 'Sum' Vector##############################
    sumVector = np.dot(inputData, id2)
    ##############################Bias Vector##############################
    biasGuess = np.array([0.0, 0.0])
    ##############################Weight Vector##############################
    weightGuess = np.array([0.0, 0.0])
    canvas.create_text(3*CELL_SIZE, 7*CELL_SIZE, text = "Hidden Layer 2 (Bias)")
    currentPlot, plotCanvas = embed_plot.embedPlot(window, 8*CELL_SIZE, 2*CELL_SIZE, 2, inputData[:, 0], inputData[:, 1])
    INC_AMOUNT = 0.05
    
    def testParams(sumVector, weightGuess, biasGuess):
        global bestGuess
        totalSE = 0
        MSE = 0
        ##############################Add Bias##############################
        #print("Original linearized vector: " + str(sumVector))
        sumVector = np.add(sumVector, biasGuess)
        #print("Bias Added: " + str(sumVector))
        sumVector = relu2DVector(sumVector)
        #print("Relu: " + str(sumVector))
        sumVector = np.dot(sumVector,  weightGuess)
        #print("Weight Multiplied: " + str(sumVector))
        print("Bias Guess: " + str(biasGuess))
        print("Weight Guess: " + str(weightGuess))
        bestGuess = np.copy(sumVector)
        sumVector = np.subtract(sumVector, outputVector)
        #print("Subtracted Actual: " + str(sumVector))
        sumVector = np.square(sumVector)
        #print("Squared error: " + str(sumVector))
        MSE = np.sum(sumVector) / 4
        #print("Mean squared error: " + str(MSE))
        return MSE
    leastMSE = 1
    while leastMSE > 0:
        bestText = "Best Guess: " + str(bestGuess)
        bestGuessLabel = canvas.create_text(3*CELL_SIZE, 9*CELL_SIZE, text = bestText)
        currentPlot, plotCanvas = embed_plot.embedPlot(window, 8*CELL_SIZE, 6*CELL_SIZE, 2, sumVector, bestGuess)
        window.update()
        canvas.delete(bestGuessLabel)
        currentPlot.remove()
        leastMSE = 1
        ##############################Guess Increment Weight 1##############################
        weightGuessCopy = np.copy(weightGuess)
        weightGuess[0] = weightGuessCopy[0] + INC_AMOUNT
        incW1 = testParams(sumVector, weightGuess, biasGuess)
        ##############################Guess Decrement Weight 1##############################
        weightGuess[0] = weightGuessCopy[0] - INC_AMOUNT
        decW1  = testParams(sumVector, weightGuess, biasGuess)
        ##############################Guess Increment Weight 2##############################
        weightGuess = np.copy(weightGuessCopy)
        weightGuess[1] = weightGuessCopy[1] + INC_AMOUNT
        incW2 = testParams(sumVector, weightGuess, biasGuess)
        ##############################Guess Decrement Weight 2##############################
        weightGuess[1] = weightGuessCopy[1] - INC_AMOUNT
        decW2  = testParams(sumVector, weightGuess, biasGuess)
        ##############################Guess Increment Bias 1##############################
        weightGuess = np.copy(weightGuessCopy)
        biasGuessCopy = np.copy(biasGuess)
        biasGuess[0] = biasGuessCopy[0] + INC_AMOUNT
        incB1 = testParams(sumVector, weightGuess, biasGuess)
        ##############################Guess Decrement Bias 1##############################
        biasGuess[0] = biasGuessCopy[0] - INC_AMOUNT
        decB1  = testParams(sumVector,weightGuess, biasGuess)
        ##############################Guess Increment Bias 2##############################
        biasGuess = np.copy(biasGuessCopy)
        biasGuess[1] = biasGuessCopy[1] + INC_AMOUNT
        incB2 = testParams(sumVector, weightGuess, biasGuess)
        ##############################Guess Decrement Bias 2##############################
        biasGuess[1] = biasGuessCopy[1] - INC_AMOUNT
        decB2  = testParams(sumVector,weightGuess, biasGuess)
        biasGuess = np.copy(biasGuessCopy)
        ##############################Determine Phi (Least MSE)##############################
        leastMSE = min(incW1, decW1, incW2, decW2, incB1, decB1, incB2, decB2)
        if leastMSE == incW1:
            weightGuess[0] = weightGuess[0] + INC_AMOUNT
        elif leastMSE == decW1:
            weightGuess[0] = weightGuess[0] - INC_AMOUNT
        elif leastMSE == incW2:
            weightGuess[1] = weightGuess[1] + INC_AMOUNT
        elif leastMSE == decW2:
            weightGuess[1] = weightGuess[1] - INC_AMOUNT
        elif leastMSE == incB1:
            biasGuess[0] = biasGuess[0] + INC_AMOUNT
        elif leastMSE == decB1:
            biasGuess[0] = biasGuess[0] - INC_AMOUNT
        elif leastMSE == incB2:
            biasGuess[1] = biasGuess[1] + INC_AMOUNT
        elif leastMSE == decB2:
            biasGuess[1] = biasGuess[1] - INC_AMOUNT
        # print(str(weight1) + ", " + str(weight2) + ", " + str(bias) + ": " + str(leastMSE) + ", " + str(guessOutput))
learn(inputData)

##############################Animation Loop##############################
while True:
    window.update()
