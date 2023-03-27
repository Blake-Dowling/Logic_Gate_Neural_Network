from tkinter import *
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import draw_network

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
##############################Neural Learning Algorithm##############################
##########################################################################################
weight1 = 0
weight2 = 0
bias = 0
INC_AMOUNT = 0.1
guessOutput = [0, 0, 0, 0]
def testParams(weight1, weight2, bias):
    totalSE = 0
    MSE = 0
    for i in range(len(inputData)):
        input1  = inputData[i][0]
        input2 = inputData[i][1]
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

##############################Animation Loop##############################
while True:
    window.update()
