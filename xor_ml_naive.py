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
##############################Relu function (num)##############################
def relu(x):
    return max(0, x)
##############################Relu function (2D vector)##############################
def relu2DVector(vv):
    for v in vv: #for each vector in a 2D vector
        np.copyto(v, np.array(list(map(relu, v)))) #Apply relu(num) to each element and
        #copy new vector to original 2D vector
    return vv
if __name__ == "__main__":
    ##########################################################################################
    ##############################Input and Output Data##############################
    ##########################################################################################
    inputData = xor_neural.getInputVector() #Retrieve binary combination inputs generated in xor_neural.py
    draw_network.displayData(canvas, 2, inputData, "Input Data") #Draw actual neural network for xor
    outputVector = xor_neural.getOutputVector() #Expected result: Retrieve xor distribution, given actual weights
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
    learnNaive(inputData)