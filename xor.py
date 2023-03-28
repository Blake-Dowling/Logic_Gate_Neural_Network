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
##########################################################################################
##############################Input and Output Data##############################
##########################################################################################
inputData = xor_neural.getInputVector() #Retrieve binary combination inputs generated in xor_neural.py
draw_network.displayData(canvas, 2, inputData, "Input Data") #Draw actual neural network for xor
outputVector = xor_neural.getOutputVector() #Expected result: Retrieve xor distribution, given actual weights
##########################################################################################
##############################Neural Deep Learning Algorithm##############################
##########################################################################################
bestGuess = np.full((4,), 0) #Vector to store the current best guess for display
def learn():
    global bestGuess
    ##############################Display Input Data##############################
    #canvas.create_text(10*CELL_SIZE, 1*CELL_SIZE, text = "Initial Distribution")
    #embed_plot.embedPlot(window, 8*CELL_SIZE, 2*CELL_SIZE, 2, inputData[:, 0], inputData[:, 1])
    #canvas.create_text(3*CELL_SIZE, 7*CELL_SIZE, text = "Hidden Layer 1 (Dot Product to Linearize)")
    ##############################2x2 Identity Function for Linearizing Input##############################
    id2 = np.full((2, 2), 1)
    ##############################Linearized 'Sum' Vector##############################
    ##############################[[0,0][0,1][1,0][1,1]].[[1,1][1,1]]->[[0,0][1,1][1,1][2,2]]##############################
    sumVector = np.dot(inputData, id2)
    ##############################Bias Guess Vector Initialized##############################
    biasGuess = np.array([0.0, 0.0])
    ##############################Weight Guess Vector Initialized##############################
    weightGuess = np.array([0.0, 0.0])
    #canvas.create_text(3*CELL_SIZE, 7*CELL_SIZE, text = "Hidden Layer 2 (Bias)")
    #currentPlot, plotCanvas = embed_plot.embedPlot(window, 8*CELL_SIZE, 2*CELL_SIZE, 2, inputData[:, 0], inputData[:, 1])
    ##############################Amount by Which Weights and Biases are Altered##############################
    INC_AMOUNT = 0.05
    ##########################################################################################
    ##############################Calculate Phi (MSE) Using One Altered Weight or Bias##############################
    ##########################################################################################
    def testParams(sumVector, weightGuess, biasGuess):
        global bestGuess
        MSE = 0
        ##########################################################################################
        ##############################Apply Learned Function (Using Guess Parameters)##############################
        ##########################################################################################
        ##############################Add Bias##############################
        sumVector = np.add(sumVector, biasGuess)
        ##############################Apply Activation Function##############################
        sumVector = relu2DVector(sumVector)
        ##############################Dot Product Weights##############################
        sumVector = np.dot(sumVector,  weightGuess)
        bestGuess = np.copy(sumVector)
        ##########################################################################################
        ##############################Calculate Phi (Mean Squared Error)##############################
        ##########################################################################################
        ##############################Calculate Error##############################
        sumVector = np.subtract(sumVector, outputVector)
        ##############################Calculate Squared Error##############################
        sumVector = np.square(sumVector)
        ##############################Calculate Mean Squared Error##############################
        MSE = np.sum(sumVector) / 4
        return MSE
    ##############################Least MSE Per Iteration##############################
    loss = 1 
    ##########################################################################################
    ##############################Back Propagation Loop##############################
    ##########################################################################################
    while loss > 0:
        ##############################Current Output With Least MSE (Most Desirable Phi)##############################
        bestText = "Best Guess: " + str(bestGuess)
        bestGuessLabel = canvas.create_text(3*CELL_SIZE, 9*CELL_SIZE, text = bestText)
        ##############################Plot Best Output##############################
        currentPlot, plotCanvas = embed_plot.embedPlot(window, 8*CELL_SIZE, 6*CELL_SIZE, 2, sumVector, bestGuess)
        window.update()
        canvas.delete(bestGuessLabel)
        currentPlot.remove()
        ##############################Reset Best MSE Each Iteration##############################
        minLoss = 1
        ##########################################################################################
        ##############################Apply Loss Function (Least Mean Squared)##############################
        ##########################################################################################
        ##############################Calculate Loss with Increment Weight 1##############################
        weightGuessCopy = np.copy(weightGuess)
        weightGuess[0] = weightGuessCopy[0] + INC_AMOUNT
        incW1 = testParams(sumVector, weightGuess, biasGuess)
        ##############################Calculate Loss with Decrement Weight 1##############################
        weightGuess[0] = weightGuessCopy[0] - INC_AMOUNT
        decW1  = testParams(sumVector, weightGuess, biasGuess)
        ##############################Calculate Loss with Increment Weight 2##############################
        weightGuess = np.copy(weightGuessCopy)
        weightGuess[1] = weightGuessCopy[1] + INC_AMOUNT
        incW2 = testParams(sumVector, weightGuess, biasGuess)
        ##############################Calculate Loss with Decrement Weight 2##############################
        weightGuess[1] = weightGuessCopy[1] - INC_AMOUNT
        decW2  = testParams(sumVector, weightGuess, biasGuess)
        ##############################Calculate Loss with Increment Bias 1##############################
        weightGuess = np.copy(weightGuessCopy)
        biasGuessCopy = np.copy(biasGuess)
        biasGuess[0] = biasGuessCopy[0] + INC_AMOUNT
        incB1 = testParams(sumVector, weightGuess, biasGuess)
        ##############################Calculate Loss with Decrement Bias 1##############################
        biasGuess[0] = biasGuessCopy[0] - INC_AMOUNT
        decB1  = testParams(sumVector,weightGuess, biasGuess)
        ##############################Calculate Loss with Increment Bias 2##############################
        biasGuess = np.copy(biasGuessCopy)
        biasGuess[1] = biasGuessCopy[1] + INC_AMOUNT
        incB2 = testParams(sumVector, weightGuess, biasGuess)
        ##############################Calculate Loss with Decrement Bias 2##############################
        biasGuess[1] = biasGuessCopy[1] - INC_AMOUNT
        decB2  = testParams(sumVector,weightGuess, biasGuess)
        biasGuess = np.copy(biasGuessCopy)
        ##############################Determine Phi (Least MSE)##############################
        minLoss = min(incW1, decW1, incW2, decW2, incB1, decB1, incB2, decB2)
        ##########################################################################################
        ##############################Back Propagation (Adjust Parameter Resulting Optimal Loss)##############################
        ##########################################################################################
        if minLoss == incW1:
            weightGuess[0] = weightGuess[0] + INC_AMOUNT
        elif minLoss == decW1:
            weightGuess[0] = weightGuess[0] - INC_AMOUNT
        elif minLoss == incW2:
            weightGuess[1] = weightGuess[1] + INC_AMOUNT
        elif minLoss == decW2:
            weightGuess[1] = weightGuess[1] - INC_AMOUNT
        elif minLoss == incB1:
            biasGuess[0] = biasGuess[0] + INC_AMOUNT
        elif minLoss == decB1:
            biasGuess[0] = biasGuess[0] - INC_AMOUNT
        elif minLoss == incB2:
            biasGuess[1] = biasGuess[1] + INC_AMOUNT
        elif minLoss == decB2:
            biasGuess[1] = biasGuess[1] - INC_AMOUNT
learn()
##############################Animation Loop##############################
while True:
    window.update()
