from tkinter import *
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import draw_network
import embed_plot
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

def generateXor(weightVector, biasVector):
    ##############################Hidden Vector h##############################
    hiddenVector1 = np.dot(inputData, weight2DVector) #X * W
    hiddenVector1 = np.add(hiddenVector1, biasVector) #(X * W) + c
    hiddenVector1 = relu2DVector(hiddenVector1) #relu((X * W) + c)
    #draw_network.displayData(canvas, 4, hiddenVector1, "Hidden Vector")
    ##############################Output Data##############################
    outputVector = np.dot(hiddenVector1, weightVector)
    #draw_network.displayData(canvas, 6, outputVector, "Output Data")
    return outputVector
##############################Input Data##############################
inputData = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    #draw_network.displayData(canvas, 2, inputData, "Input Data")
##############################W##############################
weight2DVector = np.array([[1, 1], [1, 1]])
##############################w##############################
weightVector = np.array([1, -2]) #Correct weights
##############################c##############################
biasVector = np.array([0, -1]) #Correct bias
outputVector = generateXor(weightVector, biasVector)



def getInputVector():
    return inputData
def getOutputVector():
    return outputVector