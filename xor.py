from tkinter import *
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

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



class Node:
    def __init__(self, x, y, l):
        self.location = (x, y)
        self.object = canvas.create_oval(x * CELL_SIZE, 
                                    y * CELL_SIZE, 
                                    (x + 1) * CELL_SIZE, 
                                    (y + 1) * CELL_SIZE, 
                                    fill = "lime",
                                    outline = "blue")
        self.label = canvas.create_text((x + 0.5) * CELL_SIZE, 
                                    (y + 0.5) * CELL_SIZE, 
                                    text = l,
                                    fill = "deeppink")
##############################Relu function (num)##############################
def relu(x):
    return max(0, x)
##############################Relu function (2D vector)##############################
def relu2DVector(vv):
    for v in vv: #for each vector in a 2D vector
        np.copyto(v, np.array(list(map(relu, v)))) #Apply relu(num) to each element and
        #copy new vector to original 2D vector
    return vv
def displayData(xLocation, dataVector, label):
    ##############################Data Label##############################
    canvas.create_text((xLocation + 0.5) * CELL_SIZE, 
                                    (1 + 0.5) * CELL_SIZE, 
                                    text = label,
                                    fill = "deeppink")
    ##############################Draw Nodes##############################
    for i in range(0, len(dataVector)):
        node = Node(xLocation, 2 + i, str(dataVector[i]))


##############################Input Data##############################
inputData = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
displayData(2, inputData, "Input Data")

##############################W##############################
weight2DVector = np.array([[1, 1], [1, 1]])
##############################c##############################
biasVector = np.array([0, -1])

##############################Hidden Vector h##############################
hiddenVector1 = np.dot(inputData, weight2DVector) #X * W
hiddenVector1 = np.add(hiddenVector1, biasVector) #(X * W) + c
hiddenVector1 = relu2DVector(hiddenVector1) #relu((X * W) + c)
displayData(4, hiddenVector1, "Hidden Vector")

##############################Output Data##############################
##############################w##############################
weightVector = np.array([1, -2])
outputVector = np.dot(hiddenVector1, weightVector)
displayData(6, outputVector, "Output Data")
##############################Animation Loop##############################
while True:
    window.update()
