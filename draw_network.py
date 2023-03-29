from tkinter import *
import numpy as np
CELL_SIZE = 40

class Node:
    def __init__(self, canvas, x, y, l):
        self.location = (x, y)
        color = "lime"
        labelIsScalar = True
        try: 
            l = float(l)
        except ValueError:
            labelIsScalar = False
        if labelIsScalar:
            rgb = np.array([50, 205, 50])
            print("tick")
            color = np.multiply(rgb, max(0.0, min(1.0, l)))
            color = list(map(lambda x : int(x), color))
            color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
        self.object = canvas.create_oval(x * CELL_SIZE, 
                                        y * CELL_SIZE, 
                                        (x + 1) * CELL_SIZE, 
                                        (y + 1) * CELL_SIZE, 
                                        fill = color,
                                        outline = "blue",)
        self.label = canvas.create_text((x + 0.5) * CELL_SIZE, 
                                        (y + 0.5) * CELL_SIZE, 
                                        text = str(l),
                                        fill = "deeppink")
class Layer:
    def __init__(self, canvas, xLocation, dataVector, label):
        self.objs = []          
        self.canvas = canvas
        self.xLocation = xLocation 
        self.dataVector = dataVector
        self.label = label 
    def deleteObjs(self):
        for obj in self.objs:
            self.canvas.delete(obj)
        self.objs = []
    def displayData(self):
        self.deleteObjs()
        createdObjs = []
        ##############################Data Label##############################
        titleObj = self.canvas.create_text((self.xLocation + 0.5) * CELL_SIZE, 
                                        (1 + 0.5) * CELL_SIZE, 
                                        text = self.label,
                                        fill = "deeppink")
        createdObjs.append(titleObj)
        ##############################Draw Nodes##############################
        for i in range(0, len(self.dataVector)):
            node = Node(self.canvas, self.xLocation, 2 + i, str(self.dataVector[i]))
            createdObjs.append(node.object)
            createdObjs.append(node.label)
        self.objs.extend(createdObjs)
        self.canvas.update()
class Network:
    def __init__(self, canvas, numLayers):
        self.layers = [] #Layers of neural network for displaying
        for i in range(numLayers):
            self.layers.append(Layer(canvas, 4*i + 2, [], "Layer " + str(i)))
    def updateLayerData(self, layer, newData):
        self.layers[layer].dataVector = newData
    def drawNetwork(self):
        for layer in self.layers:
            layer.displayData()

   