from tkinter import *
import numpy as np
CELL_SIZE = 40
def setBrightness(brightness, rgb):
    labelIsScalar = True
    try: 
        brightness = float(brightness)
    except ValueError:
        labelIsScalar = False
    if labelIsScalar:
        rgb = np.array(rgb)
        rgb = np.multiply(rgb, max(0.0, min(1.0, brightness)))
        rgb = list(map(lambda x : int(x), rgb))
    rgb = f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
    return rgb
class Node:
    def __init__(self, canvas, x, y, l):
        self.location = (x, y)
        color = "lime"
        color = setBrightness(l, [50, 205, 50])
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
        newNodes = [] #To add to network's list
        ##############################Data Label##############################
        titleObj = self.canvas.create_text((self.xLocation + 0.5) * CELL_SIZE, 
                                        (1 + 0.5) * CELL_SIZE, 
                                        text = self.label,
                                        fill = "deeppink")
        createdObjs.append(titleObj)
        ##############################Draw Nodes##############################
        for i in range(0, len(self.dataVector)):
            node = Node(self.canvas, self.xLocation, 2 + i, str(self.dataVector[i]))
            newNodes.append(node)
            createdObjs.append(node.object)
            createdObjs.append(node.label)
        self.objs.extend(createdObjs)
        self.canvas.update()
        return newNodes
class Network:
    def __init__(self, canvas, numLayers):
        self.canvas = canvas
        self.layers = [] #Layers of neural network for displaying
        self.nodes = []
        self.connections = []
        for i in range(numLayers):
            self.layers.append(Layer(canvas, 4*i + 2, [], "Layer " + str(i)))
        self.createConnections()
    def updateLayerData(self, layer, newData):
        self.layers[layer].dataVector = newData
    def createConnections(self):
        for node in self.nodes:
            for otherNode in self.nodes:
                self.canvas.create_line(node.location[0], 
                                        node.location[1], 
                                        otherNode.location[0], 
                                        otherNode.location[1],
                                        color = "white")
                print(node.location)
    def drawNetwork(self):
        self.nodes = []
        for layer in self.layers:
            newNodes = layer.displayData()
            self.nodes.extend(newNodes)
        


   