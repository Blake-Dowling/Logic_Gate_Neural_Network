from tkinter import *
import numpy as np
#from xor import neural
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
        self.nodes = []   
        self.xLocation = xLocation 
        self.dataVector = dataVector
        self.label = label 
        self.canvas = canvas
    def drawLayer(self):
        createdObjs = []
        self.nodes = []
        ##############################Data Label##############################
        titleObj = self.canvas.create_text((self.xLocation + 0.5) * CELL_SIZE, 
                                        (1 + 0.5) * CELL_SIZE, 
                                        text = self.label,
                                        fill = "deeppink")
        createdObjs.append(titleObj)
        ##############################Draw Nodes##############################
        dataNotArray = True
        ##############################Test if data array of numbers##############################
        try:
            dataNum = int(self.dataVector[0])
        except ValueError:
            dataNotArray = False
        except  TypeError:
            dataNotArray = False
        for i in range(len(self.dataVector)):
            ##############################If data is array of numbers,##############################
            ##############################simply create node for each number##############################
            if dataNotArray:
                node = Node(self.canvas, self.xLocation, 2 + i, str(self.dataVector[i]))
                self.nodes.append(node)
                createdObjs.append(node.object)
                createdObjs.append(node.label)
            ##############################If data is array of arrays,##############################
            ##############################Iterate through each array##############################
            else:
                for j in range(len(self.dataVector[i])):
                    node = Node(self.canvas, self.xLocation, 
                                2 + (i * len(self.dataVector[i])) + j, 
                                str(self.dataVector[i][j]))
                    self.nodes.append(node)
                    createdObjs.append(node.object)
                    createdObjs.append(node.label)
        return createdObjs
class Network:
    def __init__(self, canvas, neural, numLayers):
        self.objs = []
        self.canvas = canvas
        self.neural = neural
        self.layers = [] #Layers of neural network for displaying
        
    # def updateLayerData(self, layer, newData):
    #     self.layers[layer].dataVector = newData
    def updateNeural(self, neural):
        self.eraseNetwork()
        self.neural = neural
        
        self.drawNetwork()
    def eraseNetwork(self):
        self.layers = []
        self.nodes = []
        for obj in self.objs:
            self.canvas.delete(obj)
        self.objs = []
    def drawNetwork(self):
        for i in range(len(self.neural.layers)):
            newLayer = Layer(self.canvas, 4*i + 2, self.neural.layers[i], "Layer " + str(i))
            self.layers.append(newLayer)
            self.objs.extend(newLayer.drawLayer())
        print(self.neural.params.biases)
        
        for i in range(len(self.layers[2].nodes)):
            node = self.layers[2].nodes[i]
            otherNode = self.layers[3].nodes[int(i/2)]
            color1 = setBrightness(self.neural.params.weights[int(i%2)], [255, 255, 255])
            
            newLine = self.canvas.create_line(node.location[0]*CELL_SIZE,
                                                node.location[1]*CELL_SIZE,
                                                otherNode.location[0]*CELL_SIZE,
                                                otherNode.location[1]*CELL_SIZE,
                                                fill = color1,
                                                width = 1)
            self.objs.append(newLine)



        # for i in range(len(self.layers)):
        #     if i + 1 < len(self.layers):
        #         for node in self.layers[i].nodes:
        #             for otherNode in self.layers[i+1].nodes:
        #                 newLine = self.canvas.create_line(node.location[0]*CELL_SIZE,
        #                                         node.location[1]*CELL_SIZE,
        #                                         otherNode.location[0]*CELL_SIZE,
        #                                         otherNode.location[1]*CELL_SIZE,
        #                                         fill = "white",
        #                                         width = 1)
        #                 self.objs.append(newLine)

        


   