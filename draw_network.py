from tkinter import *
import numpy as np
CELL_SIZE = 40

class Node:
    def __init__(self, canvas, x, y, l):
        self.location = (x, y)
        # color = np.array([(50.0, 205.0, 50.0)])
        # brightness = np.array(l)
        # brightness = np.sum(l)
        # color = np.multiply(color, l)
        # color = f'#{50:02x}{205:02x}{50:02x}'
        self.object = canvas.create_oval(x * CELL_SIZE, 
                                        y * CELL_SIZE, 
                                        (x + 1) * CELL_SIZE, 
                                        (y + 1) * CELL_SIZE, 
                                        fill = "lime",
                                        outline = "blue")
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

   