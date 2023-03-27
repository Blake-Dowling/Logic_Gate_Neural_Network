from tkinter import *
CELL_SIZE = 40

class Node:
    def __init__(self, canvas, x, y, l):
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
            
def displayData(canvas, xLocation, dataVector, label):
    ##############################Data Label##############################
    canvas.create_text((xLocation + 0.5) * CELL_SIZE, 
                                    (1 + 0.5) * CELL_SIZE, 
                                    text = label,
                                    fill = "deeppink")
    ##############################Draw Nodes##############################
    for i in range(0, len(dataVector)):
        node = Node(canvas, xLocation, 2 + i, str(dataVector[i]))