import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class PlotObj:
    def __init__(self, window, xLocation, yLocation, size, xData, yData):
        self.lines = []
        self.fig = Figure(figsize = (size, size), dpi = 100, facecolor = "black", edgecolor = "blue")
        self.plot = self.fig.add_subplot(111, facecolor = "black")

        self.plot.set_xlim(left = -2, right = 2)
        self.plot.set_ylim(bottom = -2, top = 2)

        self.plot.spines["right"].set_color("blue")
        self.plot.spines["bottom"].set_color("blue")
        self.plot.spines["left"].set_color("blue")
        self.plot.spines["top"].set_color("blue")

        self.plot.tick_params(color = "blue", labelcolor = "blue")
        #plot.xaxis.label.set_color("blue")

        self.line = self.plot.plot(xData, yData, ".", color = "lime")

        self.plotCanvas = FigureCanvasTkAgg(self.fig, master = window)
        self.plotCanvas.draw()
        self.plotCanvas.get_tk_widget().place(x = xLocation, y = yLocation)
        #return (self.plot, self.plotCanvas)
    def addLine(self, xData, yData):
        for line in self.lines:
            line.remove()
            
        #print(self.plotCanvas.get_tk_widget().find_all())
        self.lines = []
        newLine = self.plot.plot(xData, yData, ".", color = "lime")
        self.plotCanvas.draw()
        self.lines.extend(newLine)