import os
import csv
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from enum import Enum
from PIL import Image
from collections import Counter

class GraphNode:
    def __init__(self,x,y,label):
        self.x = x
        self.y = y
        self.label = label

    def GetDistance(self,node):
        x_diff = self.x - node.x
        y_diff = self.y - node.y
        distance = math.sqrt((x_diff*x_diff)+(y_diff*y_diff))

        return distance

class ImageSolver:
    def __init__(self, problemType):
        self.ptype=problemType
        self.problemImages = dict()
        self.solutionImages = dict()
        self.xOrDict = dict()
        self.graph = dict()

    def AddImages(self, problem,solutions):
        for p in problem:
            im = Image.open(problem[p])
            self.problemImages[p] = np.array(im.getdata())

        for s in solutions:
            im = Image.open(solutions[s])
            self.solutionImages[s] = np.array(im.getdata())

    def GetDifferences(self):
        if self.ptype=="2x2":
            self.xOrDict['AB'] = self.problemImages['A'] ^ self.problemImages['B']
            self.xOrDict['C1'] = self.problemImages['C'] ^ self.solutionImages['1']
            self.xOrDict['C2'] = self.problemImages['C'] ^ self.solutionImages['2']
            self.xOrDict['C3'] = self.problemImages['C'] ^ self.solutionImages['3']
            self.xOrDict['C4'] = self.problemImages['C'] ^ self.solutionImages['4']
            self.xOrDict['C5'] = self.problemImages['C'] ^ self.solutionImages['5']
            self.xOrDict['C6'] = self.problemImages['C'] ^ self.solutionImages['6']
            self.xOrDict['AC'] = self.problemImages['A'] ^ self.problemImages['C']
            self.xOrDict['B1'] = self.problemImages['B'] ^ self.solutionImages['1']
            self.xOrDict['B2'] = self.problemImages['B'] ^ self.solutionImages['2']
            self.xOrDict['B3'] = self.problemImages['B'] ^ self.solutionImages['3']
            self.xOrDict['B4'] = self.problemImages['B'] ^ self.solutionImages['4']
            self.xOrDict['B5'] = self.problemImages['B'] ^ self.solutionImages['5']
            self.xOrDict['B6'] = self.problemImages['B'] ^ self.solutionImages['6']

        elif self.ptype=="3x3":
            #row 1
            self.xOrDict['AB'] = self.problemImages['A'] ^ self.problemImages['B']
            self.xOrDict['BC'] = self.problemImages['B'] ^ self.problemImages['C']
            #row 2
            self.xOrDict['DE'] = self.problemImages['D'] ^ self.problemImages['E']
            self.xOrDict['EF'] = self.problemImages['E'] ^ self.problemImages['F']
            #row3
            self.xOrDict['GH'] = self.problemImages['G'] ^ self.problemImages['H']
            self.xOrDict['H1'] = self.problemImages['H'] ^ self.solutionImages['1']
            self.xOrDict['H2'] = self.problemImages['H'] ^ self.solutionImages['2']
            self.xOrDict['H3'] = self.problemImages['H'] ^ self.solutionImages['3']
            self.xOrDict['H4'] = self.problemImages['H'] ^ self.solutionImages['4']
            self.xOrDict['H5'] = self.problemImages['H'] ^ self.solutionImages['5']
            self.xOrDict['H6'] = self.problemImages['H'] ^ self.solutionImages['6']
            #column 1
            self.xOrDict['AD'] = self.problemImages['A'] ^ self.problemImages['D']
            self.xOrDict['DG'] = self.problemImages['D'] ^ self.problemImages['G']
            #column 2
            self.xOrDict['BE'] = self.problemImages['B'] ^ self.problemImages['E']
            self.xOrDict['EH'] = self.problemImages['E'] ^ self.problemImages['H']
            #column 3
            self.xOrDict['CF'] = self.problemImages['C'] ^ self.problemImages['F']
            self.xOrDict['F1'] = self.problemImages['F'] ^ self.solutionImages['1']
            self.xOrDict['F2'] = self.problemImages['F'] ^ self.solutionImages['2']
            self.xOrDict['F3'] = self.problemImages['F'] ^ self.solutionImages['3']
            self.xOrDict['F4'] = self.problemImages['F'] ^ self.solutionImages['4']
            self.xOrDict['F5'] = self.problemImages['F'] ^ self.solutionImages['5']
            self.xOrDict['F6'] = self.problemImages['F'] ^ self.solutionImages['6']
            #diagonal
            self.xOrDict['AE'] = self.problemImages['A'] ^ self.problemImages['E']
            self.xOrDict['E1'] = self.problemImages['E'] ^ self.solutionImages['1']
            self.xOrDict['E2'] = self.problemImages['E'] ^ self.solutionImages['2']
            self.xOrDict['E3'] = self.problemImages['E'] ^ self.solutionImages['3']
            self.xOrDict['E4'] = self.problemImages['E'] ^ self.solutionImages['4']
            self.xOrDict['E5'] = self.problemImages['E'] ^ self.solutionImages['5']
            self.xOrDict['E6'] = self.problemImages['E'] ^ self.solutionImages['6']
    
    def CreateGraph(self, doPlot=False):
        for x in self.xOrDict:
           mean = np.mean(self.xOrDict[x])
#            y = np.var(self.xOrDict[x])
           y = np.std(self.xOrDict[x])
           self.graph[x] = GraphNode(mean,y,x)

        if doPlot == True:
            xdata = list()
            ydata = list()
            labels = list()
            for s, node in self.graph:
                xdata.append(node.x)
                ydata.append(node.y)
                labels.append(node.label)
            plt.scatter(xdata,ydata,marker='o')
            for label, x, y in zip(labels, xdata, ydata):
                plt.annotate(label, xy=(x, y), textcoords='data')
            plt.show()

    def GetDistances(self):
        self.distances = dict()
        if self.ptype=="2x2":
            self.distances['AB'] = dict()
            self.distances['AB']['1']=self.graph['AB'].GetDistance(self.graph['C1'])
            self.distances['AB']['2']=self.graph['AB'].GetDistance(self.graph['C2'])
            self.distances['AB']['3']=self.graph['AB'].GetDistance(self.graph['C3'])
            self.distances['AB']['4']=self.graph['AB'].GetDistance(self.graph['C4'])
            self.distances['AB']['5']=self.graph['AB'].GetDistance(self.graph['C5'])
            self.distances['AB']['6']=self.graph['AB'].GetDistance(self.graph['C6'])

            self.distances['AC'] = dict()
            self.distances['AC']['1']=self.graph['AC'].GetDistance(self.graph['B1'])
            self.distances['AC']['2']=self.graph['AC'].GetDistance(self.graph['B2'])
            self.distances['AC']['3']=self.graph['AC'].GetDistance(self.graph['B3'])
            self.distances['AC']['4']=self.graph['AC'].GetDistance(self.graph['B4'])
            self.distances['AC']['5']=self.graph['AC'].GetDistance(self.graph['B5'])
            self.distances['AC']['6']=self.graph['AC'].GetDistance(self.graph['B6'])


        elif self.ptype=="3x3":
            self.distances['AB'] = dict()
            self.distances['AB']['1']=self.graph['AB'].GetDistance(self.graph['C1'])
            self.distances['AB']['2']=self.graph['AB'].GetDistance(self.graph['C2'])
            self.distances['AB']['3']=self.graph['AB'].GetDistance(self.graph['C3'])
            self.distances['AB']['4']=self.graph['AB'].GetDistance(self.graph['C4'])
            self.distances['AB']['5']=self.graph['AB'].GetDistance(self.graph['C5'])
            self.distances['AB']['6']=self.graph['AB'].GetDistance(self.graph['C6'])
