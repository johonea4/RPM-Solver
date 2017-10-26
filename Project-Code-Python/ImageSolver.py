import os
import csv
import numpy as np
import time
import math
from PIL import Image

class GraphNode:
    def __init__(self,label,p1,p2,p3=0,p4=0):
        self.points = list()
        self.points.append(p1)
        self.points.append(p2)
        self.points.append(p3)
        self.points.append(p4)
        self.label = label

    def GetDistance(self,node):
        ss=0
        for i in range(0,len(self.points)):
            diff = self.points[i] - node.points[i]
            ss += diff*diff
        distance = math.sqrt(ss)

        return distance

class ImageSolver:
    def __init__(self, problemType):
        self.ptype=problemType
        self.problemImages = dict()
        self.solutionImages = dict()
        self.xOrDict = dict()
        self.andDict = dict()
        self.graph = dict()
        self.matrix_2x2 = [ [ 'A', 'B'],
                            [ 'C', "#"] ]
        self.matrix_3x3 = [ [ 'A', 'B', 'C'],
                            [ 'D', 'E', 'F'],
                            [ 'G', 'H', "#"] ]
        self.compare_2x2 = [ ['AB', 'C'], ['AC', 'B'] ]
        self.compare_3x3 = [ ['AB', 'BC'], ['DE', 'EF'], ['GH', 'H'],
                             ['AD', 'DG'], ['BE', 'EH'], ['CF', 'F'] ]

    def AddImages(self, problem,solutions):
        for p in problem:
            im = Image.open(problem[p])
            im = im.convert('1')
            self.problemImages[p] = np.array(im.getdata())

        for s in solutions:
            im = Image.open(solutions[s])
            im = im.convert('1')
            self.solutionImages[s] = np.array(im.getdata())

    def __Get_xor_And__(self,r1,c1,r2,c2,m):
        i1 = m[r1][c1]
        i2 = m[r2][c2]
        if i2 == '#':
            for i in range(1,7):
                self.xOrDict[i1 + str(i)] = self.problemImages[i1] ^ self.solutionImages[str(i)]
                self.andDict[i1 + str(i)] = self.problemImages[i1] & self.solutionImages[str(i)]                       
        else:
            self.xOrDict[i1 + i2] = self.problemImages[i1] ^ self.problemImages[i2]
            self.andDict[i1 + i2] = self.problemImages[i1] & self.problemImages[i2]                       


    def GetDifferences(self):
        nr = nc = 0
        if self.ptype=="2x2":
            m = self.matrix_2x2
            nr = nc = 2
        else:
            m = self.matrix_3x3
            nr = nc = 3

        for r in range(0,nr):
            for c in range(0,nc-1):
                self.__Get_xor_And__(r,c,r,c+1,m)
        for c in range(0,nc):
            for r in range(0,nr-1):
                 self.__Get_xor_And__(r,c,r+1,c,m)
   
    def CreateGraph(self):
        for label in self.xOrDict:
           p1 = np.mean(self.xOrDict[label])
           p2 = np.mean(self.andDict[label])
           p3 = np.std(self.xOrDict[label])
           p4 = np.std(self.andDict[label])
           sim1 = p2/(p2+(0.5*p1))
           sim2 = p4/(p4+(0.5*p3))
           self.graph[label] = GraphNode(label,sim1,sim2)

    def GetDistances(self):
        self.distances = dict()
        if self.ptype=="2x2":
            m = self.compare_2x2
        else:
            m = self.compare_3x3
        
        for comp in m:
            self.distances[comp[0]] = dict()
            if len(comp[1]) == 1:
                for i in range(1,7):
                    self.distances[comp[0]][str(i)] = self.graph[comp[0]].GetDistance(self.graph[comp[1]+str(i)])
            else:
                self.distances[comp[0]][comp[1]] = self.graph[comp[0]].GetDistance(self.graph[comp[1]])
               
    def GetAnswer(self):
        if self.ptype == "2x2":
            distanceTotal = [0] * 6
            for comp in self.distances:
                for n in self.distances[comp]:
                    distanceTotal[int(n)-1] += self.distances[comp][n] * self.distances[comp][n]
            distanceTotal = np.sqrt(distanceTotal)
            return (np.argmin(distanceTotal)+1)
        else:
            distanceTotal = [0] * 6
            dr1 = self.distances['AB']['BC']
            dr2 = self.distances['DE']['EF']
            dc1 = self.distances['AD']['DG']
            dc2 = self.distances['BE']['EH']

            for i in range(0,6):
                diff1 = dr1 - self.distances['GH'][str(i+1)]
                diff2 = dc1 - self.distances['CF'][str(i+1)]
                diff3 = dr2 - self.distances['GH'][str(i+1)]
                diff4 = dc2 - self.distances['CF'][str(i+1)]
                
                distanceTotal[i] = (diff1*diff1) + (diff2*diff2) + (diff3*diff3) + (diff4*diff4)
            distanceTotal = np.sqrt(distanceTotal)
            return (np.argmin(distanceTotal)+1)

        # distanceTotal  = list()
        # distanceVotes = [0] * 6
        # for comp in self.distances:
        #     tmp = [0] * 6
        #     for n in self.distances[comp]:
        #         tmp[int(n)-1] = self.distances[comp][n]
        #     distanceTotal.append(tmp)
        
        # for total in distanceTotal:
        #     distanceVotes[np.argmin(total)] += 1

        # #Now Check if there are multiple same number of votes
        # maxVotesIndex = np.argmax(distanceVotes)
        # maxVotes = distanceVotes[maxVotesIndex]

        # countMax = distanceVotes.count(maxVotes)
        # if countMax > 1:
        #     maxItems = [float('inf')] * 6
        #     for i in range(0,6):
        #         if distanceVotes[i] == maxVotes:
        #             for total in distanceTotal:
        #                 if total[i] < maxItems[i]:
        #                     maxItems[i] = total[i]
        #     maxVotesIndex = np.argmin(maxItems)
        # return (maxVotesIndex+1)