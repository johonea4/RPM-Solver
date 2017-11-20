import os
import csv
import numpy as np
import time
import math
from PIL import Image
import Clusters

class VisualProcessor:
    """description of class"""

    def __init__(self, figures, ptype):
        #############################################################
        #---------------------Class Variables------------------------
        #############################################################
        self.problemFiles = dict()
        self.solutionFiles = dict()
        self.solutionImages = dict()
        self.problemImages = dict()

        #Data used for GMM's
        self.solutionInverted = dict()
        self.problemInverted = dict()
        self.solutionMM = dict()
        self.problemMM = dict()

        #Data used for Image differences
        self.xored = dict()
        self.nored = dict()
        self.noredObjects = dict()
        self.xoredObjects = dict()

        #############################################################
        #---------------------Create Images--------------------------
        #############################################################
        for p in figures:
            if p >= 'A' and p <= 'H':
                self.problemFiles[p] = figures[p].visualFilename
            elif p >= '1' and p <= '8':
                self.solutionFiles[p] = figures[p].visualFilename
        for p in self.problemFiles:
            im = Image.open(self.problemFiles[p])
            im = im.convert('1')
            self.problemImages[p] = np.array(im.getdata())
            self.problemImages[p][self.problemImages[p]>0] = 1
            self.problemInverted[p] = np.logical_not(self.problemImages[p]).astype(int)
        for s in self.solutionFiles:
            im = Image.open(self.solutionFiles[s])
            im = im.convert('1')
            self.solutionImages[s] = np.array(im.getdata())
            self.solutionImages[s][self.solutionImages[s]>0] = 1
            self.solutionInverted[s] = np.logical_not(self.solutionImages[s]).astype(int)

        self.numAnswers = len(self.solutionFiles)
        #############################################################
        #----------------------matrix helpers------------------------
        #############################################################
        if ptype=="2x2":
            self.matrix = [ [ 'A', 'B'],
                            [ 'C', "#"] ]
            self.numRows = self.numColumns = 2
        else:
            self.matrix = [ [ 'A', 'B', 'C'],
                            [ 'D', 'E', 'F'],
                            [ 'G', 'H', "#"] ]
            self.numRows = self.numColumns = 3
        #############################################################
        #----------------------Get Combinations----------------------
        #############################################################
        self.__GetImageCombinations__()
        self.__GetGMM__()
        self.__BuildFeatureFrames__()

    def __Get_xor_And__(self,r1,c1,r2,c2):
        i1 = self.matrix[r1][c1]
        i2 = self.matrix[r2][c2]
        if i2 == '#':
            for i in range(1,self.numAnswers+1):
                self.xored[i1 + str(i)] = self.problemImages[i1] ^ self.solutionImages[str(i)]
                self.nored[i1 + str(i)] = self.problemImages[i1] | self.solutionImages[str(i)]
                self.nored[i1 + str(i)] = np.logical_not(self.nored[i1 + str(i)]).astype(int)
                
        else:
            self.xored[i1 + i2] = self.problemImages[i1] ^ self.problemImages[i2]
            self.nored[i1 + i2] = self.problemImages[i1] | self.problemImages[i2]
            self.nored[i1 + i2] = np.logical_not(self.nored[i1 + i2]).astype(int)

    def __GetImageCombinations__(self):
        for r in range(0,self.numRows):
            for c in range(0,self.numColumns-1):
                self.__Get_xor_And__(r,c,r,c+1)
        for c in range(0,self.numColumns):
            for r in range(0,self.numRows-1):
                 self.__Get_xor_And__(r,c,r+1,c)

    def __GetGMM__(self):
        for p in self.problemInverted:
            self.problemMM[p] = Clusters.GetClusters(self.problemInverted[p], (184,184))
            if self.problemMM[p] != None: print "Image: " + p + ";".join("("+str(i)+","+str(j)+")" for i,j in self.problemMM[p].means) + "\n"
            else: print "\tImage %s returned None. Image is most likely blank." %(p)
        for s in self.solutionInverted:
            self.solutionMM[s] = Clusters.GetClusters(self.solutionInverted[s],(184,184))
            if self.solutionMM[s] != None: print "Image: " + s + ";".join("("+str(i)+","+str(j)+")" for i,j in self.solutionMM[s].means) + "\n"
            else: print "\tImage %s GMM returned None. Image is most likely blank." %(s)

    def __BuildFeatureFrames__(self):
        return

    def OutputImageCombinations(self, problemName):
        path = os.path.join("CombinedImages",problemName)
        if not os.path.isdir(path):
            os.makedirs(path)
        for x in self.xored:
            imname = "xor_" + x + ".png"
            imname = os.path.join(path,imname)
            im = Image.new('1',(184,184))
            im.putdata(self.xored[x])
            im.save(imname)
        for x in self.nored:
            imname = "nor_" + x + ".png"
            imname = os.path.join(path,imname)
            im = Image.new('1',(184,184))
            im.putdata(self.nored[x])
            im.save(imname)
