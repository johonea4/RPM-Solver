import os
import csv
import numpy as np
import time
import math
from PIL import Image
import Clusters

class Features_t():
    def __init__(self):
        self.name = ""
        self.numComponentsDifference=0
        self.imageMeans=[]
        self.imageStd=[]
        self.xoredMean=0.0
        self.xoredStd=0.0
        self.noredMean=0.0
        self.noredStd=0.0
        self.clusterCenters = []
        self.clusterDensities = []
        self.clusterDistributions = []
        self.xoredCenters = []
        self.xoredDensities = []
        self.xoredDistributions = []
        self.noredCenters = []
        self.noredDensities = []
        self.noredDistributions = []


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
        self.imageInverted = dict()
        self.imageMM = dict()

        #Data used for Image differences
        self.xored = dict()
        self.nored = dict()
        self.noredMM = dict()
        self.xoredMM = dict()

        #data combined into a feature set
        self.features = dict()

        #############################################################
        #---------------------Create Images--------------------------
        #############################################################
        for p in figures:
            if p.isalpha():
                self.problemFiles[p] = figures[p].visualFilename
            elif p.isdigit():
                self.solutionFiles[p] = figures[p].visualFilename
        for p in self.problemFiles:
            im = Image.open(self.problemFiles[p])
            im = im.convert('1')
            self.problemImages[p] = np.array(im.getdata())
            self.problemImages[p][self.problemImages[p]>0] = 1
            self.imageInverted[p] = np.logical_not(self.problemImages[p]).astype(int)
        for s in self.solutionFiles:
            im = Image.open(self.solutionFiles[s])
            im = im.convert('1')
            self.solutionImages[s] = np.array(im.getdata())
            self.solutionImages[s][self.solutionImages[s]>0] = 1
            self.imageInverted[s] = np.logical_not(self.solutionImages[s]).astype(int)

        self.numAnswers = len(self.solutionFiles)
        #############################################################
        #----------------------matrix helpers------------------------
        #############################################################
        self.problemType = ptype
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
        self.__BuildFeatureSummary__()

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
        for p in self.imageInverted:
            self.imageMM[p] = Clusters.GetClusters(self.imageInverted[p], (184,184))
        for x in self.xored:
            self.xoredMM[x] = Clusters.GetClusters(self.xored[x],(184,184))
        for n in self.nored:
            self.noredMM[n] = Clusters.GetClusters(self.nored[n],(184,184))



    def __BuildFeatureSummary__(self):
        for x in self.xored:
            tmp = Features_t()
            tmp.name = x
            
            tmp.xoredMean = np.mean(self.xored[x])
            tmp.xoredStd = np.std(self.xored[x])
            
            tmp.noredMean = np.mean(self.nored[x])
            tmp.noredMean = np.std(self.nored[x])
            
            tmp.imageMeans.append(np.mean(self.imageInverted[x[0]]))
            tmp.imageMeans.append(np.mean(self.imageInverted[x[1]]))
            tmp.imageStd.append(np.std(self.imageInverted[x[0]]))
            tmp.imageStd.append(np.std(self.imageInverted[x[1]]))
            
            tmp.clusterCenters.append(None if self.imageMM[x[0]]==None else self.imageMM[x[0]].means)
            tmp.clusterCenters.append(None if self.imageMM[x[1]]==None else self.imageMM[x[1]].means)
            tmp.clusterDensities.append(None if self.imageMM[x[0]]==None else [ np.size(self.imageMM[x[0]].segments[i],0) for i in range(0,self.imageMM[x[0]].components)])
            tmp.clusterDensities.append(None if self.imageMM[x[1]]==None else [ np.size(self.imageMM[x[1]].segments[i],0) for i in range(0,self.imageMM[x[1]].components)])
            tmp.clusterDistributions.append(None if self.imageMM[x[0]]==None else [ np.std(self.imageMM[x[0]].segments[i],0) for i in range(0,self.imageMM[x[0]].components)])
            tmp.clusterDistributions.append(None if self.imageMM[x[1]]==None else [ np.std(self.imageMM[x[1]].segments[i],0) for i in range(0,self.imageMM[x[1]].components)])

            tmp.xoredCenters.append(None if self.xoredMM[x]==None else self.xoredMM[x].means)
            tmp.xoredDensities.append(None if self.xoredMM[x]==None else [ np.size(self.xoredMM[x].segments[i],0) for i in range(0,self.xoredMM[x].components)])
            tmp.xoredDistributions.append(None if self.xoredMM[x]==None else [ np.std(self.xoredMM[x].segments[i],0) for i in range(0,self.xoredMM[x].components)])

            tmp.noredCenters.append(None if self.noredMM[x]==None else self.noredMM[x].means)
            tmp.noredDensities.append(None if self.noredMM[x]==None else [ np.size(self.noredMM[x].segments[i],0) for i in range(0,self.noredMM[x].components)])
            tmp.noredDistributions.append(None if self.noredMM[x]==None else [ np.std(self.noredMM[x].segments[i],0) for i in range(0,self.noredMM[x].components)])

            c1 = 0 if self.imageMM[x[0]]==None else self.imageMM[x[0]].components
            c2 = 0 if self.imageMM[x[1]]==None else self.imageMM[x[1]].components
            tmp.numComponentsDifference = math.fabs(c1 - c2)

            self.features[x] = tmp

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
