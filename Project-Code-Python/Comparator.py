import numpy as np
from VisualProcessor import VisualProcessor

class Solutions():
    closestXored = -1
    closestNored = -1
    closestImage = -1
    closestXoredClusters = -1
    closestNoredClusters = -1
    closestImageClusters = -1

class GraphNode:
    def __init__(self,label,*args):
        self.points = list()
        self.numPoints = len(args)
        for i in range(0,self.numPoints):
            self.points.append(args[i])
        self.label = label

    def GetDistance(self,node):
        ss=0
        for i in range(0,self.numPoints):
            diff = self.points[i] - node.points[i]
            ss += diff*diff
        distance = math.sqrt(ss)

        return distance

class Comparator(object):
    def __init__(self, visualProcessor):
        ###############################################
        #### Get Information from Visual Processor ####
        ###############################################
        self.featuresList=visualProcessor.features
        self.problemType = visualProcessor.problemType
        self.helperMatrix = visualProcessor.matrix
        self.numColumns = visualProcessor.numColumns
        self.numRows = visualProcessor.numRows
        self.numAnswers = visualProcessor.numAnswers

        ###############################################
        ################### VARIABLES #################
        ###############################################
        self.xoredNodes = dict()
        self.noredNodes = dict()
        self.ImageNodes = dict()
        self.xoredClusterNodes = dict()
        self.noredClusterNodes = dict()
        self.imageClusterNodes = dict()
        self.xoredDensitiesNodes = dict()
        self.noredDensitiesNodes = dict()
        self.imageDensitiesNodes = dict()
        self.xoredDeviationNodes = dict()
        self.noredDeviationNodes = dict()
        self.imageDeviationNodes = dict()

    def CreateGraphNodes(self):
        """
        Create the nodes for each comparison type:
        xored: (mean, std)
        nored: (mean, std)
        Image: (meanDiff, stdDiff)
        xoredClusters: (QI Centers, Q2 Centers, Q3 Centers, Q4 Centers)
        noredClusters: (QI Centers, Q2 Centers, Q3 Centers, Q4 Centers)
        imageClusters: (QI Centers, Q2 Centers, Q3 Centers, Q4 Centers)
        xoredDensities: (QI Densities, Q2 Densities, Q3 Densities, Q4 Densities)
        noredDensities: (QI Densities, Q2 Densities, Q3 Densities, Q4 Densities)
        imageDensities: (QI Densities, Q2 Densities, Q3 Densities, Q4 Densities)
        xoredDeviations: (QI Deviations, Q2 Deviations, Q3 Deviations, Q4 Deviations)
        noredDeviations: (QI Deviations, Q2 Deviations, Q3 Deviations, Q4 Deviations)
        imageDeviations: (QI Deviations, Q2 Deviations, Q3 Deviations, Q4 Deviations)
        """

    def GetSolutions(self):
        """
        if ptype == 2x2
        [AB] is to [C#]
        [AC] is to [B#]

        if ptype == 3x3
        [AB] to [BC] is to [DE] to [EF] is to [GH] to [H#]
        [AD] to [DG] is to [BE] to [EH] is to [CF] to [F#]
        """

