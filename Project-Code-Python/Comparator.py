import numpy as np
import math
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
        self.problemType = visualProcessor.problemType

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
        image: (meanDiff, stdDiff)
        xoredClusters: (QI Centers, Q2 Centers, Q3 Centers, Q4 Centers)
        noredClusters: (QI Centers, Q2 Centers, Q3 Centers, Q4 Centers)
        imageClusters: (QI CentersDiff, Q2 CentersDiff, Q3 CentersDiff, Q4 CentersDiff)
        xoredDensities: (QI Densities, Q2 Densities, Q3 Densities, Q4 Densities)
        noredDensities: (QI Densities, Q2 Densities, Q3 Densities, Q4 Densities)
        imageDensities: (QI DensitiesDiff, Q2 DensitiesDiff, Q3 DensitiesDiff, Q4 DensitiesDiff)
        xoredDeviations: (QI Deviations, Q2 Deviations, Q3 Deviations, Q4 Deviations)
        noredDeviations: (QI Deviations, Q2 Deviations, Q3 Deviations, Q4 Deviations)
        imageDeviations: (QI DeviationsDiff, Q2 DeviationsDiff, Q3 DeviationsDiff, Q4 DeviationsDiff)
        """

        for f in self.featuresList:
            #Means and Std nodes
            self.xoredNodes[f] = GraphNode(f, self.featuresList[f].xoredMean, self.featuresList[f].xoredStd)
            self.noredNodes[f] = GraphNode(f, self.featuresList[f].noredMean, self.featuresList[f].noredStd)
            meanDiff = self.featuresList[f].imageMeans[0] - self.featuresList[f].imageMeans[1]
            stdDiff = self.featuresList[f].imageStd[0] - self.featuresList[f].imageStd[1]
            self.ImageNodes[f] = GraphNode(f, math.fabs(meanDiff),math.fabs(stdDiff))

            #xored Clusters
            q1,q2,q3,q4 = self.__GetQuadrantData__(f,'xored')
            self.xoredClusterNodes[f] = GraphNode(f,q1['centers'],q2['centers'],q3['centers'],q4['centers'])
            self.xoredDeviationNodes[f] = GraphNode(f,q1['std'],q2['std'],q3['std'],q4['std'])
            self.xoredDensitiesNodes[f] = GraphNode(f,q1['densities'],q2['densities'],q3['densities'],q4['densities'])

            #nored Clusters
            q1,q2,q3,q4 = self.__GetQuadrantData__(f,'nored')
            self.noredClusterNodes[f] = GraphNode(f,q1['centers'],q2['centers'],q3['centers'],q4['centers'])
            self.noredDeviationNodes[f] = GraphNode(f,q1['std'],q2['std'],q3['std'],q4['std'])
            self.noredDensitiesNodes[f] = GraphNode(f,q1['densities'],q2['densities'],q3['densities'],q4['densities'])

            #iamge Clusters
            q1,q2,q3,q4 = self.__GetQuadrantData__(f,'image')
            self.imageClusterNodes[f] = GraphNode(f,q1['centers'],q2['centers'],q3['centers'],q4['centers'])
            self.imageDeviationNodes[f] = GraphNode(f,q1['std'],q2['std'],q3['std'],q4['std'])
            self.imageDensitiesNodes[f] = GraphNode(f,q1['densities'],q2['densities'],q3['densities'],q4['densities'])


    def __GetQuadrantData__(self,f, type):
        """
        184
        |----------------|---------------|
        |     Q2         |       Q1      |
      92|----------------|---------------|
        |     Q3         |       Q4      |
        |----------------|---------------|
        0                92             184
        """
        q1=dict();q2=dict();q3=dict();q4=dict();q12=dict();q22=dict();q32=dict();q42=dict();
        q1i=None;q2i=None;q3i=None;q4i=None;
        q1i2=None;q2i2=None;q3i2=None;q4i2=None
        #Get Indexes of centers for each quadrant
        arr = None
        arr2 = None
        if type == 'xored':
            arr = self.featuresList[f].xoredCenters[0]
        elif type == 'nored':
            arr = self.featuresList[f].noredCenters[0]
        elif type == 'image':
            arr = self.featuresList[f].clusterCenters[0]
            arr2 = self.featuresList[f].clusterCenters[1]
        if arr is not None and len(arr)>0:
            q1i = [ i for i in range(0,len(arr)) if arr[i][0]>=92 and arr[i][1]>=92 and arr[i][0]<184 and arr[i][1]<184 ]
            q2i = [ i for i in range(0,len(arr)) if arr[i][0]>=0 and arr[i][1]>=92 and arr[i][0]<92 and arr[i][1]<184 ]
            q3i = [ i for i in range(0,len(arr)) if arr[i][0]>=0 and arr[i][1]>=0 and arr[i][0]<92 and arr[i][1]<92 ]
            q4i = [ i for i in range(0,len(arr)) if arr[i][0]>=92 and arr[i][1]>=0 and arr[i][0]<184 and arr[i][1]<92 ]
        
        if arr2 is not None and len(arr2) > 0:
            q1i2 = [ i for i in range(0,len(arr2)) if arr2[i][0]>=92 and arr2[i][1]>=92 and arr2[i][0]<184 and arr2[i][1]<184 ]
            q2i2 = [ i for i in range(0,len(arr2)) if arr2[i][0]>=0 and arr2[i][1]>=92 and arr2[i][0]<92 and arr2[i][1]<184 ]
            q3i2 = [ i for i in range(0,len(arr2)) if arr2[i][0]>=0 and arr2[i][1]>=0 and arr2[i][0]<92 and arr2[i][1]<92 ]
            q4i2 = [ i for i in range(0,len(arr2)) if arr2[i][0]>=92 and arr2[i][1]>=0 and arr2[i][0]<184 and arr2[i][1]<92 ]

        if type == 'xored':
            centers = np.array(self.featuresList[f].xoredCenters[0])
            stds = np.array(self.featuresList[f].xoredDistributions[0])
            dens = np.array(self.featuresList[f].xoredDensities[0])
        elif type == 'nored':
            centers = np.array(self.featuresList[f].noredCenters[0])
            stds = np.array(self.featuresList[f].noredDistributions[0])
            dens = np.array(self.featuresList[f].noredDensities[0])
        elif type == 'image':
            centers = np.array(self.featuresList[f].clusterCenters[0])
            stds = np.array(self.featuresList[f].clusterDistributions[0])
            dens = np.array(self.featuresList[f].clusterDensities[0])
            centers2 = np.array(self.featuresList[f].clusterCenters[1])
            stds2 = np.array(self.featuresList[f].clusterDistributions[1])
            dens2 = np.array(self.featuresList[f].clusterDensities[1])

        
        q1['centers'] = 0 if q1i==None or len(q1i)==0 else np.mean(np.sqrt(np.sum(np.square(centers[q1i]),0)))
        q1['std'] = 0 if q1i==None or len(q1i)==0 else np.mean(np.sqrt(np.sum(np.square(stds[q1i]),0)))
        q1['densities'] = 0 if q1i==None or len(q1i)==0 else np.mean(dens[q1i])
        q2['centers'] = 0 if q2i==None or len(q2i)==0 else np.mean(np.sqrt(np.sum(np.square(centers[q2i]),0)))
        q2['std'] = 0 if q2i==None or len(q2i)==0 else np.mean(np.sqrt(np.sum(np.square(stds[q2i]),0)))
        q2['densities'] = 0 if q2i==None or len(q2i)==0 else np.mean(dens[q2i])
        q3['centers'] = 0 if q3i==None or len(q3i)==0 else np.mean(np.sqrt(np.sum(np.square(centers[q3i]),0)))
        q3['std'] = 0 if q3i==None or len(q3i)==0 else np.mean(np.sqrt(np.sum(np.square(stds[q3i]),0)))
        q3['densities'] = 0 if q3i==None or len(q3i)==0 else np.mean(dens[q3i])
        q4['centers'] = 0 if q4i==None or len(q4i)==0 else np.mean(np.sqrt(np.sum(np.square(centers[q4i]),0)))
        q4['std'] = 0 if q4i==None or len(q4i)==0 else np.mean(np.sqrt(np.sum(np.square(stds[q4i]),0)))
        q4['densities'] = 0 if q4i==None or len(q4i)==0 else np.mean(dens[q4i])

        if arr2 is not None and len(arr2) > 0:
            q12['centers'] = 0 if q1i2==None or len(q1i2)==0 else np.mean(np.sqrt(np.sum(np.square(centers2[q1i2]),0)))
            q12['std'] = 0 if q1i2==None or len(q1i2)==0 else np.mean(np.sqrt(np.sum(np.square(stds2[q1i2]),0)))
            q12['densities'] = 0 if q1i2==None or len(q1i2)==0 else np.mean(dens2[q1i2])
            q22['centers'] = 0 if q2i2==None or len(q2i2)==0 else np.mean(np.sqrt(np.sum(np.square(centers2[q2i2]),0)))
            q22['std'] = 0 if q2i2==None or len(q2i2)==0 else np.mean(np.sqrt(np.sum(np.square(stds2[q2i2]),0)))
            q22['densities'] = 0 if q2i2==None or len(q2i2)==0 else np.mean(dens2[q2i2])
            q32['centers'] = 0 if q3i2==None or len(q3i2)==0 else np.mean(np.sqrt(np.sum(np.square(centers2[q3i2]),0)))
            q32['std'] = 0 if q3i2==None or len(q3i2)==0 else np.mean(np.sqrt(np.sum(np.square(stds2[q3i2]),0)))
            q32['densities'] = 0 if q3i2==None or len(q3i2)==0 else np.mean(dens2[q3i2])
            q42['centers'] = 0 if q4i2==None or len(q4i2)==0 else np.mean(np.sqrt(np.sum(np.square(centers2[q4i2]),0)))
            q42['std'] = 0 if q4i2==None or len(q4i2)==0 else np.mean(np.sqrt(np.sum(np.square(stds2[q4i2]),0)))
            q42['densities'] = 0 if q4i2==None or len(q4i2)==0 else np.mean(dens2[q4i2])

            q1['centers'] = math.fabs(q1['centers']-q12['centers'])
            q1['std'] = math.fabs(q1['std']-q12['std'])
            q1['densities'] = math.fabs(q1['densities']-q12['densities'])
            q2['centers'] = math.fabs(q2['centers']-q22['centers'])
            q2['std'] = math.fabs(q2['std']-q22['std'])
            q2['densities'] = math.fabs(q2['densities']-q22['densities'])
            q3['centers'] = math.fabs(q3['centers']-q32['centers'])
            q3['std'] = math.fabs(q3['std']-q32['std'])
            q3['densities'] = math.fabs(q3['densities']-q32['densities'])
            q4['centers'] = math.fabs(q4['centers']-q42['centers'])
            q4['std'] = math.fabs(q4['std']-q42['std'])
            q4['densities'] = math.fabs(q4['densities']-q42['densities'])

        return q1,q2,q3,q4

    def __GetDistances__(self,left,node, type):
        distances = []
        nodes = None
        if type == 'xored':
            nodes = self.xoredNodes
        elif type == 'nored':
            nodes = self.noredNodes
        elif type == 'image':
            nodes = self.ImageNodes
        elif type == 'xoredclusters':
            nodes = self.xoredClusterNodes
        elif type == 'xoredstd':
            nodes = self.xoredDeviationNodes
        elif type == 'xoreddens':
            nodes = self.xoredDensitiesNodes
        elif type == 'noredclusters':
            nodes = self.noredClusterNodes
        elif type == 'noredstd':
            nodes = self.noredDeviationNodes
        elif type == 'noreddens':
            nodes = self.noredDensitiesNodes
        elif type == 'imageclusters':
            nodes = self.imageClusterNodes
        elif type == 'imagestd':
            nodes = self.imageDeviationNodes
        elif type == 'imagedens':
            nodes = self.imageDensitiesNodes

        for i in range(0,self.numAnswers):
            distances.append(nodes[left+str(i+1)].GetDistance(node))

        return distances

    def GetSolutions(self):
        """
        if ptype == 2x2
        [AB] is to [C#]
        [AC] is to [B#]

        if ptype == 3x3
        [AB] to [BC] is to [DE] to [EF] is to [GH] to [H#]
        [AD] to [DG] is to [BE] to [EH] is to [CF] to [F#]
        """
        answers = dict()
        if self.problemType == '2x2':
            x1 = self.__GetDistances__('C',self.xoredNodes['AB'],'xored')
            x2 = self.__GetDistances__('B',self.xoredNodes['AC'],'xored')
            ans = np.sqrt(np.add(np.square(x1),np.square(x2)))
            idx = np.argmin(ans)
            answers['xored'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('C',self.noredNodes['AB'],'nored')
            x2 = self.__GetDistances__('B',self.noredNodes['AC'],'nored')
            ans = np.sqrt(np.add(np.square(x1),np.square(x2)))
            idx = np.argmin(ans)
            answers['nored'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('C',self.ImageNodes['AB'],'image')
            x2 = self.__GetDistances__('B',self.ImageNodes['AC'],'image')
            ans = np.sqrt(np.add(np.square(x1),np.square(x2)))
            idx = np.argmin(ans)
            answers['image'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('C',self.xoredClusterNodes['AB'],'xoredclusters')
            x2 = self.__GetDistances__('B',self.xoredClusterNodes['AC'],'xoredclusters')
            ans = np.sqrt(np.add(np.square(x1),np.square(x2)))
            idx = np.argmin(ans)
            answers['xoredClusters'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('C',self.xoredDeviationNodes['AB'],'xoredstd')
            x2 = self.__GetDistances__('B',self.xoredDeviationNodes['AC'],'xoredstd')
            ans = np.sqrt(np.add(np.square(x1),np.square(x2)))
            idx = np.argmin(ans)
            answers['xoredStd'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('C',self.xoredDensitiesNodes['AB'],'xoreddens')
            x2 = self.__GetDistances__('B',self.xoredDensitiesNodes['AC'],'xoreddens')
            ans = np.sqrt(np.add(np.square(x1),np.square(x2)))
            idx = np.argmin(ans)
            answers['xoredDensities'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('C',self.xoredClusterNodes['AB'],'noredclusters')
            x2 = self.__GetDistances__('B',self.xoredClusterNodes['AC'],'noredclusters')
            ans = np.sqrt(np.add(np.square(x1),np.square(x2)))
            idx = np.argmin(ans)
            answers['noredClusters'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('C',self.xoredDeviationNodes['AB'],'noredstd')
            x2 = self.__GetDistances__('B',self.xoredDeviationNodes['AC'],'noredstd')
            ans = np.sqrt(np.add(np.square(x1),np.square(x2)))
            idx = np.argmin(ans)
            answers['noredStd'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('C',self.xoredDensitiesNodes['AB'],'noreddens')
            x2 = self.__GetDistances__('B',self.xoredDensitiesNodes['AC'],'noreddens')
            ans = np.sqrt(np.add(np.square(x1),np.square(x2)))
            idx = np.argmin(ans)
            answers['noredDensities'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('C',self.xoredClusterNodes['AB'],'imageclusters')
            x2 = self.__GetDistances__('B',self.xoredClusterNodes['AC'],'imageclusters')
            ans = np.sqrt(np.add(np.square(x1),np.square(x2)))
            idx = np.argmin(ans)
            answers['imageClusters'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('C',self.xoredDeviationNodes['AB'],'imagestd')
            x2 = self.__GetDistances__('B',self.xoredDeviationNodes['AC'],'imagestd')
            ans = np.sqrt(np.add(np.square(x1),np.square(x2)))
            idx = np.argmin(ans)
            answers['imageStd'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('C',self.xoredDensitiesNodes['AB'],'imagedens')
            x2 = self.__GetDistances__('B',self.xoredDensitiesNodes['AC'],'imagedens')
            ans = np.sqrt(np.add(np.square(x1),np.square(x2)))
            idx = np.argmin(ans)
            answers['imageDensities'] = [idx,ans[idx]]
        else:
            x1 = self.__GetDistances__('H',self.xoredNodes['GH'],'xored')
            x2 = self.__GetDistances__('F',self.xoredNodes['CF'],'xored')
            dr1 = self.xoredNodes['AB'].GetDistance(self.xoredNodes['BC'])
            dr2 = self.xoredNodes['DE'].GetDistance(self.xoredNodes['EF'])
            dc1 = self.xoredNodes['AD'].GetDistance(self.xoredNodes['DG'])
            dc2 = self.xoredNodes['BE'].GetDistance(self.xoredNodes['EH'])
            diff1 = np.square(np.subtract(x1,dr1))
            diff2 = np.square(np.subtract(x1,dr2))
            diff3 = np.square(np.subtract(x2,dc1))
            diff4 = np.square(np.subtract(x2,dc2))
            ans = np.sqrt(np.add(np.add(np.add(diff1,diff2),diff3),diff4))
            idx = np.argmin(ans)
            answers['xored'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('H',self.noredNodes['GH'],'nored')
            x2 = self.__GetDistances__('F',self.noredNodes['CF'],'nored')
            dr1 = self.noredNodes['AB'].GetDistance(self.noredNodes['BC'])
            dr2 = self.noredNodes['DE'].GetDistance(self.noredNodes['EF'])
            dc1 = self.noredNodes['AD'].GetDistance(self.noredNodes['DG'])
            dc2 = self.noredNodes['BE'].GetDistance(self.noredNodes['EH'])
            diff1 = np.square(np.subtract(x1,dr1))
            diff2 = np.square(np.subtract(x1,dr2))
            diff3 = np.square(np.subtract(x2,dc1))
            diff4 = np.square(np.subtract(x2,dc2))
            ans = np.sqrt(np.add(np.add(np.add(diff1,diff2),diff3),diff4))
            idx = np.argmin(ans)
            answers['nored'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('H',self.ImageNodes['GH'],'image')
            x2 = self.__GetDistances__('F',self.ImageNodes['CF'],'image')
            dr1 = self.ImageNodes['AB'].GetDistance(self.ImageNodes['BC'])
            dr2 = self.ImageNodes['DE'].GetDistance(self.ImageNodes['EF'])
            dc1 = self.ImageNodes['AD'].GetDistance(self.ImageNodes['DG'])
            dc2 = self.ImageNodes['BE'].GetDistance(self.ImageNodes['EH'])
            diff1 = np.square(np.subtract(x1,dr1))
            diff2 = np.square(np.subtract(x1,dr2))
            diff3 = np.square(np.subtract(x2,dc1))
            diff4 = np.square(np.subtract(x2,dc2))
            ans = np.sqrt(np.add(np.add(np.add(diff1,diff2),diff3),diff4))
            idx = np.argmin(ans)
            answers['image'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('H',self.xoredClusterNodes['GH'],'xoredclusters')
            x2 = self.__GetDistances__('F',self.xoredClusterNodes['CF'],'xoredclusters')
            dr1 = self.xoredClusterNodes['AB'].GetDistance(self.xoredClusterNodes['BC'])
            dr2 = self.xoredClusterNodes['DE'].GetDistance(self.xoredClusterNodes['EF'])
            dc1 = self.xoredClusterNodes['AD'].GetDistance(self.xoredClusterNodes['DG'])
            dc2 = self.xoredClusterNodes['BE'].GetDistance(self.xoredClusterNodes['EH'])
            diff1 = np.square(np.subtract(x1,dr1))
            diff2 = np.square(np.subtract(x1,dr2))
            diff3 = np.square(np.subtract(x2,dc1))
            diff4 = np.square(np.subtract(x2,dc2))
            ans = np.sqrt(np.add(np.add(np.add(diff1,diff2),diff3),diff4))
            idx = np.argmin(ans)
            answers['xoredClusters'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('H',self.xoredDeviationNodes['GH'],'xoredstd')
            x2 = self.__GetDistances__('F',self.xoredDeviationNodes['CF'],'xoredstd')
            dr1 = self.xoredDeviationNodes['AB'].GetDistance(self.xoredDeviationNodes['BC'])
            dr2 = self.xoredDeviationNodes['DE'].GetDistance(self.xoredDeviationNodes['EF'])
            dc1 = self.xoredDeviationNodes['AD'].GetDistance(self.xoredDeviationNodes['DG'])
            dc2 = self.xoredDeviationNodes['BE'].GetDistance(self.xoredDeviationNodes['EH'])
            diff1 = np.square(np.subtract(x1,dr1))
            diff2 = np.square(np.subtract(x1,dr2))
            diff3 = np.square(np.subtract(x2,dc1))
            diff4 = np.square(np.subtract(x2,dc2))
            ans = np.sqrt(np.add(np.add(np.add(diff1,diff2),diff3),diff4))
            idx = np.argmin(ans)
            answers['xoredStd'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('H',self.xoredDensitiesNodes['GH'],'xoreddens')
            x2 = self.__GetDistances__('F',self.xoredDensitiesNodes['CF'],'xoreddens')
            dr1 = self.xoredDensitiesNodes['AB'].GetDistance(self.xoredDensitiesNodes['BC'])
            dr2 = self.xoredDensitiesNodes['DE'].GetDistance(self.xoredDensitiesNodes['EF'])
            dc1 = self.xoredDensitiesNodes['AD'].GetDistance(self.xoredDensitiesNodes['DG'])
            dc2 = self.xoredDensitiesNodes['BE'].GetDistance(self.xoredDensitiesNodes['EH'])
            diff1 = np.square(np.subtract(x1,dr1))
            diff2 = np.square(np.subtract(x1,dr2))
            diff3 = np.square(np.subtract(x2,dc1))
            diff4 = np.square(np.subtract(x2,dc2))
            ans = np.sqrt(np.add(np.add(np.add(diff1,diff2),diff3),diff4))
            idx = np.argmin(ans)
            answers['xoredDensities'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('H',self.noredClusterNodes['GH'],'noredclusters')
            x2 = self.__GetDistances__('F',self.noredClusterNodes['CF'],'noredclusters')
            dr1 = self.noredClusterNodes['AB'].GetDistance(self.noredClusterNodes['BC'])
            dr2 = self.noredClusterNodes['DE'].GetDistance(self.noredClusterNodes['EF'])
            dc1 = self.noredClusterNodes['AD'].GetDistance(self.noredClusterNodes['DG'])
            dc2 = self.noredClusterNodes['BE'].GetDistance(self.noredClusterNodes['EH'])
            diff1 = np.square(np.subtract(x1,dr1))
            diff2 = np.square(np.subtract(x1,dr2))
            diff3 = np.square(np.subtract(x2,dc1))
            diff4 = np.square(np.subtract(x2,dc2))
            ans = np.sqrt(np.add(np.add(np.add(diff1,diff2),diff3),diff4))
            idx = np.argmin(ans)
            answers['noredClusters'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('H',self.noredDeviationNodes['GH'],'noredstd')
            x2 = self.__GetDistances__('F',self.noredDeviationNodes['CF'],'noredstd')
            dr1 = self.noredDeviationNodes['AB'].GetDistance(self.noredDeviationNodes['BC'])
            dr2 = self.noredDeviationNodes['DE'].GetDistance(self.noredDeviationNodes['EF'])
            dc1 = self.noredDeviationNodes['AD'].GetDistance(self.noredDeviationNodes['DG'])
            dc2 = self.noredDeviationNodes['BE'].GetDistance(self.noredDeviationNodes['EH'])
            diff1 = np.square(np.subtract(x1,dr1))
            diff2 = np.square(np.subtract(x1,dr2))
            diff3 = np.square(np.subtract(x2,dc1))
            diff4 = np.square(np.subtract(x2,dc2))
            ans = np.sqrt(np.add(np.add(np.add(diff1,diff2),diff3),diff4))
            idx = np.argmin(ans)
            answers['noredStd'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('H',self.noredDensitiesNodes['GH'],'noreddens')
            x2 = self.__GetDistances__('F',self.noredDensitiesNodes['CF'],'noreddens')
            dr1 = self.noredDensitiesNodes['AB'].GetDistance(self.noredDensitiesNodes['BC'])
            dr2 = self.noredDensitiesNodes['DE'].GetDistance(self.noredDensitiesNodes['EF'])
            dc1 = self.noredDensitiesNodes['AD'].GetDistance(self.noredDensitiesNodes['DG'])
            dc2 = self.noredDensitiesNodes['BE'].GetDistance(self.noredDensitiesNodes['EH'])
            diff1 = np.square(np.subtract(x1,dr1))
            diff2 = np.square(np.subtract(x1,dr2))
            diff3 = np.square(np.subtract(x2,dc1))
            diff4 = np.square(np.subtract(x2,dc2))
            ans = np.sqrt(np.add(np.add(np.add(diff1,diff2),diff3),diff4))
            idx = np.argmin(ans)
            answers['noredDensities'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('H',self.imageClusterNodes['GH'],'imageclusters')
            x2 = self.__GetDistances__('F',self.imageClusterNodes['CF'],'imageclusters')
            dr1 = self.imageClusterNodes['AB'].GetDistance(self.imageClusterNodes['BC'])
            dr2 = self.imageClusterNodes['DE'].GetDistance(self.imageClusterNodes['EF'])
            dc1 = self.imageClusterNodes['AD'].GetDistance(self.imageClusterNodes['DG'])
            dc2 = self.imageClusterNodes['BE'].GetDistance(self.imageClusterNodes['EH'])
            diff1 = np.square(np.subtract(x1,dr1))
            diff2 = np.square(np.subtract(x1,dr2))
            diff3 = np.square(np.subtract(x2,dc1))
            diff4 = np.square(np.subtract(x2,dc2))
            ans = np.sqrt(np.add(np.add(np.add(diff1,diff2),diff3),diff4))
            idx = np.argmin(ans)
            answers['imageClusters'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('H',self.imageDeviationNodes['GH'],'imagestd')
            x2 = self.__GetDistances__('F',self.imageDeviationNodes['CF'],'imagestd')
            dr1 = self.imageDeviationNodes['AB'].GetDistance(self.imageDeviationNodes['BC'])
            dr2 = self.imageDeviationNodes['DE'].GetDistance(self.imageDeviationNodes['EF'])
            dc1 = self.imageDeviationNodes['AD'].GetDistance(self.imageDeviationNodes['DG'])
            dc2 = self.imageDeviationNodes['BE'].GetDistance(self.imageDeviationNodes['EH'])
            diff1 = np.square(np.subtract(x1,dr1))
            diff2 = np.square(np.subtract(x1,dr2))
            diff3 = np.square(np.subtract(x2,dc1))
            diff4 = np.square(np.subtract(x2,dc2))
            ans = np.sqrt(np.add(np.add(np.add(diff1,diff2),diff3),diff4))
            idx = np.argmin(ans)
            answers['imageStd'] = [idx,ans[idx]]

            x1 = self.__GetDistances__('H',self.imageDensitiesNodes['GH'],'imagedens')
            x2 = self.__GetDistances__('F',self.imageDensitiesNodes['CF'],'imagedens')
            dr1 = self.imageDensitiesNodes['AB'].GetDistance(self.imageDensitiesNodes['BC'])
            dr2 = self.imageDensitiesNodes['DE'].GetDistance(self.imageDensitiesNodes['EF'])
            dc1 = self.imageDensitiesNodes['AD'].GetDistance(self.imageDensitiesNodes['DG'])
            dc2 = self.imageDensitiesNodes['BE'].GetDistance(self.imageDensitiesNodes['EH'])
            diff1 = np.square(np.subtract(x1,dr1))
            diff2 = np.square(np.subtract(x1,dr2))
            diff3 = np.square(np.subtract(x2,dc1))
            diff4 = np.square(np.subtract(x2,dc2))
            ans = np.sqrt(np.add(np.add(np.add(diff1,diff2),diff3),diff4))
            idx = np.argmin(ans)
            answers['imageDensities'] = [idx,ans[idx]]
        return answers