import os
import csv
import numpy as np
import time
import math
from PIL import Image

class KM_Clusters(object):
    def __init__(self, imageArray, dim, k):
        #Transform array into a 2d matrix representing the image
        arr = np.reshape(imageArray,dim)
        #Now find all locations that equal 1
        points = np.where(arr==1)
        sz = np.size(points,1)

        #form the data array
        self.data = []
        for i in range(0,sz):
            self.data.append(np.array([points[0][i],points[1][i]]))
        if len(self.data)==0:
            return

        n = 184/k
        self.means = []
        for i in range(0,k):
            self.means.append([i*n,i*n])

        self.data = np.array(self.data)
        #initiate the variables
        self.components = k

    def getDistances(self):
        kArray = []
        for i in range(self.components):
            tmp = np.subtract(self.data,self.means[i])
            tmp = np.square(tmp)
            tmp = np.sum(tmp,1)
            tmp = np.sqrt(tmp)
            kArray.append(tmp)
        return np.array(kArray)

    def getMeans(self,kArray):
        minValues = np.min(kArray,0)
        kValues = []
        kIndexes = []
        kAvg = []
        for i in range(self.components):
            tmp = np.where(kArray[i] <= minValues)
            if np.size(tmp) <=0:
                kIndexes.append([])
                kValues.append([])
                kAvg.append([0,0])
                continue
            kIndexes.append(tmp)
            kValues.append(self.data[tmp])
            kAvg.append(np.mean(self.data[tmp],0))
        self.segments = np.array(kValues)
        self.segmentIdx = np.array(kIndexes)
        self.means = np.array(kAvg)

    def getConvergence(self,kMeans, initial_means):
        convergence=False
        convArray = np.subtract(initial_means,kMeans)
        convArray = np.absolute(convArray)
        for row in convArray:
            for item in row:
                if item < 1e-6:
                    convergence=True
                else:
                    convergence=False
                    break
            if convergence==False: break
        return convergence

    def FindClusters(self):

        initial_means = np.copy(self.means)
        kArray = self.getDistances()
        self.getMeans(kArray)
        while(self.getConvergence(self.means,initial_means)==False):
            initial_means = np.copy(self.means)
            kArray = self.getDistances()
            self.getMeans(kArray)

        #Cleanup
        finalMeans = []
        numClusters = 0
        finalSegments = []
        finalIndexes = []
        for i in range(self.components):
            if self.means[i][0]==0 and self.means[i][1]==0:
                continue
            else:
                finalMeans.append(self.means[i])
                finalSegments.append(self.segments[i])
                finalIndexes.append(self.segmentIdx[i])
                numClusters+=1
        self.means = finalMeans
        self.segments = finalSegments
        self.segmentIdx = finalIndexes
        self.components = numClusters


class EM_Clusters(object):
    def __init__(self, imageArray, dim, k):
        #Transform array into a 2d matrix representing the image
        arr = np.reshape(imageArray,dim)
        #Now find all locations that equal 1
        points = np.where(arr==1)
        sz = np.size(points,1)

        #form the data array
        self.data = []
        for i in range(0,sz):
            self.data.append(np.array([points[0][i],points[1][i]]))
        if len(self.data)==0:
            return

        self.data = np.array(self.data)
        #initiate the variables
        self.components = k
        idx = np.random.randint(0,len(self.data),k)
        self.means = self.data[idx]
        #v = np.max(dim)

        self.variances = [[190,190]] * k
        v = 1.0/k
        self.coefficients = [[v,v]] * k

    def convergence_check(self,prev_likelihood, new_likelihood, conv_ctr, conv_ctr_cap=10):
        increase_convergence_ctr = (abs(prev_likelihood) * 0.9 <
                                abs(new_likelihood) <
                                abs(prev_likelihood) * 1.1)

        if increase_convergence_ctr:
            conv_ctr += 1
        else:
            conv_ctr = 0

        return conv_ctr, conv_ctr > conv_ctr_cap


    def x_prob(self):
        """
        Since the data is 2 dimensional we will use a 
        bivariate normal distribution for the gaussians
        """
        arr = self.data
        px = []

        for i in range(self.components):
            #a1 = 1.0/np.sqrt(np.multiply(2.0*np.pi,self.variances[i]))
            #a2 = np.subtract(arr,self.means[i])
            #a2 = np.square(a2)
            #a2 = np.divide(a2,np.multiply(2.0,self.variances[i]))
            #a2 = np.multiply(a2,-1.0)
            #a2 = np.exp(a2)
            #a3 = np.multiply(a1,a2)
            a1 = -0.5*np.log(np.multiply(2.0*np.pi,self.variances[i]))
            a2 = np.subtract(arr,self.means[i])
            a2 = np.square(a2)
            a2 = np.divide(a2,np.multiply(2.0,self.variances[i]))
            a3 = np.subtract(a1,a2)
            a3 = np.exp(a3)
            px.append(a3)
        
        return px

    def x_resp(self):
        px = self.x_prob()
        resp = []

        for i in range(self.components):
            a1 = px[i]
            a1 = np.multiply(a1,self.coefficients[i])
            a2 = []
            for j in range(self.components):
                b1 = px[j]
                b1 = np.multiply(b1,self.coefficients[j])
                if(j==0):
                    a2=b1
                else:
                    a2 = np.add(a2,b1)
            a3 = np.divide(a1,a2)
            resp.append(a3)
        return resp

    def likelihood(self):
        """Assign a log
        likelihood to the trained
        model based on the following
        formula for posterior probability:
        ln(Pr(X | mixing, mean, stdev)) = sum((n=1 to N), ln(sum((k=1 to K),
                                          mixing_k * N(x_n | mean_k,stdev_k))))

        returns:
        log_likelihood = float [0,1]
        """
        px = self.x_prob()
        px = np.transpose(px,(1,0,2))
        px = np.multiply(px,self.coefficients)
        px = np.sum(px,1)
        px = np.log(px)
        
        l = np.sum(px)

        #l = np.square(l)
        #l = np.sum(l)
        #l = np.sqrt(l)
        return l


    def FindClusters(self):

        convergence = False
        pl=self.likelihood()
        arrf = self.data
        nl=pl
        c =0

        while convergence == False:
            #1. Take the initial means and stdev and find the probabilty that they fit
            px = self.x_resp()

            coef = np.mean(px,1)
            self.coefficients=coef
            if np.any(self.coefficients==0):
                return None
            if np.isnan(np.sum(self.coefficients)) or np.isinf(np.sum(self.coefficients)):
                return None

            m = np.multiply(arrf,px)
            m = np.sum(m,1)
            m = np.divide(m,np.sum(px,1))
            self.means = m
            
            if np.any(self.means==0):
                return None
            if np.isnan(np.sum(self.means)) or np.isinf(np.sum(self.means)):
                return None

            v = []

            for i in range(self.components):
                v.append(np.subtract(arrf,self.means[i]))
            v = np.square(v)
            v = np.multiply(v,px)
            v = np.sum(v,1)
            v = np.divide(v,np.sum(px,1))
            self.variances = v

            if np.any(self.variances==0):
                return None
            if np.isnan(np.sum(self.variances)) or np.isinf(np.sum(self.variances)):
                return None

            nl = self.likelihood()
            c, convergence = self.convergence_check(pl,nl,c)
            pl=nl
        return 1


def BIC(gmm):
    k = 3 * gmm.components
    n = np.size(gmm.data,0)
    L = gmm.likelihood()

    return np.log(n)*k - 2*L

def GetEMClusters(imageData, dimensions):

    minClusters = 1
    maxClusters = 20

    bestFit=0
    bestModel = None

    for k in range(minClusters,maxClusters+1):
        gmm = EM_Clusters(imageData,dimensions,k)
        if len(gmm.data) == 0:
            break
        if gmm.FindClusters() == None:
            print "Error: Data Evaluated to an invalid mixture model\n"
            continue

        b = BIC(gmm)

        if(k==minClusters):
            bestFit=b
            bestModel=gmm
        else:
            if b<bestFit: 
                bestFit=b
                bestModel=gmm

    return bestModel

def GetKMClusters(imageData, dimensions):
    kmm = KM_Clusters(imageData,dimensions,20)
    if len(kmm.data) == 0:
        return None
    kmm.FindClusters()
    return kmm

def GetClusters(imageData, dimensions, type='KMM'):
    if type == 'KMM':
        return GetKMClusters(imageData,dimensions)
    else:
        return GetEMClusters(imageData,dimensions)
