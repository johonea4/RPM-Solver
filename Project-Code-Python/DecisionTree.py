import numpy as np
from collections import Counter
import time
from enum import Enum

"""
This File implements classes necessary to build a decision tree for 
different translation types. Much of this file is re-use of code
from a previous class -> CS6601 Artificial Intelligence
"""
#List of the different type of available/known transformations
class FigureTranslations(Enum):
    Same=0x1
    Deleted=0x2
    NewShape=0x4
    TransformedShape=0x8
    Mirrored=0x10
    Rotated=0x20
    Filled=0x40
    HalfFilled=0x80
    Unfilled=0x100
    HalfUnfilled=0x200
    MovedLeft=0x400
    MovedRight=0x800
    MovedUp=0x1000
    MovedDown=0x2000

class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.

        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.

        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.

        Args:
            feature (list(int)): vector for feature.

        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)

################################################################################
################################################################################
#             DECISION TREE IMPLEMENTATION
################################################################################
################################################################################

class DecisionTree:
    """Class for automatic tree-building and classification."""

#################################################################################
#             PUBLIC CLASS FUNCTIONS
#################################################################################
    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.

        Starts with an empty root.

        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def TrainModel(self, features, classes):
        """Build the tree from root using __build_tree__().

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        self.root = self.__build_tree__(features, classes) 

    def classify(self, features):
        """Use the fitted tree to classify an example features.

        Args:
            features list(int)

        Return:
            class label.
        """

        class_label = 0

        class_label = (self.root.decide(features))

        return class_label


#######################################################################
#             INTERNAL CLASS FUNCTIONS
#######################################################################

    def testAllSame(self,classes):
        test = classes[0]
        for c in classes:
            if c != test:
                return None
        return DecisionNode(None,None,None,test)

    def testDepth(self,depth,classes):
        if depth > self.depth_limit:
            nt = classes.count(1)
            nf = classes.count(0)
            if nt>nf:
                return DecisionNode(None,None,None,1)
            elif nf>nt:
                return DecisionNode(None,None,None,0)
            else:
                return DecisionNode(None,None,None,classes[0])
        return None

    def getGaussian(self,arr):
        """
        takes an array of data and returns a dictionary
        containing the gaussian distribution information
        """
        rtn = dict()
        rtn['mean'] = np.mean(arr)
        rtn['std'] = np.std(arr)
        rtn['var'] = rtn['std'] * rtn['std']

    def testPoint(self,x,gaus):
        """
        Tests a single point against the gaussian
        distribution and returns false if the distance
        from the mean is greater than the variance
        """
        if(abs(x-gaus['mean'])>gaus['var'])
            return False
        return True

    def gini_impurity(self, class_vector):
        """Compute the gini impurity for a list of classes.
        This is a measure of how often a randomly chosen element
        drawn from the class_vector would be incorrectly labeled
        if it was randomly labeled according to the distribution
        of the labels in the class_vector.
        It reaches its minimum at zero when all elements of class_vector
        belong to the same class.

        Args:
            class_vector (list(int)): Vector of classes given as 0 or 1.

        Returns:
            Floating point number representing the gini impurity.
        """
        nvals = len(class_vector)
        n0 = 0
        n1 = 0

        if nvals <= 0: 
            return 0
        for c in class_vector:
            if(c==0):
                n0+=1
            elif(c==1):
                n1+=1

        impurity = 1 - (pow((n0/nvals),2) + pow((n1/nvals),2))

        return impurity

    def gini_gain(self,previous_classes, current_classes):
        """Compute the gini impurity gain between the previous and current classes.
        Args:
            previous_classes (list(int)): Vector of classes given as 0 or 1.
            current_classes (list(list(int): A list of lists where each list has
                0 and 1 values).
        Returns:
            Floating point number representing the information gain.
        """
        nvals = len(previous_classes)
        if nvals <= 0:
            return 0
        impurity = self.gini_impurity(previous_classes)
        summation = 0

        for c in current_classes:
            summation += (len(c)/nvals) * gini_impurity(c)

        gain = impurity - summation

        return gain


    def getGains(self,features,classes,numAttr):
        """
        Uses the above functions to determine gains for
        a list of features. This will return the information for
        the best attribute to split on.
        """
        gains = list()
        splitlist = list()

        for i in range(numAttr):
            attr = features[:,i]
            if attr[0]==None or np.isnan(attr[0]):
                splitlist.append([list(),list()])
                gains.append(-1)
                continue
            gaus = self.getGaussian(attr)
            pList = list()
            nList = list()
            for j in range(len(attr)):
                if(self.testPoint(attr[j])):
                    pList.append(classes[j])
                else:
                    nList.append(classes[j])
            splitlist.append([pList,nList])
            gains.append(gini_gain(classes,splitlist[i]))
        alpha_max = max(gains)
        alpha_index = gains.index(alpha_max)

        results = dict()
        results['gains'] = gains
        results['split_classes'] = splitlist[alpha_index]
        results['alpha_max'] = alpha_max
        results['alpha_index'] = alpha_index
        results['gaus'] = self.getGaussian(features[:,alpha_index])
        return results

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
            depth (int): max depth of tree.  Default is 0.

        Returns:
            Root node of decision tree.
        """
        #First Check if all classes are the same
        test = self.testAllSame(classes)
        if test != None:
            return test

        #Next Check if depth > depthLimit and return the most frequent class
        numfeatures = np.size(features,0)
        numattributes = np.size(features,1)

        test = self.testDepth(depth,classes)
        if test != None:
            return test
        
        #Get all the GiniGains for the features
        results = self.getGains(features,classes,numattributes)
        alpha_index = results['alpha_index']
        posFeatures = list()
        negFeatures = list()
        for i,f in enumerate(features):
            if self.testPoint(f[alpha_index],results['gaus']):
                posFeatures.append(list(f))
            else:
                negFeatures.append(list(f))
        if len(results['split_classes'][0])<=0:
            return self.testDepth(self.depth_limit+1,numattributes,results['split_classes'][1])
        elif len(results['split_classes'][1])<=0:
            return self.testDepth(self.depth_limit+1,numattributes,results['split_classes'][0])

        node = DecisionNode(None,None,lambda feat: (results['gaus']['mean']-feat[alpha_index])<=results['gaus']['var'])
        node.left = self.__build_tree__(np.array(posFeatures),results['split_classes'][0],depth+1)
        node.right = self.__build_tree__(np.array(negFeatures),results['split_classes'][1],depth+1)

        return node
#######################################################################
#######################################################################
#             RANDOM FOREST IMPLEMENTATION
#######################################################################
#######################################################################
class RandomForest:
    """Random forest classification."""

#######################################################################
#            PUBLIC CLASS FUNCTIONS
######################################################################
    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.

         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.

            images (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """
        numFeatures = np.size(features,0)
        numAttr = np.size(features,1)
        
        for i in range(self.num_trees):
            forest = self.getForest(features,classes)
            for f in forest[0]:
                for j in range(numAttr):
                    if j not in forest[2]:
                        f[j] = None
            tree = DecisionTree(self.depth_limit)
            tree.fit(forest[0],forest[1])
            self.trees.append(tree)

    def getProbability(self, features):
        """Classify a feature

        Args:
            features list(int).
        """

        classList = list()
        test = list()
        numFeatures = np.size(features,0)

        for tree in self.trees:
            c = tree.classify(features)
            test.append(c)
        test = np.array(test)
        nt = np.count_nonzero(test)

        return nt/np.size(test)
        
##############################################################
#              PRIVATE CLASS FUNCTIONS
# ############################################################ 
    def getForest(self,features,classes):
        numFeatures = np.size(features,0)
        numAttr = np.size(features,1)
        numSubFeatures = int(self.example_subsample_rate * numFeatures)
        numSubAttr = int(self.attr_subsample_rate * numAttr)

        randomFeatures = set()
        randomAttr = set()
        while len(randomFeatures) < numSubFeatures: randomFeatures.add(np.random.randint(0,numFeatures))
        while len (randomAttr) < numSubAttr: randomAttr.add(np.random.randint(0,numAttr))
        subFeatures = [ features[r] for r in randomFeatures ]
        subClasses = [ classes[r] for r in randomFeatures ]

        return [np.array(subFeatures),np.array(subClasses),randomAttr]

################################################################################
################################################################################
#              TRANSLATION DECISION FORESTS
################################################################################
################################################################################

class TranslationTree:
    def __init__(self, training, t_type):
        """
        If in training mode, will allow for training of the
        forests for the given translation type. If not, the tree 
        data should be loaded from a CSV file stored locally
        """
        self.type = FigureTranslations()
        self.type=t_type
        self.training=training

        if(training==False):
            #Load Data
            return
