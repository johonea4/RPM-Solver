# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
from PIL import Image
import numpy as np
from VisualProcessor import VisualProcessor
from Comparator import Comparator

class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        pass

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints 
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self,problem):

        print("Starting Solver for problem %s\n" % problem.name)

        processor = VisualProcessor(problem.figures,problem.problemType)
        compare = Comparator(processor)
        compare.CreateGraphNodes()
        proposedAnswers = compare.GetSolutions()

        if proposedAnswers == None or len(proposedAnswers)<=0:
            return -1;

        answers = []
        for answer in proposedAnswers:
            ans = proposedAnswers[answer][0]
            val = proposedAnswers[answer][1]
            print "\tAnswer for " + answer + " method: [" + str(ans) + ", " + str(val) + "]"
            answers.append(proposedAnswers[answer][0])

        answerCounts = []
        for i in range(0,processor.numAnswers):
            answerCounts.append(answers.count(i))
        #solver.OutputImageCombinations(problem.name)
        
        if np.all(np.array(answerCounts)<=1):
            return -1

        maxCounts = np.max(answerCounts)
        indexes = np.argwhere(answerCounts==maxCounts)
        indexes = list(indexes.flatten())

        if len(indexes)>1:
            return -1
            #bestAns = -1
            #bestDist = float('inf')
            #for answer in proposedAnswers:
            #    for i in indexes:
            #        if proposedAnswers[answer][0]==i and proposedAnswers[answer][1] < bestDist:
            #            bestDist = proposedAnswers[answer][1]
            #            bestAns = proposedAnswers[answer][0]
            #agreedAnswer = bestAns
        else:
            agreedAnswer = indexes[0]

        return agreedAnswer+1