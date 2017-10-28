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
import numpy
from ImageSolver import ImageSolver

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

        solver = ImageSolver(problem.problemType)
        problems = dict()
        solutions = dict()

        for p in problem.figures:
            if p >= 'A' and p <= 'H':
                problems[p] = problem.figures[p].visualFilename
            elif p >= '1' and p <= '6':
                solutions[p] = problem.figures[p].visualFilename

        solver.AddImages(problems,solutions)
        solver.GetDifferences()
        solver.CreateGraph()
        solver.GetDistances()

        solver.OutputImageData(problem.name)
        solver.OutputGraphNodes(problem.name)

        rtn = solver.GetAnswer()

        return rtn
