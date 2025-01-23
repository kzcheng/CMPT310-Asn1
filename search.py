# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


# Old version of depthFirstSearch, gonna rewrite it using an actual fringe
def depthFirstSearchV1(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    # Q1: Finding a Fixed Food Dot using Depth First Search

    # Variables
    currentState = problem.getStartState()
    currentPath = []
    visitedStates = [currentState]
    # In each tuple, the first element is the state and the second element is how we got there
    history = util.Stack()
    longestFailure = currentPath.copy()
    # A place to store all the expansion results of states, this is a dictionary
    successorsDict = {}

    # Note: Using problem.getSuccessors is considered as expanding a state. The autograder will check for this, so don't use it repeatedly.
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    history.push((currentState, currentPath.copy()))

    while not problem.isGoalState(currentState):
        print("\n\n\n")
        print("Now we are at: ", currentState)
        print("We reached here by: ", currentPath)

        # Get all the successors
        successors = util.Stack()
        # If we have already expanded this state, we don't need to do it again
        if currentState in successorsDict:
            print("Already expanded this state, using database")
            successors = successorsDict[currentState]
        else:
            print("Expanding the state")
            for successor in problem.getSuccessors(currentState):
                print("Potential Successor: ", successor)
                successors.push(successor)
            successorsDict[currentState] = successors

        # Check the list of successors to see if we have any unvisited states
        print("Visited States: ", visitedStates)
        nextNode = None
        while not successors.isEmpty():
            successor = successors.pop()
            successorState = successor[0]

            # If the successor is not visited, then this will be the next node we analyze
            if successorState not in visitedStates:
                nextNode = successor
                break

        # If we have no unvisited states, then we need to backtrack
        if nextNode is None:
            print("No unvisited states found on ", currentState, ", backtracking")
            if history.isEmpty():
                print("No solution found, returning longest path attempted")
                return longestFailure
            currentState, currentPath = history.pop()
            print("Backtracking to: ", currentState)
            print("Previous Path: ", currentPath)
            continue

        # Now, we attempt to visit the next node

        print("Going towards unvisited state: ", nextNode)

        nextState = nextNode[0]
        nextAction = nextNode[1]

        visitedStates.append(nextState)

        # Note: Spend a very long time debugging this, leaving a note
        # It is very very very important to record the current situation as history before we update the current state
        history.push((currentState, currentPath.copy()))

        currentState = nextState
        currentPath.append(nextAction)

        if len(currentPath) > len(longestFailure):
            longestFailure = currentPath.copy()

    return currentPath


# Rewrite version, using a fringe to store the states to be expanded
def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    # Q1: Finding a Fixed Food Dot using Depth First Search

    # Variables
    currentState = problem.getStartState()
    # The path taken to reach the current state
    currentPath = []
    visitedStates = [currentState]
    longestFailure = currentPath.copy()
    # Fringe contains tuples, the first element is the state, the second element is the path taken to reach this state, and the third element is the successor
    fringe = util.Stack()
    stepCounter = 0

    while not problem.isGoalState(currentState):
        stepCounter += 1
        print("\n")
        print("[ Step ", stepCounter, "]")
        print("Current State: ", currentState)
        print("We reached here by: ", currentPath)

        # Get all the successors
        successors = problem.getSuccessors(currentState)
        print("Successors: ", successors)
        for successor in successors:
            if successor[0] not in visitedStates:
                fringe.push((currentState, currentPath.copy(), successor))

        # If the fringe is empty, we have no solution
        if fringe.isEmpty():
            print("No solution found, returning longest path attempted")
            return longestFailure

        # Get the next node to expand
        print("")
        nextSituation = fringe.pop()
        currentState = nextSituation[0]
        currentPath = nextSituation[1].copy()
        nextNode = nextSituation[2]

        print("Jumping to State: ", currentState)
        print("Path to this state: ", currentPath)
        print("Going towards unvisited state: ", nextNode)

        nextState = nextNode[0]
        nextAction = nextNode[1]

        currentState = nextState
        currentPath.append(nextAction)
        visitedStates.append(currentState)

        if len(currentPath) > len(longestFailure):
            longestFailure = currentPath.copy()

    return currentPath


def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""

    # Q2: Breadth First Search

    # Variables
    currentState = problem.getStartState()
    # The path taken to reach the current state
    currentPath = []
    visitedStates = []
    longestFailure = currentPath.copy()
    # Fringe contains tuples, the first element is the state, the second element is the path taken to reach this state, and the third element is the successor
    fringe = util.Queue()
    stepCounter = 0

    while not problem.isGoalState(currentState):
        stepCounter += 1
        print("\n")
        print("[ Step ", stepCounter, "]")
        print("Current State: ", currentState)
        print("We reached here by: ", currentPath)

        # If this is the first time visiting this state, we expand it
        if currentState not in visitedStates:
            visitedStates.append(currentState)
            successors = problem.getSuccessors(currentState)
            for successor in successors:
                fringe.push((currentState, currentPath.copy(), successor))

        # If the fringe is empty, we have no solution
        if fringe.isEmpty():
            print("No solution found, returning longest path attempted")
            return longestFailure

        # Get the next node to expand
        print("")
        nextSituation = fringe.pop()
        currentState = nextSituation[0]
        currentPath = nextSituation[1].copy()
        nextNode = nextSituation[2]

        print("Jumping to State: ", currentState)
        print("Path to this state: ", currentPath)
        print("Going towards unvisited state: ", nextNode)

        nextState = nextNode[0]
        nextAction = nextNode[1]

        currentState = nextState
        currentPath.append(nextAction)

        if len(currentPath) > len(longestFailure):
            longestFailure = currentPath.copy()

    return currentPath


def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
n = Directions.NORTH
e = Directions.EAST
s = Directions.SOUTH
w = Directions.WEST
