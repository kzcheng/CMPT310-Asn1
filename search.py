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

# This assignment is completed by Kevin Cheng for the course CMPT 310 (Intro Artificial Intelligence) at Simon Fraser University
# The repo for this assignment can be found at https://github.com/kzcheng/CMPT310-Asn1


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import logging
import util
from game import Directions
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


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


def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    logging.debug("Start:", problem.getStartState())
    logging.debug("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    logging.debug("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    # Q1: Finding a Fixed Food Dot using Depth First Search

    # Variables
    currentState = problem.getStartState()
    # The path taken to reach the current state
    currentPath = []
    visitedStates = []
    longestFailure = currentPath.copy()
    # Fringe contains tuples, the first element is the state, the second element is the path taken to reach this state, and the third element is the successor
    fringe = util.Stack()
    stepCounter = 0

    while not problem.isGoalState(currentState):
        stepCounter += 1
        logging.debug("\n")
        logging.debug("[ Step %r ]", stepCounter)
        logging.debug("Current State: %r", currentState)
        logging.debug("We reached here by: %r", currentPath)

        # If this is the first time visiting this state, we expand it
        if currentState not in visitedStates:
            visitedStates.append(currentState)
            successors = problem.getSuccessors(currentState)
            for successor in successors:
                fringe.push((currentState, currentPath.copy(), successor))

        # If the fringe is empty, we have no solution
        if fringe.isEmpty():
            logging.info("No solution found, returning longest path attempted")
            return longestFailure

        # Get the next node to expand
        nextSituation = fringe.pop()
        currentState = nextSituation[0]
        currentPath = nextSituation[1].copy()
        nextNode = nextSituation[2]

        logging.debug("Jumping to State: %r", currentState)
        logging.debug("Path to this state: %r", currentPath)
        logging.debug("Going towards unvisited state: %r", nextNode)

        nextState = nextNode[0]
        nextAction = nextNode[1]

        currentState = nextState
        currentPath.append(nextAction)

        if len(currentPath) > len(longestFailure):
            longestFailure = currentPath.copy()

    return currentPath


def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""

    # Q2: Breadth First Search

    # The fringe online stores 1 thing, which is the state we are going towards
    fringe = util.Queue()

    # Each element records the path to reach each state
    # In BFS, only the first way of reaching each state is recorded
    paths = {}

    state = problem.getStartState()
    path = []   # The path is the path to reach the current state from start

    # Stuff used for debugging
    stepCounter = 0

    # Before we begin looping through the fringe to analyze every state, we need to add the initial condition into the fringe
    fringe.push(state)
    paths[state] = path.copy()

    while stepCounter < 1000000:  # Arbitrary cap to number of loops
        if fringe.isEmpty():
            logging.info("\nThe fringe is empty")
            logging.info("No solution found")
            return []

        # If there are still things we can analyze in the fringe, we should do it
        stepCounter += 1
        logging.debug("\n\n[ Step %r ]", stepCounter)
        # logging.debug("Current entire fringe: %r", fringe.list)
        logging.debug("\nThe paths to reach every state: %r", paths)

        # Popping the fringe to decide what is the next state we need to visit, which we will then go to it
        state = fringe.pop()
        path = paths[state].copy()
        logging.debug("Current state we are analyzing: %r", state)

        # Check if we are analyzing the goal, if so, we are done
        if problem.isGoalState(state):
            logging.debug("")
            logging.info("Goal State Reached")

            logging.debug("Path Found")
            logging.debug("Final Path: %r", path)
            return path

        # If this is not the goal, we must expand it
        for successor in problem.getSuccessors(state):
            nextState, nextAction, _ = successor
            nextPath = path + [nextAction]
            if nextState not in paths:
                fringe.push(nextState)
                logging.debug("Added new state to fringe: %r", nextState)
                paths[nextState] = nextPath.copy()
                logging.debug("With planned path to reach state: %r", nextPath)

        # logging.debug("Current entire fringe: %r", fringe.list)

    logging.error("Loop limit reached, aborting search")
    return []


def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""

    # Note: Change of plan
    # Previously, the fringe stores both the state we are going towards, and the relevant current states, which we will need to load in.
    # However, for how util.PriorityQueue is implemented, it seems like a better idea to only store the end state after taking each action.
    # In addition, to make things easier to read, it's probably better to store things in the dictionary type of objects.

    # region Variables
    currentState = problem.getStartState()
    currentPath = []        # The path taken to reach the current state
    visitedStates = []

    # The fringe only stores two things, with the item being the state we are going towards, and the priority being the cost of reaching that state.
    fringe = util.PriorityQueue()
    # Which means, we need another dictionary to store the plan to each state. This dictionary should only store the lowest cost (most optimal) plan we know. But this also needs to store relevant information like which state we were in for each action taken.
    planToState = {}

    # Used for debugging, not mandatory
    longestFailure = currentPath.copy()
    stepCounter = 0
    # endregion

    while not problem.isGoalState(currentState):
        stepCounter += 1
        logging.debug("\n")
        logging.debug("[ Step %r ]", stepCounter)
        logging.debug("Current state we are analyzing: %r", currentState)
        logging.debug("We reached here by: %r", currentPath)

        # If this is the first time visiting this state, we expand it
        if currentState not in visitedStates:
            visitedStates.append(currentState)
            successors = problem.getSuccessors(currentState)
            # For each successor of the current state
            for successor in successors:
                successorState, successorAction, _ = successor

                # Debug prints to check the structure of successor and currentPath
                # logging.debug(f"Successor: {successor}")
                # logging.debug(f"Successor[1]: {successor[1]}")
                # logging.debug(f"CurrentPath: {currentPath}")
                # logging.debug(f"CurrentPath + [successor[1]]: {currentPath + [successor[1]]}")

                nextPath = currentPath + [successorAction]
                costToSuccessor = problem.getCostOfActions(nextPath)

                # Check if this new plan is not better than the previous plan, or if there is already a plan
                oldPlan = planToState.get(successorState, {}).get("costOfPath", float("inf"))
                logging.debug("Old Plan: %r", oldPlan)
                if costToSuccessor >= oldPlan:
                    continue

                # Add the successor state to the fringe, with the cost of reaching that state as the priority
                fringe.update(successorState, costToSuccessor)
                # Add the plan to reach this state to the dictionary
                planToState[successorState] = {
                    "previousState": currentState,
                    "previousPath": currentPath,
                    "currentPath": nextPath,
                    "costOfPath": costToSuccessor,
                }
                logging.debug("Added to fringe: %r", successorState)
                logging.debug("With cost: %r", costToSuccessor)
                logging.debug("Plan to reach this state: %r", planToState[successorState])

        # If the fringe is empty, we have no solution
        if fringe.isEmpty():
            logging.info("No solution found, returning longest path attempted")
            return longestFailure

        # Thought: Do we need to update the plan to reach each state? Since if we found a cheaper way to reach a state, the plan may not be optimal anymore.

        # Popping the fringe to decide what is the next situation to deal with
        nextState = fringe.pop()

        # Move to it
        currentState = nextState
        currentPath = planToState[currentState]["currentPath"]

        if len(currentPath) > len(longestFailure):
            longestFailure = currentPath.copy()

    return currentPath


def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


class SearchContext:
    def __init__(self, problem, heuristic, fringe, paths):
        self.problem = problem
        self.heuristic = heuristic
        self.fringe = fringe
        self.paths = paths


def getCostAStar(context, state, path):
    return context.problem.getCostOfActions(path) + context.heuristic(state, context.problem)


def updateFringe(context, state, successor):
    path = context.paths[state]
    nextState, nextAction, _ = successor    # The 3rd element is the cost of the next single action
    nextPath = path + [nextAction]
    nextCost = getCostAStar(context, nextState, nextPath)

    # Decide if we should add the next state to the fringe
    # Check if we have already found a path to it
    # If so, check if our new path is better than the old path
    shouldUpdate = nextState not in context.paths or nextCost < context.problem.getCostOfActions(context.paths[nextState]) + context.heuristic(nextState, context.problem)

    if shouldUpdate:
        context.fringe.update(nextState, nextCost)
        logging.debug("Added new state to fringe: %r", nextState)
        logging.debug("With cost: %r", nextCost)

        context.paths[nextState] = nextPath
        logging.debug("With planned path to reach state: %r", nextPath)


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""

    # Each element in the fringe stores 2 things
    # 1. The next state we are going towards
    # 2. The priority of dealing with it
    fringe = util.PriorityQueue()

    # Each element records the path to reach each state
    # It should only store the most optimal way of reaching each state
    paths = {}

    # The context stores some important parts about the search to pass into methods
    context = SearchContext(problem, heuristic, fringe, paths)

    # If a variable don't have anything before it, it means that this is the current one we are working on
    state = problem.getStartState()
    path = []  # The path is the path to reach the current state from start

    # Stuff used for debugging
    stepCounter = 0

    # Before we begin looping through the fringe to analyze every state, we need to add the initial condition into the fringe
    cost = getCostAStar(context, state, path)
    context.fringe.update(state, cost)
    context.paths[state] = path

    while stepCounter < 1000000:  # Arbitrary cap to number of loops
        if context.fringe.isEmpty():
            logging.info("\nThe fringe is empty")
            logging.info("No solution found")
            return []

        # If there are still things we can analyze in the fringe, we should do it
        stepCounter += 1
        logging.debug("\n\n[ Step %r ]", stepCounter)
        logging.debug("Current entire fringe: %r", context.fringe.heap)
        logging.debug("The paths to reach every state: %r", context.paths)

        # Popping the fringe to decide what is the next state we need to visit, which we will then go to it
        state = context.fringe.pop()
        logging.debug("Current state we are analyzing: %r", state)

        # Check if we are analyzing the goal, if so, we are done
        if context.problem.isGoalState(state):
            logging.debug("")
            logging.debug("Goal State Reached")

            path = context.paths[state]
            logging.debug("Path Found")
            logging.debug("Final Path: %r", path)
            return path

        # If this is not the goal, we must expand it
        for successor in context.problem.getSuccessors(state):
            updateFringe(context, state, successor)

    logging.error("Loop limit reached, aborting search")
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
n = Directions.NORTH
e = Directions.EAST
s = Directions.SOUTH
w = Directions.WEST
