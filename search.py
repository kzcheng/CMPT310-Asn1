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

import logging
import util
from game import Directions
from typing import List

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
# logging.disable(logging.DEBUG)


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


# Rewrite version, using a fringe to store the states to be expanded
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
        # Turns out, it isn't needed, since Uniform Cost Search means that when we actually expand the state, we already found the optimal plan to reach that state.

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


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""

    return []

    # Q4: A* search

    # region Variables
    currentState = problem.getStartState()
    currentPath = []        # The path taken to reach the current state
    visitedStates = []

    # The fringe only stores two things, with the item being the state we are going towards, and the priority being the cost of reaching that state.
    fringe = util.PriorityQueue()
    # Which means, we need another dictionary to store the plan to each state. This dictionary should only store the lowest cost (most optimal) plan we know. But this also needs to store relevant information like which state we were in for each action taken.
    planToState = {}

    stepCounter = 0

    # endregion

    # Add the initial problem to the fringe and plan
    costAStar = heuristic(currentState, problem)
    fringe.update(currentState, costAStar)
    planToState[currentState] = {
        "previousState": None,
        "previousPath": None,
        "path": currentPath,
        "costAStar": costAStar,
    }

    while stepCounter < 2147483648:  # Basically while true
        # If the fringe is empty, we have no solution
        if fringe.isEmpty():
            logging.info("\nThe fringe is empty")
            logging.info("No solution found")
            return []

        stepCounter += 1
        logging.debug("\n\n[ Step %r ]", stepCounter)
        logging.debug("The entire fringe: %r", fringe.heap)
        logging.debug("The plans to reach every state: %r", planToState)

        # Popping the fringe to decide what is the next state we need to visit, which we will then go to it
        currentState = fringe.pop()
        currentPath = planToState[currentState]["path"]

        logging.debug("Current state we are analyzing: %r", currentState)
        logging.debug("We reached here by: %r", currentPath)

        # Check if we reached the goal
        if problem.isGoalState(currentState):
            logging.info("Goal State Reached")
            break

        # If some conditions are met, we will skip expanding this fringe
        if currentState in visitedStates:
            # If we already visited this state, we need to check if we have a better plan to reach this state
            oldCostAStar = planToState[currentState]["costAStar"]
            newCostAStar = problem.getCostOfActions(currentPath) + heuristic(currentState, problem)
            logging.debug("Old path to reach this state: %r", planToState[currentState]["path"])
            logging.debug("New path to reach this state: %r", currentPath)
            logging.debug("Total A Star Cost of old plan: %r", oldCostAStar)
            logging.debug("Total A Star Cost of new plan: %r", newCostAStar)
            if newCostAStar >= oldCostAStar:
                logging.debug("Already visited this state, and the previous plan is better, no need to expand this state, skipping")
                continue
        else:
            # If this is the first time visiting this state, we should add it to the visited states
            visitedStates.append(currentState)
        
        # Add the plan to reach this state to the dictionary
        planToState[currentState] = {
            "path": currentPath,
            "costAStar": problem.getCostOfActions(currentPath) + heuristic(currentState, problem),
        }

        # Now we should expand this state
        successors = problem.getSuccessors(currentState)
        logging.debug("")
        logging.debug("Successors of the current state: %r", successors)
        # For each successor of the current state
        for successor in successors:
            nextState, nextAction, _ = successor
            logging.debug("")
            logging.debug("Analyzing successor: %r", successor)

            nextPath = currentPath + [nextAction]
            nextCostToReach = problem.getCostOfActions(nextPath)
            logging.debug("Next path: %r", nextPath)
            logging.debug("nextCostToReach: %r", nextCostToReach)
            nextCostAStar = nextCostToReach + heuristic(nextState, problem)

            # Check if this new plan is not better than the previous plan, or if there is already a plan
            oldCostAStar = planToState.get(nextState, {}).get("costAStar", float("inf"))
            logging.debug("Old Plan: %r", oldCostAStar)
            if nextCostAStar >= oldCostAStar:
                continue

            # Add the successor state to the fringe, with the cost of reaching that state as the priority
            fringe.update(nextState, nextCostAStar)
            logging.debug("Added to fringe: %r", nextState)
            logging.debug("With A star cost: %r", nextCostAStar)


        

    logging.debug("")
    logging.debug("Path Found")
    logging.debug("Final Path: %r", currentPath)
    return currentPath


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

# Old version of A Star Search, doesn't work properly
# Gonna try to restructure it a bit


def aStarSearchV1(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""

    return []

    # Q4: A* search

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
        # if currentState not in visitedStates:
        visitedStates.append(currentState)
        successors = problem.getSuccessors(currentState)
        # For each successor of the current state
        for successor in successors:
            nextState, nextAction, _ = successor
            logging.debug("")
            logging.debug("Analyzing successor: %r", successor)

            nextPath = currentPath + [nextAction]
            nextCostToReach = problem.getCostOfActions(nextPath)
            logging.debug("Next path: %r", nextPath)
            logging.debug("nextCostToReach: %r", nextCostToReach)
            nextCostAStar = nextCostToReach + heuristic(nextState, problem)

            # Check if this new plan is not better than the previous plan, or if there is already a plan
            oldCostAStar = planToState.get(nextState, {}).get("costAStar", float("inf"))
            logging.debug("Old Plan: %r", oldCostAStar)
            if nextCostAStar >= oldCostAStar:
                continue

            # Add the successor state to the fringe, with the cost of reaching that state as the priority
            fringe.update(nextState, nextCostAStar)
            # Add the plan to reach this state to the dictionary
            planToState[nextState] = {
                "previousState": currentState,
                "previousPath": currentPath,
                "currentPath": nextPath,
                "costAStar": nextCostAStar,
            }
            logging.debug("Added to fringe: %r", nextState)
            logging.debug("With A star cost: %r", nextCostAStar)
            logging.debug("Plan to reach this state: %r", planToState[nextState])
            logging.debug("The entire fringe: %r", fringe.heap)

        # If the fringe is empty, we have no solution
        if fringe.isEmpty():
            logging.info("No solution found, returning longest path attempted")
            return longestFailure

        # Popping the fringe to decide what is the next situation to deal with
        nextState = fringe.pop()

        # Move to it
        currentState = nextState
        currentPath = planToState[currentState]["currentPath"]

        if len(currentPath) > len(longestFailure):
            longestFailure = currentPath.copy()

    logging.debug("")
    logging.debug("Path Found")
    logging.debug("Final Path: %r", currentPath)
    return currentPath


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
n = Directions.NORTH
e = Directions.EAST
s = Directions.SOUTH
w = Directions.WEST
