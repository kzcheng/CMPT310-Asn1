# searchAgents.py
# ---------------
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
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

import logging
from typing import List, Tuple, Any
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import pacman

logging.basicConfig(level=logging.INFO, format='%(message)s')


class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################


class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None:
            raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state)  # Makes a new search problem
        self.actions = self.searchFunction(problem)  # Find a path
        if self.actions == None:
            self.actions = []
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem):
            print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self):
            self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP


class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn=lambda x: 1, goal=(1, 1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None:
            self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display):  # @UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist)  # @UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None:
            return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += self.costFn((x, y))
        return cost


class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """

    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        def costFn(pos): return .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)


class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """

    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        def costFn(pos): return 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)


def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################


class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    # Q5: Finding All the Corners

    # Note: Turns out, creating a class to store the state isn't a good idea.
    # Since classes are more like pointers towards the location of the object, two classes can be "different" even if the contents are the same.
    # It's best to keep it simple and just use tuples instead.

    # State Tuple = (position, corners)
    # state[0] = Position of Pacman
    # state[1] = True if the corner have been visited ((1, 1), (1, top), (right, 1), (right, top))

    def __init__(self, startingGameState: pacman.GameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height - 2, self.walls.width - 2
        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded = 0  # DO NOT CHANGE; Number of search nodes expanded

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        # The state should include the current position of Pacman, and whether the 4 corners have been visited
        return (self.startingPosition, (False, False, False, False))

    def isGoalState(self, state: any):
        """
        Returns whether this search state is a goal state of the problem.
        """
        # The goal state is fulfilled when all corners have been visited
        # This checks if all values in state[1] are True
        return all(state[1])

    def getSuccessors(self, state: any):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            x, y = state[0]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextPosition = (nextx, nexty)

                # Change the flag indicating if a corner has been reached to True if the next position is a corner
                # Because the flags stored in state[1] is in a tuple, we need to convert it to a list first
                cornerReached = list(state[1])
                for i, corner in enumerate(self.corners):
                    if nextPosition == corner:
                        cornerReached[i] = True

                # Then convert it back to a tuple
                # Combine with the next position to form the next state
                nextState = (nextPosition, tuple(cornerReached))
                cost = 1  # Forced to be 1 for this problem
                successors.append((nextState, action, cost))

        self._expanded += 1  # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None:
            return 999999
        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
        return len(actions)


def getManhattanDistance(positionA, positionB):
    return abs(positionA[0] - positionB[0]) + abs(positionA[1] - positionB[1])


def cornersHeuristic(state: Any, problem: CornersProblem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the shortest path from the state to a goal of the problem; i.e.  it should be admissible.
    """
    corners = problem.corners  # These are the corner coordinates

    # For a simple heuristic like this, we should find the manhattan distance to the closest corner that we haven't flagged yet, then find the shortest path to reach the remaining corners, assuming the map is empty.

    # To do this, we first order the corners clockwise, with the first corner being the closest one. We will call those corners ABCD for now.
    # After A is reached, if B still needs to be reached, we go to it, then we go to C, then D, if any needs to be reached.
    # If B can be skipped, we check D then C.
    # If B and C can both be skipped, we go directly to D.

    # By default, the order of the corners are ((1, 1), (1, top), (right, 1), (right, top)).

    # To begin, find the closest corner first, and the distance to reach it.
    position = state[0]
    closestCornerID = -1
    distanceToClosestCorner = float('inf')

    # For each corner
    for i, corner in enumerate(corners):
        # If we have already visited it, skip it
        if state[1][i]:
            continue
        distanceToCorner = getManhattanDistance(position, corner)
        if (distanceToCorner < distanceToClosestCorner):
            closestCornerID = i
            distanceToClosestCorner = distanceToCorner

    # If we can't find a corner that we haven't visited before, it likely means we have already solved the problem, which means we should return 0.
    if closestCornerID == -1:
        return 0

    # This switch includes the order we should check each corner. With the first corner being the closest corner, and the remaining ones being ordered clock wise.
    switch = {
        0: (0, 1, 3, 2),
        1: (1, 3, 2, 0),
        2: (3, 2, 0, 1),
        3: (2, 0, 1, 3),
    }
    cornersCW = (
        corners[switch[closestCornerID][0]],
        corners[switch[closestCornerID][1]],
        corners[switch[closestCornerID][2]],
        corners[switch[closestCornerID][3]],
    )
    cornersCWFlag = (
        state[1][switch[closestCornerID][0]],
        state[1][switch[closestCornerID][1]],
        state[1][switch[closestCornerID][2]],
        state[1][switch[closestCornerID][3]],
    )

    totalDistance = distanceToClosestCorner
    position = cornersCW[0]

    if cornersCWFlag[1] == False:
        totalDistance += getManhattanDistance(position, cornersCW[1])
        position = cornersCW[1]
        if cornersCWFlag[2] == False:
            totalDistance += getManhattanDistance(position, cornersCW[2])
            position = cornersCW[2]
        if cornersCWFlag[3] == False:
            totalDistance += getManhattanDistance(position, cornersCW[3])
            position = cornersCW[3]
    elif cornersCWFlag[3] == False:
        totalDistance += getManhattanDistance(position, cornersCW[3])
        position = cornersCW[3]
        if cornersCWFlag[2] == False:
            totalDistance += getManhattanDistance(position, cornersCW[2])
            position = cornersCW[2]
    elif cornersCWFlag[2] == False:
        totalDistance += getManhattanDistance(position, cornersCW[2])
        position = cornersCW[2]

    return totalDistance  # Default to trivial solution


class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"

    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem


class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """

    def __init__(self, startingGameState: pacman.GameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0  # DO NOT CHANGE
        self.heuristicInfo = {}  # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1  # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append((((nextx, nexty), nextFood), direction, 1))
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x, y = self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost


class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"

    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem


# Given a position and a list of food coordinates, find the furthest food from the position, and remove that food from the list
def findFurthestFood(position, foodList):
    if not foodList:
        return None, foodList
    furthestFood = None     # The furthest food from pacman
    furthestFoodDistance = 0    # The distance to the furthest food from pacman
    for food in foodList:
        distance = getManhattanDistance(position, food)
        if distance > furthestFoodDistance:
            furthestFood = food
            furthestFoodDistance = distance
    newFoodList = foodList[:]
    newFoodList.remove(furthestFood)
    return furthestFood, newFoodList


def convertFoodSearchToPositionSearch(foodSearchProblem, startPosition, goalPosition):
    """
    Converts a FoodSearchProblem to a PositionSearchProblem targeting a specific food position.

    foodSearchProblem: The FoodSearchProblem instance.
    startPosition: The starting position of Pacman.
    goalPosition: The position of the food to target.

    Returns: A PositionSearchProblem instance.
    """
    # Extract the current game state from the FoodSearchProblem
    gameState = foodSearchProblem.startingGameState

    # Create a new PositionSearchProblem with the start position and the goal position
    positionSearchProblem = PositionSearchProblem(
        gameState=gameState,
        goal=goalPosition,
        start=startPosition,  # Pacman's starting position
        warn=False,
        visualize=False
    )

    return positionSearchProblem


def foodHeuristic(state: Tuple[Tuple, List[List]], problem: FoodSearchProblem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your search may have a but our your heuristic is not admissible!  On the
    other hand, inadmissible heuristics may find optimal solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state

    # Q7: Eating All The Dots

    # Initialize heuristicInfo['calculatedPathCosts'] if it doesn't exist
    if 'calculatedPathCosts' not in problem.heuristicInfo:
        problem.heuristicInfo['calculatedPathCosts'] = {}

    # Plan for heuristic:
    # So, there are three important locations that pacman will need to visit. The current location pacman is at, the dot that is furthest away from pacman, and the dot that is the furthest away from that dot.
    # Those 3 locations form a "triangle", and is roughly the furthest 3 places pacman must visit. So I think recording those 2 dots, and let pacman visit the closer one, than the further one would work.

    # logging.getLogger().setLevel(logging.DEBUG)

    hCost = 0   # Total estimated heuristic cost to be returned
    foodList = foodGrid.asList()    # List of food coordinates

    logging.debug(f'foodList: {foodList}')

    # If the food list is empty, that means pacman has already reached the goal state, so return 0
    if not foodList:
        return hCost

    # New idea:
    # What if we just A star path find from the first food to the second one after reaching one food?

    # Food B is the furthest food, and food A is the food that is furthest from that
    # Because food B must be the furthest one away, food A is always closer to the starting position than food B
    foodB, foodList = findFurthestFood(position, foodList)
    if not foodB:
        return hCost
    foodA, foodList = findFurthestFood(foodB, foodList)
    if not foodA:
        return hCost + getManhattanDistance(position, foodB)

    # Find manhattan distance to food A and move there
    if (position, foodA) in problem.heuristicInfo['calculatedPathCosts']:
        hCost += problem.heuristicInfo['calculatedPathCosts'][(position, foodA)]
    else:
        hCost += getManhattanDistance(position, foodA)

    if (foodA, foodB) in problem.heuristicInfo['calculatedPathCosts']:
        hCost += problem.heuristicInfo['calculatedPathCosts'][(foodA, foodB)]
    else:
        # Now, path find from food A to food B
        # But do it hardcore with A Star
        problemFoodAToB = convertFoodSearchToPositionSearch(problem, foodA, foodB)
        path = search.aStarSearch(problemFoodAToB, heuristic=manhattanHeuristic)
        hCost += len(path)
        problem.heuristicInfo['calculatedPathCosts'][(foodA, foodB)] = hCost

    logging.getLogger().setLevel(logging.INFO)

    return hCost


class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"

    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' % t)
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print('Path found with cost %d.' % len(self.actions))

    def findPathToClosestDot(self, gameState: pacman.GameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        # Q8: Suboptimal Search
        return search.breadthFirstSearch(problem)


class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def isGoalState(self, state: Tuple[int, int]):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """

        if state in self.food.asList():
            return True
        return False


def mazeDistance(point1: Tuple[int, int], point2: Tuple[int, int], gameState: pacman.GameState) -> int:
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
