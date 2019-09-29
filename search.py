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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's next_node:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE IF YOU WANT TO PRACTICE ***"
    from util import Stack
    from game import Directions

    open = Stack()
    closed = []
    open.push((problem.getStartState(), []))

    while not open.isEmpty():

        current_node, actions = open.pop()

        # if current state is the goal state
        # return list of actions
        if problem.isGoalState(current_node):
            return actions

        if current_node not in closed:
            # expand current node
            # add current node to closed list
            expand = problem.getSuccessors(current_node)
            closed.append(current_node)
            for location, direction, cost in expand:
                # if the location has not been visited, put into open list
                if (location not in closed):
                    open.push((location, actions + [direction]))

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE IF YOU WANT TO PRACTICE ***"
    from util import Queue
    from game import Directions

    open = Queue()
    closed = []

    open.push((problem.getStartState(), []))

    while not open.isEmpty():
        current_node, actions = open.pop()
        # if current state is the goal state
        # return list of actions
        if problem.isGoalState(current_node):
            return actions

        if current_node not in closed:
            expand = problem.getSuccessors(current_node)
            closed.append(current_node)
            for location, direction, cost in expand:
                # if the location has not been visited, put into open list
                if (location not in closed):
                    open.push((location, actions + [direction]))

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE IF YOU WANT TO PRACTICE ***"
    queue = util.PriorityQueueWithFunction(lambda x: x[2])

    visited = []
    actions = []
    cost = 0

    start = problem.getStartState()
    queue.push((start, None, cost))

    parents = {}
    parents[(start, None, 0)] = None

    while not queue.isEmpty():

        current_node = queue.pop()

        if problem.isGoalState(current_current_node):
            break
        else:
            current_node_state = current_current_node
            if current_node_state not in visited:
                visited.append(current_node_state)
            else:
                continue

            expand = problem.getSuccessors(current_node_state)
            for state in expand:
                cost = current_node[2] + state[2]

                # if the location has not been visited, put into open list
                if (state[0] not in visited):
                    queue.push((state[0], state[1], cost))
                    parents[(state[0], state[1])] = current_node

    child = current_node

    while (child != None):
        actions.append(child[1])
        if location != start:
            child = parents[(location, child[1])]
        else:
            child = None
    actions.reverse()
    return actions[1:]

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE IF YOU WANT TO PRACTICE ***"
    from util import Queue
    from game import Directions

    open = util.PriorityQueue()
    closed = []
    cost = 0
    actions = []
    open.push((problem.getStartState(), []), 0)

    while not open.isEmpty():

        current_node, actions = open.pop()

        if problem.isGoalState(current_node):
            return actions

        if current_node not in closed:

            expand = problem.getSuccessors(current_node)
            closed.append(current_node)
            for location, direction, tmp_cost in expand:
                cost = problem.getCostOfActions(actions +[direction]) + heuristic(location,problem)
                if (location not in closed):
                    open.push((location, actions + [direction]),cost)

    return actions

    util.raiseNotDefined()

def iterativeDeepeningSearch(problem):
    """Search the deepest node in an iterative manner."""
    "*** YOUR CODE HERE FOR TASK 1 ***"
    from util import Stack

    open = Stack()
    limit = 1

    while open.isEmpty():

        open.push(([problem.getStartState()], [], 0))

        while not open.isEmpty():
            (visited_nodes, actions, cost) = open.pop()
            current_node = visited_nodes[-1]
            if problem.isGoalState(current_node):
                return actions

            elif len(visited_nodes) < limit:
                next_node = problem.getSuccessors(current_node)
                for (location, direction, next_cost) in next_node:
                    if location not in visited_nodes:
                        open.push((visited_nodes + [location], actions + [direction], cost + next_cost))

        limit += 1
    return actions

def waStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has has the weighted (x 2) lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE FOR TASK 2 ***"
    from util import PriorityQueue

    open = PriorityQueue()
    closed = []

    w = 2
    weighted_h = w * heuristic(problem.getStartState(), problem)

    open.push((problem.getStartState(), [], 0), weighted_h)

    (current_node, actions, cost) = open.pop()

    closed.append((current_node, cost + heuristic(problem.getStartState(), problem)))

    while not problem.isGoalState(current_node):
      next_node = problem.getSuccessors(current_node)
      for location, direction, next_cost in next_node:
          visited = False

          for (visited_node, visited_cost) in closed:
              if (location == visited_node) and (cost + next_cost >= visited_cost):
                  visited = True
                  break

          if not visited:
              new_state = (location, actions + [direction], cost + next_cost)
              open.push(new_state, cost + next_cost + heuristic(location, problem))
              closed.append((location, cost + next_cost))

      (current_node, actions, cost) = open.pop()

    return actions

    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
wastar = waStarSearch
