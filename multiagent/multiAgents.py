# multiAgents.py
# --------------
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


import random
import util

from game import Agent
from util import manhattanDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPositions = successorGameState.getGhostPositions()
        oldPos = currentGameState.getPacmanPosition()
        oldFood = currentGameState.getFood().asList()

        res = 0

        for ghostCoordinate in newGhostPositions:
            if manhattanDistance(newPos, ghostCoordinate) == 2:
                res -= 5000
                return res

        if newPos in oldFood:
            res += 50

        if oldPos == newPos:
            res -= 20

        newFoodDistances = []
        if len(oldFood) > 0:
            for f in oldFood:
                newFoodDistances.append((manhattanDistance(newPos, f), f))
            closest_food = min(newFoodDistances)
            if manhattanDistance(newPos, closest_food[1]) < manhattanDistance(oldPos, closest_food[1]):
                res += 40

        return res


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, gameState, agentIndex, depth):
        if agentIndex >= gameState.getNumAgents():
            depth += 1
            agentIndex = 0

        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        else:
            return self.minValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        best_value = -float("inf")
        actions = gameState.getLegalActions(agentIndex)

        for action in actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            value = max(best_value, self.minimax(successor_game_state, agentIndex + 1, depth))
            best_value = max(best_value, value)

        return best_value

    def minValue(self, gameState, agentIndex, depth):
        best_value = float("inf")
        actions = gameState.getLegalActions(agentIndex)

        for action in actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            value = min(best_value, self.minimax(successor_game_state, agentIndex + 1, depth))
            best_value = min(best_value, value)

        return best_value

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        pacman_agent_index = 0
        max_value = -float("inf")
        actions = gameState.getLegalActions(pacman_agent_index)
        best_action = None

        for action in actions:
            successor_game_state = gameState.generateSuccessor(pacman_agent_index, action)
            value = self.minimax(successor_game_state, 1, 0)

            if value > max_value:
                max_value = value
                best_action = action

        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alpha_beta(self, game_state, alpha, beta, agent_index, depth):
        if agent_index >= game_state.getNumAgents():
            depth += 1
            agent_index = 0

        if depth == self.depth or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state)

        if agent_index == 0:
            return self.maxValue(game_state, alpha, beta, agent_index, depth)
        else:
            return self.minValue(game_state, alpha, beta, agent_index, depth)

    def maxValue(self, gameState, alpha, beta, agentIndex, depth):
        value = -float("inf")
        legal_actions = gameState.getLegalActions(agentIndex)

        for action in legal_actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            value = max(value, self.alpha_beta(successor_game_state, alpha, beta, agentIndex + 1, depth))

            if value > beta:
                return value
            alpha = max(alpha, value)

        return value

    def minValue(self, gameState, alpha, beta, agentIndex, depth):
        value = float("inf")
        legal_actions = gameState.getLegalActions(agentIndex)

        for action in legal_actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            value = min(value, self.alpha_beta(successor_game_state, alpha, beta, agentIndex + 1, depth))

            if value < alpha:
                return value
            beta = min(beta, value)

        return value

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        agent_index = 0
        alpha = -float("inf")
        beta = float("inf")
        actions = gameState.getLegalActions(agent_index)
        best_action = None
        for action in actions:
            successorGameState = gameState.generateSuccessor(agent_index, action)
            val = self.alpha_beta(successorGameState, alpha, beta, 1, 0)
            if val > alpha:
                alpha = val
                best_action = action
        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, game_state, agent_index, depth):
        if agent_index >= game_state.getNumAgents():
            depth += 1
            agent_index = 0

        if depth == self.depth or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state)

        if agent_index == 0:
            return self.max_value(game_state, agent_index, depth)
        else:
            return self.expected_value(game_state, agent_index, depth)

    def max_value(self, game_state, agent_index, depth):
        value = -float("inf")
        legal_actions = game_state.getLegalActions(agent_index)

        for action in legal_actions:
            successor_game_state = game_state.generateSuccessor(agent_index, action)
            value = max(value, self.expectimax(successor_game_state, agent_index + 1, depth))

        return value

    def expected_value(self, game_state, agent_index, depth):
        value = 0
        legal_actions = game_state.getLegalActions(agent_index)

        p = 1.0 / float(len(legal_actions))

        for action in legal_actions:
            successor_game_state = game_state.generateSuccessor(agent_index, action)
            value += p * self.expectimax(successor_game_state, agent_index + 1, depth)

        return value

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        agent_index = 0
        max_value = -float("inf")
        legal_actions = gameState.getLegalActions(agent_index)

        best_action = None

        for action in legal_actions:
            successor_game_state = gameState.generateSuccessor(agent_index, action)
            value = self.expectimax(successor_game_state, 1, 0)

            if value > max_value:
                max_value = value
                best_action = action

        return best_action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    score = 0
    pacman_pos = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    food_distances = []

    for food in food_list:
        food_distances.append(manhattanDistance(pacman_pos, food))

    if len(food_distances) > 0:
        score += 10 * 1.0 / min(food_distances)
        score += 5 * 1.0 / len(food_list)

    ghost_positions = currentGameState.getGhostPositions()
    if manhattanDistance(pacman_pos, ghost_positions[0]) < 3:
        score -= 1000

    ghost_states = currentGameState.getGhostStates()
    scared_times = [ghostState.scaredTimer for ghostState in ghost_states]
    score += 10 * sum(scared_times)

    return score + 100 * currentGameState.getScore()


# Abbreviation
better = betterEvaluationFunction
