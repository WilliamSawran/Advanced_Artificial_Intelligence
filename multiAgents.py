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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newFood = successorGameState.getFood()
	newGhostStates = successorGameState.getGhostStates()
        score=successorGameState.getScore()

        "*** YOUR CODE HERE ***"
        def sum_food_proximity(cur_pos, food_positions, norm=False):
            food_distances = []
            for food in food_positions:
                food_distances.append(util.manhattanDistance(food, cur_pos))
            if norm:
                return normalize(sum(food_distances) if sum(food_distances) > 0 else 1)
            else:
                return sum(food_distances) if sum(food_distances) > 0 else 1

            score = successorGameState.getScore()

        def ghost_stuff(cur_pos, ghost_states, radius, scores):
            num_ghosts = 0
            for ghost in ghost_states:
                if util.manhattanDistance(ghost.getPosition(), cur_pos) <= radius:
                    scores -= 30
                    num_ghosts += 1
            return scores

        def food_stuff(cur_pos, food_pos, cur_score):
            new_food = sum_food_proximity(cur_pos, food_pos)
            cur_food = sum_food_proximity(currentGameState.getPacmanPosition(), currentGameState.getFood().asList())
            new_food = 1 / new_food
            cur_food = 1 / cur_food
            if new_food > cur_food:
                cur_score += (new_food - cur_food) * 3
            else:
                cur_score -= 20

            next_food_dist = closest_dot(cur_pos, food_pos)
            cur_food_dist = closest_dot(currentGameState.getPacmanPosition(), currentGameState.getFood().asList())
            if next_food_dist < cur_food_dist:
                cur_score += (next_food_dist - cur_food_dist) * 3
            else:
                cur_score -= 20
            return cur_score

        def closest_dot(cur_pos, food_pos):
            food_distances = []
            for food in food_pos:
                food_distances.append(util.manhattanDistance(food, cur_pos))
            return min(food_distances) if len(food_distances) > 0 else 1

        def normalize(distance, layout):
            return distance

	return food_stuff(newPos, newFood.asList(), ghost_stuff(newPos, newGhostStates, 2, score))

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

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
        """
        "*** YOUR CODE HERE ***"
        # Set PACMAN as agent 0.
        self.PACMAN = 0
        # FInd the desired action by minimax, starting on depth 0.
        action = self.max_agent(gameState, 0)
        return action

    def max_agent(self, state, depth):
        # Terminal test.
        if state.isLose() or state.isWin():
            return state.getScore()

        # Collect all of PACMAN's actions.
        actions = state.getLegalActions(self.PACMAN)
        # Set best score to minus infinity to only get higher score.
        best_score = float("-inf")

        # Initialized preferred action is STOP when nothing is calculated.
        preferred = Directions.STOP
        # Loop through actions.
        for action in actions:
            # Calculate PACMAN's score recursively in the minimax three.
            # Goes to next level with agent number 1, which is the first ghost.
            score = self.min_agent(state.generateSuccessor(self.PACMAN, action), depth, 1)
            # Sets current max score to best score,
            # and the corresponding action as preferred.
            if score > best_score:
                best_score = score
                preferred = action
        # If we're at the top, return the preferred action.
        # Else, return the calculated best score.
        if depth == 0:
            return preferred
        else:
            return best_score

    def min_agent(self, state, depth, agent):
        # Terminal test.
        if state.isLose() or state.isWin():
            return state.getScore()

        # Prepare next agent. If it is PACMAN, we know this is the last ghost.
        next_agent = agent + 1
        if agent == state.getNumAgents() - 1:
            next_agent = self.PACMAN

        # Collect all of ghost's actions.
        actions = state.getLegalActions(agent)
        # Set best score to plus infinity, to only get lower score.
        best_score = float("inf")

        # Loop through actions.
        for action in actions:
            # When agent is the last ghost.
            if next_agent == self.PACMAN:
                if depth == self.depth - 1:
                    # And we're at the leaf nodes, set score form evaluationFunction.
                    score = self.evaluationFunction(state.generateSuccessor(agent, action))
                else:
                    # If not at leaf nodes, the next is PACMAN, commence one layer.
                    score = self.max_agent(state.generateSuccessor(agent, action), depth+1)
            # Every ghost but the last one.
            else:
                score = self.min_agent(state.generateSuccessor(agent, action), depth, next_agent)

            # New best score.
            best_score = min(score, best_score)

        return best_score

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # PACMAN index 0 and beta and alpha at +/- inf.
        self.PACMAN = 0
        action = self.max_agent(gameState, 0, float("-inf"), float("inf"))
	return action

    def max_agent(self, state, depth, alpha, beta):
        # Terminal test.
        if state.isLose() or state.isWin():
            return state.getScore()

        # Collect all of PACMAN's actions.
        actions = state.getLegalActions(self.PACMAN)
        # Set best score to minus infinity to only get higher score.
        best_score = float("-inf")

        # Initialized preferred action is STOP when nothing is calculated.
        preferred = Directions.STOP

        # Loop through actions.
        for action in actions:
            # Calculate PACMAN's score recursively in the minimax three.
            # Goes to next level with agent number 1, which is the first ghost.
            score = self.min_agent(state.generateSuccessor(self.PACMAN, action), depth, 1, alpha, beta)

            # Sets current max score to best score,
            # and the corresponding action as preferred.
            if score > best_score:
                best_score = score
                preferred = action
            # New alpha.
            alpha = max(alpha, best_score)
            if best_score > beta:
                return best_score
        # If we're at the top, return the preferred action.
        # Else, return the calculated best score.
        if depth == 0:
            return preferred
        else:
            return best_score

    def min_agent(self, state, depth, agent, alpha, beta):
        # Terminal test.
        if state.isLose() or state.isWin():
            return state.getScore()

        # Prepare next agent. If it is PACMAN, we know this is the last ghost.
        next_agent = agent + 1
        if agent == state.getNumAgents() - 1:
            next_agent = self.PACMAN

        # Collect all of ghost's actions.
        actions = state.getLegalActions(agent)
        # Set best score to plus infinity, to only get lower score.
        best_score = float("inf")

        # Loop through actions.
        for action in actions:
            # When agent is the last ghost.
            if next_agent == self.PACMAN:
                if depth == self.depth - 1:
                    # And we're at the leaf nodes, set score form evaluationFunction.
                    score = self.evaluationFunction(state.generateSuccessor(agent, action))
                else:
                    # If not at leaf nodes, the next is PACMAN, commence one layer.
                    score = self.max_agent(state.generateSuccessor(agent, action), depth + 1, alpha, beta)
            # Every ghost but the last one.
            else:
                score = self.min_agent(state.generateSuccessor(agent, action), depth, next_agent, alpha, beta)

            # New best score and new beta.
            best_score = min(score, best_score)
            beta = min(beta, best_score)

            if best_score < alpha:
                return best_score

        return best_score


        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

       	" Max Value "
        def max_value(gameState, depth):
            " Cases checking "
            actionList = gameState.getLegalActions(0) # Get actions of pacman
            if len(actionList) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)
            
            " Initializing the value of v and action to be returned "
            v = -(float("inf"))
            goAction = None

            for thisAction in actionList:
                successorValue = exp_value(gameState.generateSuccessor(0, thisAction), 1, depth)[0]

                " max(v, successorValue) "
                if (v < successorValue):
                    v, goAction = successorValue, thisAction

            return (v, goAction)

        " Exp Value "
        def exp_value(gameState, agentID, depth):
            " Cases checking "
            actionList = gameState.getLegalActions(agentID) # Get the actions of the ghost
            if len(actionList) == 0:
              return (self.evaluationFunction(gameState), None)

            " Initializing the value of v and action to be returned "
            v = 0
            goAction = None

            for thisAction in actionList:
                if (agentID == gameState.getNumAgents() - 1):
                    successorValue = max_value(gameState.generateSuccessor(agentID, thisAction), depth + 1)[0]
                else:
                    successorValue = exp_value(gameState.generateSuccessor(agentID, thisAction), agentID + 1, depth)[0]

                probability = successorValue/len(actionList)
                v += probability

            return (v, goAction)

	return max_value(gameState, 0)[1]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    ghostList = currentGameState.getGhostStates() 
    foods = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    # Return based on game state
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")
    # Populate foodDistList and find minFoodDist
    foodDistList = []
    for each in foods.asList():
        foodDistList = foodDistList + [util.manhattanDistance(each, pacmanPos)]
    minFoodDist = min(foodDistList)
    # Populate ghostDistList and scaredGhostDistList, find minGhostDist and minScaredGhostDist
    ghostDistList = []
    scaredGhostDistList = []
    for each in ghostList:
        if each.scaredTimer == 0:
            ghostDistList = ghostDistList + [util.manhattanDistance(pacmanPos, each.getPosition())]
        elif each.scaredTimer > 0:
            scaredGhostDistList = scaredGhostDistList + [util.manhattanDistance(pacmanPos, each.getPosition())]
    minGhostDist = -1
    if len(ghostDistList) > 0:
        minGhostDist = min(ghostDistList)
    minScaredGhostDist = -1
    if len(scaredGhostDistList) > 0:
        minScaredGhostDist = min(scaredGhostDistList)
    # Evaluate score
    score = scoreEvaluationFunction(currentGameState)
    # Distance to closest food
    score = score + (-1.5 * minFoodDist)
    # Distance to closest ghost
    score = score + (-2 * (1.0 / minGhostDist))
    # Distance to closest scared ghost
    score = score + (-2 * minScaredGhostDist)
    # Number of capsules
    score = score + (-20 * len(capsules))
    # Number of food
    score = score + (-4 * len(foods.asList()))
    return score

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

