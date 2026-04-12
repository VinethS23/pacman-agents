# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util     


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """
        self.state = state
        self.pacman_position = state.getPacmanPosition()
        self.ghost_positions = state.getGhostPositions()
        self.ghost_states = state.getGhostStates()
        self.food = state.getFood()
        self.walls = state.getWalls()
        self.score = state.getScore()
        self.capsules = state.getCapsules()
        
    def __hash__(self):
        """
        Allow states to be keys of dictionaries.
        We use a tuple of features that uniquely identify a state.
        
        Return: 
            Hash value of the game state
        
        """
        return hash((self.pacman_position, tuple(self.ghost_positions), 
                    tuple(self.capsules), self.food.count()))
        
    
    def __eq__(self, otherState):
        """
        Allow states to be compared for equality.
        Two states are equal if all their key features are equal.
        
        Args:
            otherState: GameStateFeatures object
        
        Return: 
            Boolean stating if the two states are equal
        
        """
        if otherState is None:
            return False
        return (self.pacman_position == otherState.pacman_position and
                self.ghost_positions == otherState.ghost_positions and
                self.capsules == otherState.capsules and
                self.food.count() == otherState.food.count())


class QLearnAgent(Agent):



    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        self.episodesSoFar = 0
        
        self.q_values = {}  # Dictionary storing Q-table values
        self.action_counts = {}  # Counts each state-action pair 
        
        self.last_state = None
        self.last_action = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        # Reward gained between start and end state
        reward = endState.getScore() - startState.getScore()
        
        if endState.isWin():
            reward += 500  # Reward for winning
        elif endState.isLose():
            reward -= 500  # Penalty for losing
            
        return reward

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        # Return the stored Q-value, or 0 if we haven't seen this state-action pair
        return self.q_values.get((state, action), 0.0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        legal_actions = self.state.getLegalPacmanActions()
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)
            
        # If there are no legal actions, return 0
        if not legal_actions:
            return 0.0
            
        # Return the maximum Q-value attained by a legal action from this state
        return max([self.getQValue(state, action) for action in legal_actions])

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        
        old_q_value = self.getQValue(state, action) 
        max_next_q_value = self.maxQValue(nextState)
        
        # Q-learning update rule: Q(s,a) = Q(s,a) + alpha * [R(s) + gamma * max_a' Q(s',a') - Q(s,a)]        
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_next_q_value - old_q_value)
        
        # Update the Q-table with the new Q-value
        self.q_values[(state, action)] = new_q_value

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        # Initialize count if this state-action pair hasn't been seen before
        if (state, action) not in self.action_counts:
            self.action_counts[(state, action)] = 0
            
        self.action_counts[(state, action)] += 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        # Return the count, or 0 if we've never seen this state-action pair
        return self.action_counts.get((state, action), 0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        Uses a 'Least-pick' method (encourages exploration of less visited state,action pairs)

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """

        if counts < self.maxAttempts:
            return 1000.0 # High reward to encourage exploration
        else:
            return utility 

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # Getting the legal actions for the current state
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        self.state = state
        stateFeatures = GameStateFeatures(state)
        
        # If this isn't the first step in an episode, perform a learning update
        if self.last_state is not None and self.last_action is not None:
            reward = self.computeReward(self.last_state.state, state)
            self.learn(self.last_state, self.last_action, reward, stateFeatures)
        
        # Store this state for the next learning update
        self.last_state = stateFeatures
        
        # Use Epsilon-greedy exploration strategy
        if util.flipCoin(self.epsilon):
            # Take a random action (explore) with probability Epsilon
            chosen_action = random.choice(legal)
        else:
            # Otherwise, take the action with the highest Q-value (exploitation)
            action_values = []
            for action in legal:
                q_value = self.getQValue(stateFeatures, action)
                count = self.getCount(stateFeatures, action)
                adjusted_value = self.explorationFn(q_value, count) # Adjust q-value for under explored states
                action_values.append((adjusted_value, action))
            
            best_value = max(action_values)[0]
            best_actions = [a for v, a in action_values if v == best_value]
            
            # Break ties for instances with equal q-values
            chosen_action = random.choice(best_actions)
        
        self.updateCount(stateFeatures, chosen_action)
        
        # Store the action for the next learning update
        self.last_action = chosen_action
        
        return chosen_action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")
        
        # Perform a final learning update if we have a stored previous state and action
        if self.last_state is not None and self.last_action is not None:
            final_state_features = GameStateFeatures(state) 
            reward = self.computeReward(self.last_state.state, state)
            self.learn(self.last_state, self.last_action, reward, final_state_features)
        
        # Reset the last state and action for the next episode
        self.last_state = None
        self.last_action = None

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)

class QNetwork:
    def __new__(cls, input_size, hidden_size, output_size):
        import torch
        import torch.nn as nn

        class _QNetworkImpl(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                hidden = self.fc1(x)
                activated = self.relu(hidden)
                output = self.fc2(activated)
                return output

        return _QNetworkImpl(input_size, hidden_size, output_size)

class NNQAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        import torch.optim as optim

        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        self.episodesSoFar = 0

        self.action_counts = {}  # Counts each state-action pair -> from Qlearning agent

        self.actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        self.network = QNetwork(10, 64, 5)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.alpha)

        self.last_state = None
        self.last_action = None
        self.state = None
    

    # Accessor functions for controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        # Reward gained between start and end state
        reward = endState.getScore() - startState.getScore()

        if endState.isWin():
            reward += 500  # Reward for winning
        elif endState.isLose():
            reward -= 500  # Penalty for losing

        return reward

    # Methods for the Neural Network
    def get_features(self, state: GameStateFeatures):
        import torch

        pacman_position = state.pacman_position
        ghosts = state.ghost_states
        food = state.food
        capsules = state.capsules
        features = [state.pacman_position[0], state.pacman_position[1]]

        for i in range(2):
            if i < len(ghosts):
                ghost_pos = ghosts[i].getPosition()
                dx = ghost_pos[0] - pacman_position[0]
                dy = ghost_pos[1] - pacman_position[1]
                scared = ghosts[i].scaredTimer
            else:
                dx, dy, scared = 0, 0, 0
            features.extend([dx, dy, scared])

        features.append(food.count())
        features.append(len(capsules))

        return torch.tensor(features, dtype=torch.float32)

    def learn(self, state, action, reward, next_state):
        import torch

        action_index = self.actions.index(action)

        predicted_q = self.network(state)[action_index]

        with torch.no_grad():
            max_next_q = self.network(next_state).max()

        target = reward + self.gamma * max_next_q

        loss = (predicted_q - target) ** 2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        Args:
            state: the current state

        Returns:
            The action to take
        """
        import torch

        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        self.state = state
        state_features_obj = GameStateFeatures(state)
        state_features_tensor = self.get_features(state_features_obj)

        # Perform learning update if we have a stored previous state and action
        if self.last_state is not None and self.last_action is not None:
            reward = self.computeReward(self.last_state.state, state)
            last_state_tensor = self.get_features(self.last_state)
            self.learn(last_state_tensor, self.last_action, reward, state_features_tensor)

        self.last_state = state_features_obj

        # Use Epsilon-greedy exploration strategy
        if util.flipCoin(self.epsilon):
            # Take a random action (explore) with probability epsilon
            chosen_action = random.choice(legal)
        else:
            # Otherwise, take the action with the highest Q-value (exploitation)
            with torch.no_grad():
                q_values = self.network(state_features_tensor)
            action_values = []
            for i, action in enumerate(self.actions):
                if action in legal:
                    action_values.append((q_values[i].item(), action))

            if action_values:
                best_value = max(action_values)[0]
                best_actions = [a for v, a in action_values if v == best_value]
                chosen_action = random.choice(best_actions)
            else:
                chosen_action = random.choice(legal)

        self.last_action = chosen_action

        return chosen_action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Perform a final learning update with terminal state
        if self.last_state is not None and self.last_action is not None:
            reward = self.computeReward(self.last_state.state, state)
            # Terminal state has no future Q-value, so we use reward as target
            last_state_tensor = self.get_features(self.last_state)
            action_index = self.actions.index(self.last_action)
            predicted_q = self.network(last_state_tensor)[action_index]
            loss = (predicted_q - reward) ** 2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Reset the last state and action for the next episode
        self.last_state = None
        self.last_action = None

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)


