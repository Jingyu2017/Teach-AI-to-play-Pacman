# mlLearningAgents.py
#Jingyu Li/mar-2018

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



from pacman import Directions
from game import Agent
import random
import game
import util


#The aim of this work is to implement Q-learning algorithm 
#in Pacman world.
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        # These values are either passed from the command line or are
        # set to the default values above. 
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        self.scoreTeacker = 0
        self.stateTracker = None
        #use a  dictionary to keep key-value pairs 
        #where (GameState,action) altogether serves as the key of dictionary.
        self.qValues={}
        
        
    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value
        
    def getEpsilon(self):
        return self.epsilon

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value
        
    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts
    
    def setScoreTracker(self,score):
        #save the score in current episode for later use 
        #basically the same as what has been given in previous practical work(banditAgent)
        self.scoreTeacker = score
        
    def getScoreTracker(self):
        #Accessor functions for the score from the previous state
        #basically the same as what has been given in previous practical work(banditAgent)
        return self.scoreTeacker
    
    def getFoodLocations(self,state):
    #extract the coordinates of foods to a list.
    #First, we have to convert the grid instance to a list to make it iterable
        self.FoodList = [i for i in state.getFood()]
    #get the height and width of the grid
        self.GridHeight=len(self.FoodList)
        self.GridWidth=len(self.FoodList[0])
        self.FoodLocations=[(i,j) for j in range(self.GridWidth) for i in range(self.GridHeight) if self.FoodList[i][j]==True]
        #For example, at the beginning of each game, there are two food items,
        #this function returns [(1, 1), (3, 3)]
        return self.FoodLocations  
    
    
    def setStateTracker(self,GameState):
        #save the state of the present episode for later use 
        # State is represented by a tuple of three items: 
        #(PacmanPosition,GhostPositions,food). 
        #For example, at the beginning of each game, 
        #the state will be ((1,2), [(1.0, 3.0)], [(1, 1), (3, 3)])
        self.stateTracker = GameState
    
    def getStateTracker(self):
        #Accessor functions for the previous state
        return self.stateTracker  
    
    def getQ(self,GameState,action):
        #getQ accesses the Q-value of a (GameState,action) pair which is passed to this function
        #if the (GameState,action) pair dosen't exist,
        #the corresponding Q-value is initialized with 0.
        return self.qValues.get((GameState,action),0.0)    

    def greedyPick(self,GameState,legalActions):
        # Given the current state and a list of legal actions, 
        #use epilon-greedy policy to pick action
        if random.random()<self.epsilon:   #0.0 <= x < 1.0
            #the algorithm is run allowing the agent to explore 
            #(to take stochastic actions with a probability of epsilon)
            action=random.choice(legalActions)
        else:
            #returns the action with largest Q-value
            q = [self.getQ(GameState,a) for a in legalActions]
            maxQ = max(q)
            count = q.count(maxQ)
            #in case taht we have more than 1 action with largest Q-value
            #pick one of them randomly
            if count>1:
                best=[i for i in range(len(legalActions)) if q[i]==maxQ]
                best_index=random.choice(best)
            else:
                best_index = q.index(maxQ)
            action = legalActions[best_index]
        return action
    
    def qlearning(self,state1,state2,lastAction,legalActions,change_in_score):          
        max_Q2 = max([self.getQ(state2,a) for a in legalActions])
        old_qvalue=self.qValues.get((state1,lastAction),0.0) 
        # Q,action values,indexed by state and action,initially zero
        self.qValues[(state1,lastAction)] = old_qvalue + self.alpha * (change_in_score + self.gamma * max_Q2 -old_qvalue)      
        
    
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):
       
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        
        PacmanPosition=state.getPacmanPosition() # e.g. (1,2)
        GhostPositions=state.getGhostPositions()     # e.g. [(1.0, 3.0)]      
        food=self.getFoodLocations(state)  #e.g. [(1, 1), (3, 3)]
        
        GameState = (PacmanPosition,)+tuple(GhostPositions)+tuple(food)
        #e.g. ((1,2), [(1.0, 3.0)], [(1, 1), (3, 3)])
              
        lastGameState = self.getStateTracker()      
       
        #save the state in current episode      
        self.setStateTracker(GameState)

        # get last action the agent took,in order to update Q-value
        lastAction = state.getPacmanState().configuration.direction
        
        #how the agent was rewarded (or punished) from the action it took last time?
        current_score = state.getScore()         
        change_in_score = current_score-self.getScoreTracker()
        #save the score in current episode
        self.setScoreTracker(current_score)  
                
      
        #Update Q-values by Q-function based on how well the learner plays the game
        self.qlearning(lastGameState,GameState,lastAction,legal,change_in_score)
        
        # Now pick what action to take. 
        pick = self.greedyPick(GameState,legal)
        
        return pick
            

    # Handle the end of episodes
    # This is called by the game after a win or a loss.
    def final(self, state):
        # OUTPUT is a switch,if set to True, scores for each non-training episode
        # will be written to a text file for performance analysis
        OUTPUT=False
        if OUTPUT:
            if self.getEpisodesSoFar() >= self.getNumTraining():
                name2save = "Results_" + str(self.getNumTraining()) + ".txt"
                command = "./"  + name2save
                file2write = open(command,"a")                 
                file2write.write("%d\n" % (state.getScore()))
                file2write.close()
                
        #after a win or a loss,legal actions returned by state API is a empty list
        #so here we regard all directions as "legal" in this phase to update q-values
        all_direc = [Directions.WEST,Directions.EAST,Directions.SOUTH,Directions.NORTH]
         
        #access information we need to update q-values 
        #in order to learn how to avoid the ghost and how to win the game
        PacmanPosition=state.getPacmanPosition() 
        GhostPositions=state.getGhostPositions()     
        food=self.getFoodLocations(state)
        
        change_in_score = state.getScore()-self.getScoreTracker()
        lastAction = state.getPacmanState().configuration.direction
        
        GameState = (PacmanPosition,)+tuple(GhostPositions)+tuple(food)
        lastGameState = self.getStateTracker() 
        
        self.qlearning(lastGameState,GameState,lastAction,all_direc,change_in_score)
        
        
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)


