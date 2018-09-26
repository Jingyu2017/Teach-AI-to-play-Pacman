# Teach-AI-to-play-Pacman



##### You can train the robot and evaluate the learning results by running the following command in terminal

the learner get trained for 2000 episodes and then make it run 10 episodes based on what has been learnt so far by the agent. Once reaching. 2000 episodes, the update doesn't change the Q-values because it sets epsilon (the exploration rate) and alpha (the learning rate) to zero.

```
python pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid

python pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid -q

python pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid --frameTime 0

```

  

**Introduction** 

The aim of this work is to implement Q-learning algorithm to choose the policy that maximises the agent’s long term return in Pacman’s world. 



### Functions defined in MlLearningAgents.py

**def setScoreTracker(self,score):**
**def getScoreTracker(self):**

setScoreTracker save the score in current episode for later use and getScoreTracker access the score from the previous one. 

**def getFoodLocations(self,state):**
 Food is represented by a grid of letters (T or F), getFoodLocations extract the coordinates of foods to a list. 

 **def getStateTracker(self):** 

The idea of these two functions is similar to that of setScoreTracker and getStateTracker. The former one save the state of the present episode for later use and getStateTracker access it from the previous one. State is represented by a tuple of three items: (PacmanPosition,GhostPositions,food). For example, at the beginning of each game, the state will be ((1,2), [(1.0, 3.0)], [(1, 1), (3, 3)]) 

**def getQ(self,GameState,action):**
 A dictionary named qvalues keeps key-value pairs where (GameState,action) altogether serves as the key of dictionary. getQ accesses the Q-value of a (GameState,action) pair which is passed to this function. Q-values are initialized with 0. 

**def greedyPick(self,GameState,legalActions):**
 Given the current state and a list of legal actions that pacman can move, greedyPick returns the action with largest Q-value using epilon-greedy policy. 

**def qlearning(self,state1,state2,lastAction,legalActions,change_in_score):**
 qlearning adjusts q-values based on how well the learner plays the game. The Q- function is updated for every obtained sample using the q-learning update rule.

Once a certain amount of iterations is reached (e.g. 2000 episodes), then the
update doesn't change the Q-values because it sets epsilon(the exploration rate) and
alpha(the learning rate) to zero.

**def getAction(self, state):**
 The main job of getAction() is to decide what move Pacman should make. After accessing all the information needed for Q-function update, it calls fuction qlearning in order to learn the world around and calls def greedyPick to make move. 

**def final(self, state):**
 This function is called by the game when a Pacman has been killed or when 

Pacman wins. getAction() is not called once the episode is over. It keeps track of the number of episodes and sets epsilon and alpha to zero when a certain amount of episodes is reached. 

After accessing the reward for winning (or the cost of losing), it calls fuction qlearning in order to learn how to avoid the ghost and how to win the game. 