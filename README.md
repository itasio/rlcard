# Creating Reinforcement Learning agents on rlcard enviroment		

# New limit holdem game
A limit holdem mode with shorter deck 4x(A, 10, J, Q, K), 1 hand card, 2 public cards		
Purpose: Shorter state space, test simplier algorithms		
Threshold Agent and Threshold Agent2:		
Rule based models betting only on high cards and combinations
new_limit_holdem_human: play againt any suitable agent
		
# Algorithms Implemented
- **Q-learning** variation algorithm implemented: ql_agent(QLAgent)
  - run_ql: train ql  
  - tune_ql: tune hyperparameters of ql
- **Policy iteration** algorithm implemented:  pi_agent
- **Value iteration** algorithm implemented: value_iteration_agent  
- **SARSA variation** algorithm implemented: sarsa_agent(SARSAAgent)

