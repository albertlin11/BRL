python_code

csv file:
1.dataset_new.csv: experiment value
2.TCAD.csv: TCAD value
3.RL_input_parameter.csv: Input parameters used in RL.py

py file:
1.DQN.py: This file include network structure and algorithm of bayesian RL
2.env.py: This file define reward, encoder(to find which state agent choose)
3.find_Rs.py: Find different Rs value in whole path and before best timestep
4.prior_prob.py: Define prior function
5.RL.py: Combine all files above to conduct Bayesian RL

dictionary file:
1.dataFromRL_old_reward: Result of normal DQN
2.dataFromRL_old_reward_bayes: Result of bayesian DQN

How to use?
Open RL.py and click run. It will generate two dictionary files to save results