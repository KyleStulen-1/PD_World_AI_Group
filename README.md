Requirements:  
- matplotlib  
- numpy  

Instructions: 
- run pd_world with command "python main.py"
- alter running parameters on lines 120-126
- policies 
    - 1: PRANDOM
    - 2: PGREEDY
    - 3: PEXPLOIT
        - set SARSA to true if you want to use SARSA instead of Q_Learning algorithm for PEXPLOIT
        - otherwise leave SARSA equaling False
- learning rate alpha and discount factor gamma are set for each agent, for this project both agents always had the same learning rate and discount factor
    