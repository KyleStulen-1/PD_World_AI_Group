import pd_world


def play(world, agent_F, agent_M, policy, max_steps, SARSA=False):
    step = 0
    total_reward = 0
    reward_log = []  # list of reward after every step
    reward_per_episode = 0  # the reward tracked for each terminal state
    reward_per_episode_log = []  # log of the reward amount required to reach each terminal state
    steps_at_terminal_log = []  # the total number of steps at each terminal state
    terminal_y = []  # tracks the total rewards at terminal states for graphic purposes

    agent_F_turn = True  # toggle for which agent's turn it is

    while step < 7000:
        print("Steps: ", step)
        step += 1
        reward = 0
        if agent_F_turn:
            current_state = world.female_current_state.copy()
            #print("F current state: ", current_state)
            #print("agent_M loc: ", world.male_current_state[0:2])
            action = agent_F.PRANDOM(world.male_current_state[0:2])
            #print("Chosen action: ", action)
            reward = world.take_action(action, agent_F.name)
            next_state = world.female_current_state.copy()
            agent_F.Q_Learning(current_state, reward, next_state, action)
            agent_F_turn = False
        #else male
        else:
            current_state = world.male_current_state.copy()
            # print("F current state: ", current_state)
            #print("agent_F loc: ", world.female_current_state[0:2])
            action = agent_M.PRANDOM(world.female_current_state[0:2])
            # print("Chosen action: ", action)
            reward = world.take_action(action, agent_M.name)
            next_state = world.male_current_state.copy()
            agent_M.Q_Learning(current_state, reward, next_state, action)
            agent_F_turn = True

        total_reward += reward
        reward_per_episode += reward
        if world.check_terminal_state():
            steps_at_terminal_log.append(step-1)
            terminal_y.append(total_reward)
            reward_per_episode_log.append(reward_per_episode)
            reward_per_episode = 0
        reward_log.append(total_reward)
        print("\n")

    print('Number of terminal states the agent reached: ', world.num_terminal_states_reached)
    print('Total reward: ', total_reward)



    agent_F.print_q_table()
    agent_M.print_q_table()

    print(reward_log)
    print(len(reward_log))

    q_values = agent_F.get_heatmap_Q_values()





world = pd_world.PDWorld()
female_agent = pd_world.Agent("F", world, alpha=0.3, gamma=0.5)
male_agent = pd_world.Agent("M", world, alpha=0.3, gamma=0.5)

play(world, female_agent, male_agent, policy=0, max_steps=7000)
