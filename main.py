import pd_world
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation

def play(world, agent_F, agent_M, policy, max_steps, SARSA=False):
    step = 0
    total_reward = 0
    reward_log = []  # list of reward after every step
    reward_per_episode = 0  # the reward tracked for each terminal state
    reward_per_episode_log = []  # log of the reward amount required to reach each terminal state
    steps_at_terminal_log = []  # the total number of steps at each terminal state
    terminal_y = []  # tracks the total rewards at terminal states for graphing purposes

    agent_F_turn = True  # toggle for which agent's turn it is

    while step < 500:
        print("Steps: ", step)
        step += 1
        reward = 0
        if agent_F_turn:
            current_state = world.female_current_state.copy()
            action = agent_F.PRANDOM(world.male_current_state[0:2])
            reward = world.take_action(action, agent_F.name)
            next_state = world.female_current_state.copy()
            agent_F.Q_Learning(current_state, reward, next_state, action)
            agent_F_turn = False
        #else male
        else:
            current_state = world.male_current_state.copy()
            action = agent_M.PRANDOM(world.female_current_state[0:2])
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

    while step < max_steps:
        print("Steps: ", step)
        step += 1
        reward = 0
        if agent_F_turn:
            current_state = world.female_current_state.copy()
            if policy == 1:
                action = agent_F.PRANDOM(world.male_current_state[0:2])
                graph_title = "PRANDOM"
            elif policy == 2:
                action = agent_F.PEXPLOIT(world.male_current_state[0:2])
                graph_title = "PEXPLOIT"
            elif policy == 3:
                #TODO
                action = agent_F.PEXPLOIT(world.male_current_state[0:2])
                graph_title = "PGREEDY"
            reward = world.take_action(action, agent_F.name)
            next_state = world.female_current_state.copy()
            agent_F.Q_Learning(current_state, reward, next_state, action)
            agent_F_turn = False

        else:
            current_state = world.male_current_state.copy()
            if policy == 1:
                action = agent_M.PRANDOM(world.female_current_state[0:2])
                graph_title = "PRANDOM"
            elif policy == 2:
                action = agent_M.PEXPLOIT(world.female_current_state[0:2])
                graph_title = "PEXPLOIT"
            elif policy == 3:
                # TODO
                action = agent_M.PEXPLOIT(world.female_current_state[0:2])
                graph_title = "PGREEDY"
            reward = world.take_action(action, agent_M.name)
            next_state = world.male_current_state.copy()
            agent_M.Q_Learning(current_state, reward, next_state, action)
            agent_F_turn = True

        total_reward += reward
        reward_per_episode += reward
        if world.check_terminal_state():
            steps_at_terminal_log.append(step - 1)
            terminal_y.append(total_reward)
            reward_per_episode_log.append(reward_per_episode)
            reward_per_episode = 0
        reward_log.append(total_reward)

    agent_F.print_q_table()
    agent_M.print_q_table()

    print("Total steps taken: ", len(reward_log))
    print('Number of terminal states the agent reached: ', world.num_terminal_states_reached)
    print('Total reward: ', total_reward)

    F_q_values = agent_F.get_heatmap_Q_values()
    M_q_values = agent_M.get_heatmap_Q_values()

    if SARSA:
        title = f'{graph_title} with SARSA=True, alpha={agent_F.alpha}, gamma={agent_F.gamma}'
    else:
        title = f'{graph_title} with SARSA=False, alpha={agent_F.alpha}, gamma={agent_F.gamma}'

    return reward_log, steps_at_terminal_log, terminal_y, reward_per_episode_log, F_q_values, M_q_values, title




world = pd_world.PDWorld()
female_agent = pd_world.Agent("F", world, alpha=0.3, gamma=0.5)
male_agent = pd_world.Agent("M", world, alpha=0.3, gamma=0.5)

reward_log, steps_at_terminal_log, terminal_y, reward_per_episode_log, F_q_values, M_q_values, title = play(world, female_agent, male_agent, policy=2, max_steps=8000)

run = ["1_c_PGREEDY","run_2"]

# Calculate number of steps between terminal states and store their indexes for graphing
steps_between_terminal_states = [steps_at_terminal_log[0]]
for step in range(1, len(steps_at_terminal_log)):
    steps = steps_at_terminal_log[step] - steps_at_terminal_log[step-1]
    steps_between_terminal_states.append(steps)
terminal_states_indexes = []
for i in range(len(steps_between_terminal_states)):
    terminal_states_indexes.append(i+1)



#Fig1: Shows graph of reward over experiment with red pips for reached terminal state
plt.plot(reward_log)
plt.xlabel('Step Count')
plt.ylabel('Total Reward')
plt.title(title)
plt.scatter(steps_at_terminal_log, terminal_y, marker=',', color='r')
plt.savefig(run[0] + " " + run[1] + " total_reward_log")

#Fig2 :
fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(6,6))
fig.suptitle(title)
fig.text(0.5, 0.04,'Terminal state', ha='center')
ax1.set_ylabel('Steps to reach a terminal state')
ax2.set_ylabel('Reward per terminal state reached')
ax1.plot(terminal_states_indexes, steps_between_terminal_states)
ax2.plot(terminal_states_indexes, reward_per_episode_log)
ax1.set_ylim(0)
plt.locator_params(axis="both", integer=True)
plt.savefig(run[0] + " " + run[1] + " steps_per_terminal_space")

# Functions for heatmap
def triangulation_for_triheatmap(M, N):
    xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))  # vertices of the little squares
    xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))  # centers of the little squares
    x = np.concatenate([xv.ravel(), xc.ravel()])
    y = np.concatenate([yv.ravel(), yc.ravel()])
    cstart = (M + 1) * (N + 1)  # indices of the centers

    trianglesN = [(i + j * (M + 1), i + 1 + j * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesE = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesS = [(i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesW = [(i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]

    return [Triangulation(x, y, triangles) for triangles in [trianglesN, trianglesE, trianglesS, trianglesW]]


def create_Q_table_heatmap(heatmap_title, filename, q_values, triangul):
    norms = [plt.Normalize(-0.5, 1) for _ in range(4)]
    fig, ax = plt.subplots()

    plt.title(heatmap_title)
    imgs = [ax.tripcolor(t, val.ravel(), cmap='RdYlGn', vmin=-1.2, vmax=1, ec='white')
            for t, val in zip(triangul, q_values)]

    for val, dir in zip(q_values, [(-1, 0), (0, 1), (1, 0), (0, -1)]):
        for i in range(M):
            for j in range(N):
                v = val[j][i]
                ax.text(i + 0.3 * dir[1], j + 0.3 * dir[0], f'{v:.2f}', color='k' if 0.2 < v < 0.8 else 'w',
                        ha='center', va='center', fontsize=9)

    cbar = fig.colorbar(imgs[0], ax=ax)
    ax.set_xticks(range(M))
    ax.set_yticks(range(N))
    ax.invert_yaxis()
    ax.margins(x=0, y=0)
    ax.set_aspect('equal', 'box')  # square cells
    plt.tight_layout()
    plt.savefig(filename)
    return

M,N = 5,5
triangulation = triangulation_for_triheatmap(M, N)

create_Q_table_heatmap(title + "\nFemale Has Block Heatmap", run[0] + " " + run[1] + " female_has_block_heatmap ", F_q_values[0], triangulation)
create_Q_table_heatmap(title + "\nFemale No Block Heatmap", run[0] + " " + run[1] + " female_no_block_heatmap ", F_q_values[1], triangulation)

create_Q_table_heatmap(title + "\nMale Has Block Heatmap", run[0] + " " + run[1] + " male_has_block_heatmap ", M_q_values[0], triangulation)
create_Q_table_heatmap(title + "\nMale No Block Heatmap", run[0] + " " + run[1] + " male_no_block_heatmap ", M_q_values[1], triangulation)


