import random
import numpy as np

class PDWorld:
    #pick, pick, drop, drop, drop, drop
    starting_state = np.array([[3,1,10],[2,4,10],[0,0,0],[0,4,0],[2,2,0],[4,4,0]])
    terminal_state = np.array([[3,1,0],[2,4,0],[0,0,5],[0,4,5],[2,2,5],[4,4,5]])

    def __init__(self):
        self.height = 5
        self.width = 5
        self.num_terminal_states_reached = 0

        self.reward_grid = np.empty((self.height,self.width), dtype=object)
        #cost of -1 for moving up,down,right,left
        self.reward_grid[:] = -1

        #pickup reward
        for i in range(0,2):
            self.reward_grid[self.starting_state[i][0],self.starting_state[i][1]] = "P"
        #drop reward
        for i in range(2,6):
            self.reward_grid[self.starting_state[i][0],self.starting_state[i][1]] = "D"

        #starting location always same
        self.female_start_state = (0,2,False)
        self.female_current_state = list(self.female_start_state)

        self.male_start_state = (4,2,False)
        self.male_current_state = list(self.male_start_state)

        #store starting state for restarting of board after reaching terminal state
        self.pd_starting_state = PDWorld.starting_state.copy()
        #updated with current values of blocks in pickup and drop locations
        self.pd_current_state = self.starting_state
        #terminal state storage
        self.pd_terminal_state = PDWorld.terminal_state.copy()

        self.actions = ("N","S","E","W","P","D")

    def print_current_state(self):
        matrix = np.zeros((self.height,self.width), dtype=object)
        matrix[self.female_current_state[0],self.female_current_state[1]] = "F"
        #matrix[self.male_current_start[0], self.male_current_state[1]] = "M"
        print(np.concatenate((self.reward_grid, matrix), axis=1).astype(object), ': Current state F: ', self.female_current_state, sep='') #, 'Current state M: ', self.male_current_state
        print(self.pd_current_state, ': Current pd state', sep='')

    def take_action(self,action,agent):
        reward = 0
        if action == "N":
            if agent == "F":
                self.female_current_state[0:2] = [self.female_current_state[0] - 1, self.female_current_state[1]]
                reward = -1
            #else male
            else:
                self.male_current_state[0:2] = [self.male_current_state[0] - 1, self.male_current_state[1]]
                reward = -1

        if action == "S":
            if agent == "F":
                self.female_current_state[0:2] = [self.female_current_state[0] + 1, self.female_current_state[1]]
                reward = -1
            #else male
            else:
                self.male_current_state[0:2] = [self.male_current_state[0] + 1, self.male_current_state[1]]
                reward = -1

        if action == "W":
            if agent == "F":
                self.female_current_state[0:2] = [self.female_current_state[0], self.female_current_state[1] - 1]
                reward = -1
            #else male
            else:
                self.male_current_state[0:2] = [self.male_current_state[0], self.male_current_state[1] - 1]
                reward = -1

        if action == "E":
            if agent == "F":
                self.female_current_state[0:2] = [self.female_current_state[0], self.female_current_state[1] + 1]
                reward = -1
            #else male
            else:
                self.male_current_state[0:2] = [self.male_current_state[0], self.male_current_state[1] + 1]
                reward = -1

        if action == "P":
            if agent == "F":
                #get whether agent is at first or 2nd pickup location
                row = np.where((self.female_current_state[0:2] == self.pd_current_state[:, 0:2]).all(axis=1))[0].tolist()
                #reduce blocks at pickup location by 1
                self.pd_current_state[row[0], 2] -= 1
                #female is now holding a block
                self.female_current_state[2] = True
                reward = 13
            #else male
            else:
                row = np.where((self.male_current_state[0:2] == self.pd_current_state[:, 0:2]).all(axis=1))[0].tolist()
                self.pd_current_state[row[0], 2] -= 1
                self.male_current_state[2] = True
                reward = 13


        if action == "D":
            if agent == "F":
                #get which pickup location agent is at (3rd,4th,5th,6th row in self.current_matrix)
                row = np.where((self.female_current_state[0:2] == self.pd_current_state[:, 0:2]).all(axis=1))[0].tolist()
                #increase blocks at drop location by 1
                self.pd_current_state[row[0], 2] += 1
                #female is no longer holding a block
                self.female_current_state[2] = False
                reward = 13
            #else male
            else:
                row = np.where((self.male_current_state[0:2] == self.pd_current_state[:, 0:2]).all(axis=1))[0].tolist()
                self.pd_current_state[row[0], 2] += 1
                self.male_current_state[2] = False
                reward = 13


        return reward

    def check_walls(self, agent):
        actions = list(self.actions)
        if agent == "F":
            if self.female_current_state[0] == 0:
                actions.remove('N')
                if self.female_current_state[1] == 0:
                    actions.remove('W')
                elif self.female_current_state[1] == self.width - 1:
                    actions.remove('E')

            if self.female_current_state[0] == self.height - 1:
                actions.remove('S')
                if self.female_current_state[1] == 0:
                    actions.remove('W')
                elif self.female_current_state[1] == self.width - 1:
                    actions.remove('E')

            if self.female_current_state[1] == 0 and 'W' in actions:
                actions.remove('W')
                if self.female_current_state[0] == 0:
                    actions.remove('N')
                elif self.female_current_state[0] == self.height - 1:
                    actions.remove('S')

            if self.female_current_state[1] == self.width - 1 and 'E' in actions:
                actions.remove('E')
                if self.female_current_state[0] == 0:
                    actions.remove('N')
                elif self.female_current_state[0] == self.height - 1:
                    actions.remove('S')
        #else male
        else:
            if self.male_current_state[0] == 0:
                actions.remove('N')
                if self.male_current_state[1] == 0:
                    actions.remove('W')
                elif self.male_current_state[1] == self.width - 1:
                    actions.remove('E')

            if self.male_current_state[0] == self.height - 1:
                actions.remove('S')
                if self.male_current_state[1] == 0:
                    actions.remove('W')
                elif self.male_current_state[1] == self.width - 1:
                    actions.remove('E')

            if self.male_current_state[1] == 0 and 'W' in actions:
                actions.remove('W')
                if self.male_current_state[0] == 0:
                    actions.remove('N')
                elif self.male_current_state[0] == self.height - 1:
                    actions.remove('S')

            if self.male_current_state[1] == self.width - 1 and 'E' in actions:
                actions.remove('E')
                if self.male_current_state[0] == 0:
                    actions.remove('N')
                elif self.male_current_state[0] == self.height - 1:
                    actions.remove('S')

        return actions

    def remove_colliding_actions(self, agent, actions, agent2loc):
        #print("actions: ", actions)
        if agent == "F":
            if 'N' in actions:
                if (self.female_current_state[0]-1) == agent2loc[0] and (self.female_current_state[1]) == agent2loc[1]:
                    actions.remove('N')
                    print("removed N collision")
            if 'S' in actions:
                if (self.female_current_state[0]+1) == agent2loc[0] and (self.female_current_state[1]) == agent2loc[1]:
                    actions.remove('S')
                    print("removed S collision")
            if 'W' in actions:
                if (self.female_current_state[0]) == agent2loc[0] and (self.female_current_state[1]-1) == agent2loc[1]:
                    actions.remove('W')
                    print("removed W collision")
            if 'E' in actions:
                if (self.female_current_state[0]) == agent2loc[0] and (self.female_current_state[1]+1) == agent2loc[1]:
                    actions.remove('E')
                    print("removed E collision")
        else:
            if 'N' in actions:
                if (self.male_current_state[0]-1) == agent2loc[0] and (self.male_current_state[1]) == agent2loc[1]:
                    actions.remove('N')
                    print("removed N collision")
            if 'S' in actions:
                if (self.male_current_state[0]+1) == agent2loc[0] and (self.male_current_state[1]) == agent2loc[1]:
                    actions.remove('S')
                    print("removed S collision")
            if 'W' in actions:
                if (self.male_current_state[0]) == agent2loc[0] and (self.male_current_state[1]-1) == agent2loc[1]:
                    actions.remove('W')
                    print("removed W collision")
            if 'E' in actions:
                if (self.male_current_state[0]) == agent2loc[0] and (self.male_current_state[1]+1) == agent2loc[1]:
                    actions.remove('E')
                    print("removed E collision")

        print(actions)
        return actions

    def get_valid_actions(self, agent, agent2loc):
        valid_actions = self.check_walls(agent).copy()
        valid_actions = self.remove_colliding_actions(agent, valid_actions, agent2loc)
        valid_actions_noPD = [i for i in valid_actions if i not in ('P', 'D')]  #list of actions without pickup or drop
        if agent == "F":
            if self.female_current_state[0:2] in self.pd_current_state[:, 0:2].tolist():
                row = np.where((self.pd_current_state[:, 0:2] == self.female_current_state[0:2]).all(axis=1))[0].tolist()
                if row[0] in [0, 1]:
                    if self.female_current_state[2] == False and int(self.pd_current_state[row[0], 2]) != 0:
                        return 'P'
                    else:
                        return valid_actions_noPD
                else:
                    if self.female_current_state[2] == True and int(self.pd_current_state[row[0], 2]) != 5:
                        return 'D'
                    else:
                        return valid_actions_noPD
            else:
                return valid_actions_noPD
        #else male
        else:
            if self.male_current_state[0:2] in self.pd_current_state[:, 0:2].tolist():
                row = np.where((self.pd_current_state[:, 0:2] == self.male_current_state[0:2]).all(axis=1))[0].tolist()
                if row[0] in [0, 1]:
                    if self.male_current_state[2] == False and int(self.pd_current_state[row[0], 2]) != 0:
                        return 'P'
                    else:
                        return valid_actions_noPD
                else:
                    if self.male_current_state[2] == True and int(self.pd_current_state[row[0], 2]) != 5:
                        return 'D'
                    else:
                        return valid_actions_noPD
            else:
                return valid_actions_noPD

    def check_terminal_state(self):
        if (self.pd_current_state == self.pd_terminal_state).all():
            self.num_terminal_states_reached += 1
            print(self.num_terminal_states_reached, " Terminal states reached. Reset to initial world state...........................")
            self.__init__()
            return True
        else:
            return False

class Agent:
    def __init__(self, name, world, alpha, gamma):
        self.name = name #F or M
        self.world = world
        self.alpha = alpha
        self.gamma = gamma

        self.q_table_has_block = dict()
        for x in range(5):
            for y in range(5):
                self.q_table_has_block[x,y] = {'N': 0, 'S': 0, 'W': 0, 'E': 0, 'P': 0, 'D': 0}

        self.q_table_no_block = dict()
        for x in range(5):  # Initialize agent without block q-table with values 0.
            for y in range(5):
                self.q_table_no_block[x, y] = {'N': 0, 'S': 0, 'W': 0, 'E': 0, 'P': 0, 'D': 0}

    def PRANDOM(self, agent2loc):
        #print("Valid actions: ", self.world.get_valid_actions(self.name))

        #if action is pick or drop
        if len(self.world.get_valid_actions(self.name, agent2loc)) == 1:
            action = self.world.get_valid_actions(self.name, agent2loc)
        #else choose from available north/south/east/west options
        else:
            action_options = self.world.get_valid_actions(self.name, agent2loc)
            action = action_options[random.randint(0, len(action_options) - 1)]
        return action

    def print_q_table(self):
        print("\n Agent, ", self.name, " Q Table Holding Block")
        for x in range(self.world.height):
            for y in range(self.world.width):
                print(f'({x},{y}): ', end='')
                for k, v in self.q_table_has_block[x, y].items():
                    v = round(v, 4)
                    print(f'{k}: {v:.4f}','\t', end='', sep='')
                print()

        print("\n Agent, ", self.name, " Q Table No Block")
        for x in range(self.world.height):
            for y in range(self.world.width):
                print(f'({x},{y}): ', end='')
                for k, v in self.q_table_no_block[x, y].items():
                    v = round(v, 4)
                    print(f'{k}: {v:.4f}', '\t', end='', sep='')
                print()

    def Q_Learning(self, current_state, reward, next_state, action):
        if current_state[2] == True:  #holding a block
            next_state_qvalues = self.q_table_has_block[next_state[0],next_state[1]].copy()
            row = np.where((next_state[0:2] == self.world.pd_current_state[:, 0:2]).all(axis=1))[0].tolist()
            if row:
                #if drop location is full delete those Q values from being considered
                if self.world.pd_current_state[row[0], 2] == 5:
                    del next_state_qvalues['D']
                    del next_state_qvalues['P']
                    highest_q_value = max(next_state_qvalues.values())
                else:
                    highest_q_value = max(next_state_qvalues.values())
            else:
                highest_q_value = max(next_state_qvalues.values())

            #print('Reward: ', reward)
            print("action: ", action)
            print("current_state: ", self.name, ": ", current_state)
            print(self.q_table_no_block)
            print("current_q_value: ", self.q_table_no_block[current_state[0], current_state[1]][action])
            current_q_value = self.q_table_has_block[current_state[0], current_state[1]][action]
            self.q_table_has_block[current_state[0], current_state[1]][action] = (
                1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * highest_q_value)

            print("Updated Q Values: ", self.q_table_has_block[current_state[0], current_state[1]], "at", current_state)

        else:  #not holding a block
            next_state_qvalues = self.q_table_no_block[next_state[0], next_state[1]].copy()
            row = np.where((next_state[0:2] == self.world.pd_current_state[:, 0:2]).all(axis=1))[0].tolist()
            print("row finder: ", np.where((next_state[0:2] == self.world.pd_current_state[:, 0:2]).all(axis=1))[0].tolist())
            if row:
                # if pickup location is full delete those Q values from being considered
                if self.world.pd_current_state[row[0], 2] == 0:
                    del next_state_qvalues['D']
                    del next_state_qvalues['P']
                    highest_q_value = max(next_state_qvalues.values())
                else:
                    highest_q_value = max(next_state_qvalues.values())
            else:
                highest_q_value = max(next_state_qvalues.values())
            #print("Reward: ", reward)
            print("action: ", action)
            print("current_state: ", self.name, ": ", current_state)
            print(self.q_table_no_block)
            print("current_q_value: ", self.q_table_no_block[current_state[0], current_state[1]][action])
            current_q_value = self.q_table_no_block[current_state[0], current_state[1]][action]
            self.q_table_no_block[current_state[0], current_state[1]][action] = (
                1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * highest_q_value)
            print("Updates Q Values: ", self.q_table_no_block[current_state[0], current_state[1]], "at ",self.name,": ", current_state)

    def get_heatmap_Q_values(self):
        #holding block
        N_block = np.arange(25,dtype=float).reshape(5,5)
        E_block = np.arange(25,dtype=float).reshape(5,5)
        S_block = np.arange(25,dtype=float).reshape(5,5)
        W_block = np.arange(25,dtype=float).reshape(5,5)

        # no block
        N = np.arange(25,dtype=float).reshape(5,5)
        E = np.arange(25,dtype=float).reshape(5,5)
        S = np.arange(25,dtype=float).reshape(5,5)
        W = np.arange(25,dtype=float).reshape(5,5)

        #fill with values
        for i in range(5):
            for j in range(5):
                N_block[i][j] = self.q_table_has_block[i,j]['N']
                E_block[i][j] = self.q_table_has_block[i,j]['E']
                S_block[i][j] = self.q_table_has_block[i,j]['S']
                W_block[i][j] = self.q_table_has_block[i,j]['W']
                N[i][j] = self.q_table_no_block[i,j]['N']
                E[i][j] = self.q_table_no_block[i,j]['E']
                S[i][j] = self.q_table_no_block[i,j]['S']
                W[i][j] = self.q_table_no_block[i,j]['W']

        return ([N_block, E_block, S_block, W_block], [N, E, S, W])

