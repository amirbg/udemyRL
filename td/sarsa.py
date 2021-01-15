
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


from grid_world import negative_grid, standard_grid, windy_grid
from helper import print_policy, print_values

ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
ALPHA = .1
GAMMA = .9
ITERATIONS = 20000
SMALL_ENOUGH = 10e-5

PACKAGE_PARENT = '..'
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.normpath(os.path.join(file_dir, "..")))

def get_random_action(a, epsilon=.1):
  if np.random.random() > epsilon:
    return a
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)

def epsilon_greedy(grid, Q, gamma, epsilon):
  state = grid.current_state()
  max_q = -10000
  greedy_action = None
  for ac in ALL_POSSIBLE_ACTIONS:
    ns = grid.get_next_state(state, ac)
    action_max = grid.rewards[ns]+gamma*max(Q[ns].values())
    if action_max > max_q:
      greedy_action = ac
  if np.random.random() > epsilon:
    return greedy_action
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)




def evaluate_policy(grid):
  #State value function
  Q = {}
  t = 1.
  epsilon=.5
  deltas = []
  for cs in grid.all_states():
    Q[cs] = {}
    for a in ALL_POSSIBLE_ACTIONS:
      Q[cs][a] = 0

  upcounts = {}
  Q_upcount = {}
  for cs in grid.all_states():
    Q_upcount[cs] = {}
    for a in ALL_POSSIBLE_ACTIONS:
      Q_upcount[cs][a] = 1
  for i in range(ITERATIONS):
    if i%2000 == 0:
      print(i)
    biggest_change = -100000
    if i%100 == 0:
      t += 10e-3

    start_state = (2, 0)
    grid.set_state(start_state)
    current_state = grid.current_state()
    next_action = max(Q[current_state], key=Q[current_state].get)
    next_action = next_action if epsilon / t < np.random.random() else \
      np.random.choice(grid.actions[current_state])
    while not grid.game_over():
      current_action = next_action
      current_state = grid.current_state()
      alpha = ALPHA / (Q_upcount[current_state][current_action])
      next_state = grid.get_next_state(current_state, current_action)
      reward = grid.rewards[next_state]
      Q_upcount[current_state][current_action] += .005
      next_action = max(Q[next_state], key=Q[next_state].get)
      randd = np.random.random()
      next_action = next_action if epsilon / t < randd else \
        np.random.choice(ALL_POSSIBLE_ACTIONS)
      delta = reward + GAMMA*Q[next_state][next_action] - Q[current_state][current_action]
      Q[current_state][current_action] = Q[current_state][current_action]+alpha*delta
      if alpha*delta > biggest_change:
        biggest_change = delta*alpha

      grid.set_state(next_state)
    deltas.append(biggest_change)
    if i > 1000 and biggest_change < SMALL_ENOUGH:
        break

  plt.plot(deltas); plt.show()
  print("Values:")
  V = {}
  for s,v in Q.items():
    V[s] = max(vv[1] for vv in v.items())
  print_values(V, grid)


if __name__ == "__main__":
  grid = negative_grid()
  print("Rewards:")
  print_values(grid.rewards, grid)
  evaluate_policy(grid)