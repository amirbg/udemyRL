
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

      # randd = np.random.random()
      # next_action = next_action if epsilon / t < randd else \
      #   np.random.choice(ALL_POSSIBLE_ACTIONS)
      delta = reward + GAMMA*max(Q[next_state].values()) - Q[current_state][current_action]
      Q[current_state][current_action] = Q[current_state][current_action]+alpha*delta
      next_action = max(Q[next_state], key=Q[next_state].get)
      randval = np.random.random()
      next_action = next_action if epsilon / t < randval else \
        np.random.choice(ALL_POSSIBLE_ACTIONS)
      grid.set_state(next_state)
      if alpha*delta > biggest_change:
        biggest_change = delta*alpha
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