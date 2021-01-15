
import os
import sys

import numpy as np

from grid_world import negative_grid, standard_grid, windy_grid
from helper import print_policy, print_values

ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
ALPHA = .1
GAMMA = .9
ITERATIONS = 2000
SMALL_ENOUGH = 10e-4

PACKAGE_PARENT = '..'
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.normpath(os.path.join(file_dir, "..")))

def get_random_action(a, epsilon=.1):
  if np.random.random() > epsilon:
    return a
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)


def play_game(grid, policy, epsilon=.1):
  """Returns an episode"""
  starting_state = (2, 0)
  grid.set_state(starting_state)
  visited = [(starting_state, 0)]
  current_state = starting_state
  while not grid.game_over():
    action = policy[current_state]
    action = get_random_action(action, epsilon)
    reward = grid.move(action)
    current_state = grid.current_state()
    visited.append((current_state, reward))
  return visited


def evaluate_policy(grid, policy):
  #State value function
  V = {}
  for cs in grid.all_states():
    V[cs] = 0

  for i in range(ITERATIONS):
    epsilon = .1 if i < 900 else 0
    states_rewards = play_game(grid, policy, epsilon)
    for t in range(len(states_rewards)-1):
      Vt = states_rewards[t][0]
      Vt1 = states_rewards[t + 1][0]
      Rt1 = states_rewards[t + 1][1]
      td_delta = Rt1+GAMMA*V[Vt1]-V[Vt]
      V[Vt] = V[Vt]+ALPHA*td_delta
  print("Values:")
  print_values(V, grid)


if __name__ == "__main__":
  grid = standard_grid()
  print("Rewards:")
  print_values(grid.rewards, grid)
  policy = {
    (2, 0): "U",
    (1, 0): "U",
    (0, 0): "R",
    (0, 1): "R",
    (0, 2): "R",
    (1, 2): "R",
    (2, 1): "R",
    (2, 2): "R",
    (2, 3): "U"
  }
  evaluate_policy(grid, policy)