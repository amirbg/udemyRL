""" Policy evaluation according to Sutton's book means to calculate the
value of each state by using a policy which in this case, we are using
the monte-carlo method to perform the calculation.
"""
import numpy as np

from grid_world import negative_grid, standard_grid
from iterative_policy_evaluation import print_policy, print_values

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
ITERATIONS = 100

def generate_episode(grid, policy):
  # Use exploring starts to ensure all state-action pairs are selected.
  # Get states with possible actions (non-terminal and non-blocked).
  start_state = list(grid.actions)[np.random.randint(len(grid.actions))]
  grid.set_state(start_state)
  state = grid.current_state()
  visited_states_rewards = [(state, 0)]
  # Generate one episode
  while not grid.game_over():
    action = policy[state]
    reward = grid.move(action)
    state = grid.current_state()
    visited_states_rewards.append((state, reward))

  # Calculate returns for states.
  # Initialize return.
  states_returns = []
  # Final state has a return of 0 and is excluded from the list.
  G = 0
  last_reward = visited_states_rewards[-1][1]
  for s, r in visited_states_rewards[:-1][::-1]:
    G = last_reward + GAMMA * G
    states_returns.append((s, G))
    last_reward = r

  return states_returns


if __name__ == "__main__":
  grid = standard_grid()
  print("Rewards:")
  print_values(grid.rewards, grid)
  policy = {(0, 0): 'R', (0, 1): 'R', (0, 2): 'R', (1, 0): 'U',
            (1, 2): 'R', (2, 0): 'U', (2, 1): 'R', (2, 2): 'R', (2, 3): 'U'}
  # Initialize values of all states.
  V = {}
  returns = {}
  all_states = grid.all_states()
  for cs in all_states:
    if cs in grid.actions:
      returns[cs] = []
    else:
      V[cs] = 0
  for i in range(ITERATIONS):
    states_returns = generate_episode(grid, policy)
    seen_states = set()
    for s, G in states_returns:
      if s not in seen_states:
        returns[s].append(G)
        V[s] = np.mean(returns[s])
        seen_states.add(s)
      # for first visit

  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)

