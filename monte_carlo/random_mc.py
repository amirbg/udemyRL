
import numpy as np

from grid_world import negative_grid, standard_grid
from helper import print_policy, print_values


GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
ITERATIONS = 5000
RANDOM_ACTION_PROB = .5


def random_action(a, possible_actions=None, exclude_optimum=True):
  val = np.random.random()
  if val > RANDOM_ACTION_PROB:
    if possible_actions is None:
      possible_actions = ALL_POSSIBLE_ACTIONS
    curr_possible_actions = list(possible_actions)
    if exclude_optimum:
      curr_possible_actions.remove(a)
    return np.random.choice(curr_possible_actions)
  else:
    return a


def generate_episode(grid, policy, only_choose_possible=False):
  # Use exploring starts to ensure all state-action pairs are selected.
  # Get states with possible actions (non-terminal and non-blocked).
  start_state = list(grid.actions)[np.random.randint(len(grid.actions))]
  grid.set_state(start_state)
  state = grid.current_state()
  visited_states_rewards = [(state, 0)]
  # Generate one episode
  while not grid.game_over():
    action = policy[state]
    possible_actions = grid.actions[state] if only_choose_possible else None
    final_action = random_action(action, possible_actions, True)
    reward = grid.move(final_action)
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

  return states_returns[::-1]


if __name__ == "__main__":
  grid = standard_grid()
  print("Rewards:")
  print_values(grid.rewards, grid)
  policy = {(0, 0): 'R', (0, 1): 'R', (0, 2): 'R', (1, 0): 'U',
            (1, 2): 'U', (2, 0): 'U', (2, 1): 'L', (2, 2): 'U', (2, 3): 'L'}
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

