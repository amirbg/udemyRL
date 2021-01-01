
import numpy as np

import world

def generate_random_policy(grid):
  states = grid.all_states()
  policy = {}
  for i in range(states.shape[0]):
    for j in range(states.shape[1]):
      policy[(i,j)] = np.random.choice(list(grid.actions.get((i,j), "Invalid")))
  return policy


def evaluate_policy(grid, policy, gamma, stop_threshold):
  iii = 0
  max_change = 1000
  states = grid.all_states()
  while max_change > stop_threshold:
    iii += 1
    max_change = 0
    for i in range(states.shape[0]):
      for j in range(states.shape[1]):
        current_s = (i, j)
        next_s = grid.get_next_state_absolute(current_s, policy.get(current_s, "invalid"))
        # If current state is a terminal state or a wall
        if next_s is None:
          continue
        ns_reward = grid.rewards.get(next_s, 0)
        new_val = ns_reward + grid.map[next_s]*gamma
        diff = np.abs(grid.map[current_s[0]][current_s[1]] - new_val)
        grid.set_value(current_s, new_val)
        if diff > max_change:
          max_change = diff



def get_greedy_policy(grid, old_policy, gamma):
  states = grid.all_states()
  action_space = grid.actions
  for i in range(states.shape[0]):
    for j in range(states.shape[1]):
      current_s = (i, j)
      c_value = grid.map[current_s]
      actions = grid.actions.get((i, j), [])
      for c_a in actions:
        next_s = grid.get_next_state_absolute(current_s, c_a)
        ns_reward = grid.rewards.get(next_s, 0)
        new_val = ns_reward + grid.map[next_s] * gamma
        if new_val > c_value:
          old_policy[current_s] = c_a
          c_value = new_val
      if actions == []:
        old_policy[current_s] = "Invalid"
  return policy


def run_policy_iteration(grid, gamma=.9, stop_threshold=.001):
  is_converged = False
  policy = generate_random_policy(grid)
  world.printpolicy(grid, policy)
  while not is_converged:
    old_map = grid.map.copy()
    evaluate_policy(grid, policy, gamma, stop_threshold)
    new_map = grid.map
    max_change_val = np.max(np.abs(np.array(new_map) - np.array(old_map)))
    if max_change_val < stop_threshold:
      break
    policy = get_greedy_policy(grid, policy, gamma)
    world.printpolicy(grid, policy)
  world.printpolicy(grid, policy)


def get_greedy_probabilistic_policy(grid, env_proba, old_policy, gamma):
  states = grid.all_states()
  action_space = grid.actions
  for i in range(states.shape[0]):
    for j in range(states.shape[1]):
      current_s = (i, j)
      c_value = grid.map[current_s]
      actions = grid.actions.get((i, j), [])
      for c_a in actions:
        if (current_s, c_a) in env_proba:
          possible_next_states = env_proba[(current_s, c_a)]
          new_val = 0
          for next_s in possible_next_states:
            state_probability = possible_next_states[next_s]
            ns_reward = grid.rewards.get(next_s, 0)
            new_val += ((ns_reward + grid.map[next_s] * gamma) * state_probability)
        else:
          next_s = grid.get_next_state_absolute(current_s, c_a)
          ns_reward = grid.rewards.get(next_s, 0)
          new_val = ns_reward + grid.map[next_s] * gamma
        if new_val > c_value:
          old_policy[current_s] = c_a
          c_value = new_val
      if actions == []:
        old_policy[current_s] = "Invalid"
  return policy


def evaluate_policy_probabilistic(grid, env_proba, policy, gamma, stop_threshold):
  max_change = 1000
  states = grid.all_states()
  while max_change > stop_threshold:
    max_change = 0
    for i in range(states.shape[0]):
      for j in range(states.shape[1]):
        current_s = (i, j)
        action = policy.get(current_s, "invalid")
        if (current_s, action) in env_proba:
          possible_next_states = env_proba[(current_s, action)]
          new_val = 0
          for next_s in possible_next_states:
            state_probability = possible_next_states[next_s]
            ns_reward = grid.rewards.get(next_s, 0)
            new_val += ((ns_reward + grid.map[next_s] * gamma)*state_probability)
        else:
          next_s = grid.get_next_state_absolute(current_s, action)
        # If current state is a terminal state or a wall
          if next_s is None:
            continue

          ns_reward = grid.rewards.get(next_s, 0)
          new_val = ns_reward + grid.map[next_s]*gamma
        diff = np.abs(grid.map[current_s[0]][current_s[1]] - new_val)
        grid.set_value(current_s, new_val)
        if np.abs(diff) > max_change:
          max_change = diff

def run_probabilistic_policy_iteration(grid, env_proba, gamma=.9, stop_threshold=.001, step_cost=0):
  is_converged = False
  print("\n\n")
  policy = generate_random_policy(grid)
  world.printpolicy(grid, policy)
  while not is_converged:
    old_map = grid.map.copy()
    evaluate_policy_probabilistic(grid, env_proba, policy, gamma, stop_threshold)
    world.printpolicy(grid, policy)
    new_map = grid.map
    max_change_val = np.max(np.abs(np.array(new_map) - np.array(old_map)))
    if max_change_val < stop_threshold:
      break
    policy = get_greedy_probabilistic_policy(grid, env_proba, policy, gamma)
    world.printpolicy(grid, policy)
  print("\nFinal map for step cost of ", step_cost)
  world.printpolicy(grid, policy)


if __name__ == "__main__":
  policy = {(2, 0): "U", (1, 0): "U", (0, 0): "R", (0, 1): "R",
            (0, 2): "R", (1, 2): "U", (2, 1): "R", (2, 2): "U", (2, 3): "L"}
  policy_proba = {(2, 0): {"U": .5, "R": .5} , (1, 0): {"U": 1.}, (0, 0): {"R": 1.},
                  (0, 1): {"R":1.}, (0, 2): {"R":1.}, (1, 2): {"U": 1.}, (2, 1): {"R": 1.},
                  (2, 2): {"U": 1.}, (2, 3): {"L": 1.}}
  gamma = .9
  env_probs = {((2, 0), "U"): {(1, 0): 1.},
           ((1, 2), "U"): {(0, 2): .5, (1, 3): .5}}
  # run_policy_iteration(world.standard_grid())
  step_cost = 0
  step_cost = -.1
  run_probabilistic_policy_iteration(world.standard_grid(step_cost), env_probs, step_cost)
  step_cost = -.2
  run_probabilistic_policy_iteration(world.standard_grid(step_cost), env_probs, step_cost)
  step_cost = -2
  run_probabilistic_policy_iteration(world.standard_grid(step_cost), env_probs, step_cost=step_cost)