
import numpy as np

import world

def run_deterministic(grid, policy, gamma=.9, stop_threshold=.001):
  max_change = 1000
  states = grid.all_states()
  action_space = policy
  rewards = {}
  transition_probability = {}
  iii = 0
  while max_change > stop_threshold:
    iii += 1
    max_change = 0
    for i in range(states.shape[0]):
      for j in range(states.shape[1]):
        new_val = 0
        current_s = (i, j)
        possible_actions = action_space.get(current_s, [])
        if possible_actions == []:
          continue
        for a in possible_actions:
          next_s = grid.get_next_state_absolute(current_s, a)
          value = grid.map[next_s[0]][next_s[1]]
          reward = grid.rewards.get(next_s, 0)
          transition_probability[(current_s, a, next_s)] = 1
          rewards[(current_s, a, next_s)] = reward
          new_val = new_val + transition_probability[(current_s, a, next_s)]*(reward + gamma*value)
        diff = np.abs(grid.map[current_s[0]][current_s[1]] - new_val)
        grid.set_value(current_s, new_val)
        # grid.rewards[current_s] = new_val
        if diff > max_change:
          max_change = diff

  print("Number of iterations: {}".format(iii))
  world.printpolicy(grid, policy)


def run_windy(grid, policy, env_probs, gamma=.9, stop_threshold=.001):
  max_change = 1000
  states = grid.all_states()
  action_space = policy
  rewards = {}
  transition_probability = env_probs.copy()
  # Creating transition probalitiy for all states.
  for i in range(states.shape[0]):
    for j in range(states.shape[1]):
      actions = action_space.get((i, j), {})
      possible_combs = set([((i, j), act) for act in actions])
      if not len(possible_combs.intersection(set(transition_probability.keys()))):
        action = action_space.get((i, j), "Invalid")
        transition_probability[((i, j), action)] = {
          grid.get_next_state_absolute((i, j), action): 1.}

  iii = 0
  while max_change > stop_threshold:
    iii += 1
    max_change = 0
    for i in range(states.shape[0]):
      for j in range(states.shape[1]):
        new_val = 0
        current_s = (i, j)
        possible_actions = action_space.get(current_s, {})
        for act in possible_actions:
          next_states = transition_probability.get((current_s, act), {})
          for next_s in next_states:
            pro_coef = next_states[next_s]
            value = grid.map[next_s]
            # next state reward
            ns_reward = grid.rewards.get(next_s, 0)
            new_val = new_val + (ns_reward+value*gamma)*pro_coef
        diff = np.abs(grid.map[current_s[0]][current_s[1]] - new_val)
        grid.set_value(current_s, new_val)
        # grid.rewards[current_s] = new_val
        if diff > max_change:
          max_change = diff

  print("Number of iterations: {}".format(iii))
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
  run_deterministic(world.standard_grid(), policy)
  run_windy(world.standard_grid(), policy, env_probs)
  # run_windy_probabilistic(world.standard_grid(), policy_proba, env_probs)