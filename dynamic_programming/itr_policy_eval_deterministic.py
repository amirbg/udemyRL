
import numpy as np

import world

def run(grid, policy, gamma=.9, stop_threshold=.05):
  max_change = 1000
  states = grid.all_states()
  action_space = policy
  rewards = {}
  transition_probability = {}
  iii = 0
  while max_change > stop_threshold:
    iii += 1
    if iii % 1000 == 0:
      import pdb;
      pdb.set_trace();
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
          reward = grid.map[next_s[0]][next_s[1]]
          transition_probability[(current_s, a, next_s)] = 1
          rewards[(current_s, a, next_s)] = reward*gamma
          new_val += reward
        new_val = gamma*(new_val/len(possible_actions))
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
  gamma = .9
  run(world.standard_grid(), policy)