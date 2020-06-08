
from . import world

def run(grid, policy, gamma=.9, stop_threshold=.01):
  max_change = 1000
  states = grid.all_states()
  while max_change > stop_threshold:


if __name__ == "__main__":
  policy = {(2, 0): "U", (1, 0): "U", (0, 0): "R", (0, 1): "R",
            (0, 2): "R", (1, 2): "U", (2, 1): "R", (2, 2): "U", (2, 3): "L"}
  gamma = .9
  run()