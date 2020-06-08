
import numpy as np

L_CHR = u"\u2190"
U_CHR = u"\u2191"
R_CHR = u"\u2192"
D_CHR = u"\u2193"
BLOCK_CHR = u"\u25A0"
HERE_CHR = u"\u0040"
TERM_CHR = u"\u002A"

class Gridworld:
  def __init__(self, rows, cols, rewards, walls, start_position=None):
    """Gridworld constructor.

    Args:
      rows: int, rows count.
      cols: int, columns count.
      rewards: dict, rewards of being in states.
      walls: list[tuples], location of walls
      start_position: tuple, starting position.
    """
    self.map = np.zeros((rows, cols))
    if start_position[0] < rows and start_position[1] < cols:
      self.position = start_position
    elif (0, 0) not in walls:
        self.position = (0, 0)
    else:
      raise Exception("Could not create the world of shape {} with the start position {}".format(
        (rows, cols), start_position
      ))
    if isinstance(start_position, int):
      current_state = start_position
    self.walls = walls
    # TODO: Add more complext positioning if needed
    self.actions, self.rewards = self.setup_map(rewards, walls)

  def setup_map(self, rewards, walls, terminal_success_val=1, terminal_failure_val=-1):
    #TODO: Set rewards
    terminal_states = []
    for k in rewards.keys():
      if rewards[k] in [terminal_success_val, terminal_failure_val]:
        terminal_states.append(k)
    actions = self._generate_actions(walls, terminal_states)
    self.rewards = rewards
    return actions, rewards

  def _generate_actions(self, walls=None, terminal_states=None):
    #TODO: If the state is a terminal state, it will not have actions.
    #TODO: Test for validity of walls.
    rows = self.map.shape[0]
    cols = self.map.shape[1]
    if walls == None:
      walls = []
    if terminal_states is None:
      terminal_states = []
    actions = {}
    movements = ["U", "D", "L", "R"]
    for i in range(rows):
      for j in range(cols):
        if (i, j) in terminal_states or (i, j) in walls:
          continue
        current_actions = movements[:]
        # Checking default walls.
        if i == 0:
          current_actions.remove("U")
        elif i == rows-1:
          current_actions.remove("D")
        if j == 0:
          current_actions.remove("L")
        elif j == cols-1:
          current_actions.remove("R")
        # Checking man-made walls
        if (i+1, j) in walls:
          try:
            current_actions.remove("D")
          except Exception as e:
            print("Wrong wall placement ignored!")
        if (i-1, j) in walls:
          try:
            current_actions.remove("U")
          except Exception as e:
            print("Wrong wall placement ignored!")
        if (i, j-1) in walls:
          try:
            current_actions.remove("L")
          except Exception as e:
            print("Wrong wall placement ignored!")
        if (i, j+1) in walls:
          try:
            current_actions.remove("R")
          except Exception as e:
            print("Wrong wall placement ignored!")
        actions[(i, j)] = set(current_actions)
    return actions

  def current_state(self):
    return self.position

  def move(self, direction):
    # TODO: Add no possible movements in the terminal states of failure or triumph!
    possible_actions = self.actions[self.position]
    current_position = list(self.position)
    if direction not in possible_actions:
      print("Invalid movement.")
    else:
      if direction == "R":
        current_position[1] += 1
      elif direction == "L":
        current_position[1] -= 1
      elif direction == "U":
        current_position[0] -= 1
      elif direction == "D":
        current_position[0] += 1
      else:
        print("Invalid movement.")
      self.position = tuple(current_position)
    # TODO: check if the state is a final state
    try:
      reward = self.rewards[self.position]
    except KeyError:
      reward = 0
    return reward

  def undo_move(self):
    #TODO: Write it if needed.
    pass

  def set_state(self, god_mod_position):
    if god_mod_position[0] < self.map.shape[0] and\
        god_mod_position[1] < self.map.shape[1]:
      self.position = god_mod_position
    else:
      print("God mode position is not valid!")

  def get_next_state(self, action):
    new_position = (0, 0)
    if action in self.actions[self.position]:
      if action == "R":
        i = self.position[1] + 1
      elif action == "L":
        i = self.position[1] - 1
      elif action == "U":
        j = self.position[0] - 1
      elif action == "D":
        j = self.position[0] + 1
      new_position = (i, j)
      return new_position
    else:
      return self.position

  def game_over(self):
    #TODO: Change this terminal accordingly
    if self.position == "terminal":
      return True
    else:
      return False


def printgrid(grid):
  printpolicy(grid, grid.actions)


def printpolicy(grid, policy):
  rows = grid.map.shape[0]
  cols = grid.map.shape[1]
  block_str = "|{:^3}{:^3}{:^3}|"
  wall_block = "|{}|".format(BLOCK_CHR*9)
  base_div = " --------- "
  for i in range(rows):
    print(base_div*cols)
    up = ["" if "U" not in policy.get((i, cj), []) else U_CHR for cj in range(cols)]
    left = ["" if "L" not in policy.get((i, cj), []) else L_CHR for cj in range(cols)]
    right = ["" if "R" not in policy.get((i, cj), []) else R_CHR for cj in range(cols)]
    down = ["" if "D" not in policy.get((i, cj), []) else D_CHR for cj in range(cols)]
    empty = ["" if (i, cj) != grid.position else HERE_CHR for cj in range(cols)]
    for j in range(cols):
      if (i, j) in grid.walls:
        print(wall_block, end="")
      else:
        print(block_str.format(empty[j], up[j], empty[j]), end="")
    print()
    for j in range(cols):
      if (i, j) in grid.walls:
        print(wall_block, end="")
      else:
        print(block_str.format(left[j], grid.rewards.get((i, j), 0), right[j]), end="")
    print()
    for j in range(cols):
      if (i, j) in grid.walls:
        print(wall_block, end="")
      else:
        print(block_str.format(empty[j], down[j], empty[j]), end="")
    print()
  print(base_div*cols)



def standard_grid():
  rewards = {(0, 3): 1, (1, 3): -1}
  g = Gridworld(3, 4, rewards, [(1, 1)], (2, 0))
  return g

def test_grid():
  g = standard_grid()
  printpolicy(g, g.actions)
  policy = {(2,0): "U", (1, 0): "U", (0, 0): "R", (0, 1): "R",
            (0, 2): "R", (1, 2): "U", (2, 1): "R", (2, 2): "U", (2, 3): "L"}
  printpolicy(g, policy)