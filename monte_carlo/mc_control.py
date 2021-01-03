import numpy as np

from grid_world import negative_grid, standard_grid, windy_grid
from iterative_policy_evaluation import print_policy, print_values


GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
ITERATIONS = 20000
RANDOM_ACTION_PROB = .5
MIN_DELTA = .005


def random_action(a, possible_actions=None, exclude_optimum=True, random_ac_prob=.5):
  val = np.random.random()
  if val < random_ac_prob:
    if possible_actions is None:
      possible_actions = ALL_POSSIBLE_ACTIONS
    curr_possible_actions = list(possible_actions)
    if exclude_optimum:
      curr_possible_actions.remove(a)
    return np.random.choice(curr_possible_actions)
  else:
    return a


def evaluate_policy(grid, policy, random_ac_prob=.5, only_choose_possible=False):
  # Use exploring starts to ensure all state-action pairs are selected.
  # Get states with possible actions (non-terminal and non-blocked).
  start_state = list(grid.actions)[np.random.randint(len(grid.actions))]
  grid.set_state(start_state)
  state = grid.current_state()
  state_action_rewards = []
  hit_wall = False
  # Generate one episode
  while not grid.game_over():
    action = policy[state]
    possible_actions = grid.actions[state] if only_choose_possible else None
    final_action = random_action(action, possible_actions, True, random_ac_prob)
    # This is more consistent with the convention used by Sutton's book. reward is not
    # the signal related to the next state, it's when you leave a state.
    last_state = grid.current_state()
    reward = grid.move(final_action)
    state = grid.current_state()
    if state != last_state:
      state_action_rewards.append(((last_state, final_action), reward))
    else:
      # If we face a wall, the action is useless.
      state_action_rewards.append(((last_state, final_action), -100))
      hit_wall = True
      break
  if grid.game_over():
    state_action_rewards.append(((grid.current_state(), None), 0))


  # There are two possible last_states: terminal and wall, both have G of 0
  # Calculate returns for states.
  # Initialize return.
  state_action_returns = []
  # Final state has a return of 0 and is excluded from the list.
  G = 0
  for s_a, r in state_action_rewards[:-1][::-1]:
    G = r + GAMMA * G
    state_action_returns.append((s_a, G))

  if hit_wall:
    return state_action_returns[::-1] + state_action_rewards[-1:]

  return state_action_returns[::-1]


if __name__ == "__main__":
  grid = negative_grid()
  print("Rewards:")
  print_values(grid.rewards, grid)
  optimal_policy = {(0, 0): 'R', (0, 1): 'R', (0, 2): 'R', (1, 0): 'U',
            (1, 2): 'U', (2, 0): 'U', (2, 1): 'L', (2, 2): 'U', (2, 3): 'L'}
  policy = {}
  for s in grid.actions.keys():
    policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
  # Initialize values of all states.
  biggest_change = -10000
  Q = {}
  returns = {}

  for i in range(ITERATIONS):
    try:
      random_action_prob = max(0, 0.5 - int(i/1000)*.1)
      # random_action_prob = RANDOM_ACTION_PROB
      state_action_returns = evaluate_policy(grid, policy, random_action_prob)
      sar = state_action_returns
      seen_state_actions = set()
      for s_a, G in state_action_returns:
        if s_a not in returns:
          returns[s_a] = []
        # for first visit
        if s_a not in seen_state_actions:
          returns[s_a].append(G)
          Q[s_a] = np.median(returns[s_a])
          seen_state_actions.add(s_a)

      if not(len(Q.keys())):
        continue

      all_states = np.array(list(Q.keys()))[:,0]
      for s in all_states:
        returns_per_action = []
        for k, v in Q.items():
          if k[0] == s:
            returns_per_action.append((v, k[1]))
        best_idx = np.argmax(np.array(returns_per_action)[:,0].astype(float))
        best_return = returns_per_action[best_idx][0]
        best_action = returns_per_action[best_idx][1]
        # If best return is better than the return according to current policy
        curr_policy_return = Q.get((s, policy[s]),-20)
        delta = best_return - curr_policy_return
        if delta > 0:
          # import pdb; pdb.set_trace();
          print("changing policy for {}, {} -> {}".format(s, policy[s], best_action))
          policy[s] = best_action
          if delta > biggest_change:
            biggest_change = delta

      if (i+1) % 5000 == 0:
        print(i, biggest_change)
        if biggest_change < MIN_DELTA:
          print("\n\nNo more improvement at iteration {}".format(i))
          break
        biggest_change = 0
    except Exception as e:
      import pdb; pdb.set_trace();
      x=2

  Vs = {}
  for cs in all_states:
    cp = policy[cs]
  for s_a, Qs in Q.items():
    if s_a[0] not in Vs:
      Vs[s_a[0]] = []
    Vs[s_a[0]].append(Qs)
  V = {s: max(Vs[s]) for s in Vs}
  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)
