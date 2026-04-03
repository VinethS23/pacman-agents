# Architecture

This project implements a tabular Q-learning agent that learns to play Pacman through reinforcement learning. It is built on the UC Berkeley Pacman AI framework, adapted for KCL by Simon Parsons and updated to Python 3 by Dylan Cope and Lin Li.

## High-Level Overview

```
CLI (pacman.py)
     │
     ▼
ClassicGameRules          ← controls win/loss/timeout logic
     │
     ▼
Game (game loop)          ← solicits actions from all agents in round-robin order
     │              │
     ▼              ▼
QLearnAgent        GhostAgent(s)
(Pacman, index=0)  (indices 1..N)
     │
     ▼
GameStateFeatures  ← compressed state representation used as Q-table key
```

## Component Descriptions

### Entry Point: `pacman.py`

Parses command-line arguments (`readCommand`) and launches `runGames`. Hosts the `GameState` class, which is the authoritative game state object passed to every agent on each turn. Also contains the game rules sub-classes:

- `ClassicGameRules` — manages game lifecycle (win, lose, timeout).
- `PacmanRules` — enforces Pacman movement and consumption of food/capsules.
- `GhostRules` — enforces ghost movement, scared timers, and collision/death logic.

Key constants defined here:
- `SCARED_TIME = 40` — number of moves a ghost stays scared after a capsule is eaten.
- `COLLISION_TOLERANCE = 0.7` — Manhattan distance threshold for ghost-Pacman collision.
- `TIME_PENALTY = 1` — score deducted each time step.

### Core Framework: `pacman_utils/`

| File | Responsibility |
|---|---|
| `game.py` | `Agent` base class, `Directions`, `Configuration`, `AgentState`, `Grid`, `Actions`, `GameStateData`, `Game` (main control loop) |
| `layout.py` | Parses `.lay` map files into `Layout` objects (walls, food, capsules, agent start positions) |
| `ghostAgents.py` | `GhostAgent` base class; `RandomGhost` (uniform random); `DirectionalGhost` (probabilistic chase/flee) |
| `util.py` | `Counter`, `flipCoin`, `manhattanDistance`, `nearestPoint`, priority queues, `TimeoutFunction` |
| `graphicsDisplay.py` | Tkinter-based GUI display |
| `textDisplay.py` | ASCII text display and `NullGraphics` (for quiet/training mode) |
| `keyboardAgents.py` | Human keyboard-controlled agent |
| `pacmanAgents.py` | Greedy agent and other built-in Pacman agents |
| `projectParams.py` | Project-level configuration constants |
| `layouts/` | Map definition files (`.lay`) |

### Q-Learning Agent: `mlLearningAgents.py`

Contains two classes:

#### `GameStateFeatures`

A lightweight wrapper around `GameState` that defines the state representation used as Q-table keys. The hash and equality are based on:

- Pacman position `(x, y)`
- Ghost positions (tuple of positions)
- Capsule locations (tuple)
- Food count (integer — not exact food grid, to limit state space)

This abstraction deliberately loses some information (exact food grid positions) to keep the Q-table tractable.

#### `QLearnAgent`

Extends `Agent`. Implements the full Q-learning loop.

**Data structures:**
- `q_values: dict[(GameStateFeatures, Directions), float]` — the Q-table.
- `action_counts: dict[(GameStateFeatures, Directions), int]` — visit counts per state-action pair.
- `last_state: GameStateFeatures` — state from the previous time step.
- `last_action: Directions` — action taken at the previous time step.

**Learning flow (per time step):**

```
getAction(state)
  ├── wrap state → GameStateFeatures
  ├── if last_state exists:
  │     reward = computeReward(last_state.state, state)
  │     learn(last_state, last_action, reward, stateFeatures)
  ├── action selection (ε-greedy + exploration bonus)
  ├── updateCount(stateFeatures, chosen_action)
  └── store last_state, last_action
```

**End-of-episode flow:**

```
final(state)
  ├── perform final learn() update with terminal state
  ├── reset last_state, last_action to None
  ├── incrementEpisodesSoFar()
  └── if episodes == numTraining: set alpha=0, epsilon=0
```

## Data Flow

```
pacman.py main()
  └── runGames(layout, pacman, ghosts, display, numGames, ...)
        └── for each game:
              ClassicGameRules.newGame() → Game object
              Game.run()
                └── loop until gameOver:
                      for each agent (Pacman, Ghost1, Ghost2, ...):
                        observation = state.deepCopy()
                        action = agent.getAction(observation)
                        state = state.generateSuccessor(agentIndex, action)
                        display.update(state.data)
                        rules.process(state, game)   ← checks win/lose
              └── agent.final(state)  ← called on all agents with final state
```

## Q-Learning Algorithm

**Update rule (Bellman equation):**

```
Q(s, a) ← Q(s, a) + α · [R(s, a, s') + γ · max_{a'} Q(s', a') − Q(s, a)]
```

**Reward function:**

```
R = score(s') − score(s)
  + 500   if s' is a win
  − 500   if s' is a loss
```

**Action selection:**

```
exploration_value(s, a) =
    1000.0           if count(s, a) < maxAttempts   (optimistic exploration bonus)
    Q(s, a)          otherwise

With probability ε: choose random legal action
Otherwise:          choose argmax_a exploration_value(s, a), breaking ties randomly
```

**Training termination:**

When `episodesSoFar == numTraining`, both `alpha` and `epsilon` are set to 0. The agent then acts greedily on the frozen Q-table with no further updates.

## Scoring System (from `pacman.py` / `game.py`)

| Event | Score change |
|---|---|
| Each time step | −1 |
| Eating a food pellet | +10 |
| Eating all food (win) | +500 |
| Ghost collision (normal) | −500, game over |
| Eating a capsule | 0 (enables scared ghosts) |
| Eating a scared ghost | +200, ghost respawns |

## Dependencies

- Python 3.x (standard library only for the agent)
- `six` — Python 2/3 compatibility shim used by the framework
- `tkinter` — optional, only required for GUI display
