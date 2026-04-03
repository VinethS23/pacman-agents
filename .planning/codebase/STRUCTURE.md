# Project Structure

```
pacman-cw2-2/
│
├── mlLearningAgents.py       # Q-learning agent implementation (primary submission file)
├── pacman.py                 # Game engine, entry point, GameState, game rules
├── sampleAgents.py           # Baseline agents: RandomAgent, RandomishAgent
├── __init__.py               # Package marker (empty)
│
└── pacman_utils/             # UC Berkeley Pacman framework (read-only infrastructure)
    ├── __init__.py
    ├── game.py               # Agent base class, Directions, Grid, Actions, Game loop, GameStateData
    ├── layout.py             # Layout parser — reads .lay files into Layout objects
    ├── ghostAgents.py        # GhostAgent base class, RandomGhost, DirectionalGhost
    ├── pacmanAgents.py       # GreedyAgent and other built-in Pacman agents
    ├── keyboardAgents.py     # Human keyboard-controlled agent
    ├── util.py               # Counter, flipCoin, manhattanDistance, priority queues, TimeoutFunction
    ├── graphicsDisplay.py    # Tkinter GUI display (PacmanGraphics)
    ├── graphicsUtils.py      # Low-level Tkinter drawing utilities
    ├── textDisplay.py        # ASCII display for -t mode and NullGraphics for -q mode
    ├── projectParams.py      # Project-level constants (e.g. STUDENT_CODE_DEFAULT)
    ├── VERSION               # Framework version string
    └── layouts/              # Map definition files
        ├── smallGrid.lay         # Tiny grid — fastest convergence, used for testing
        ├── mediumGrid.lay        # Medium-sized grid
        ├── smallClassic.lay      # Small classic Pacman layout
        ├── mediumClassic.lay     # Standard medium classic layout (default)
        ├── originalClassic.lay   # Full original Pacman layout
        ├── capsuleClassic.lay    # Layout with power capsules
        ├── contestClassic.lay
        ├── minimaxClassic.lay
        ├── openClassic.lay
        ├── testClassic.lay
        ├── trappedClassic.lay
        └── trickyClassic.lay
```

## File Roles

### Files you interact with

| File | Purpose |
|---|---|
| `mlLearningAgents.py` | The Q-learning implementation. Contains `GameStateFeatures` (state abstraction) and `QLearnAgent` (the learning agent). This is the only file modified for the coursework. |
| `pacman.py` | Entry point. Run this to launch the game. Also defines `GameState` — the full game state object passed to agents. |
| `sampleAgents.py` | Simple baseline agents (`RandomAgent`, `RandomishAgent`) useful for comparison. |

### Framework files (do not modify)

| File | Purpose |
|---|---|
| `pacman_utils/game.py` | Defines the `Agent` base class that all agents extend. Also contains `Directions` (North/South/East/West/Stop), `Game` (the main game loop), `GameStateData`, `Grid`, `Actions`, and spatial data structures. |
| `pacman_utils/layout.py` | Reads `.lay` map files and builds `Layout` objects encoding walls, food, capsules, and agent starting positions. |
| `pacman_utils/ghostAgents.py` | Ghost implementations. `RandomGhost` chooses uniformly at random. `DirectionalGhost` chases Pacman with probability 0.8, flees when scared. |
| `pacman_utils/util.py` | Shared utilities: `Counter` (defaultdict-like), `flipCoin(p)`, `manhattanDistance`, `nearestPoint`, heap-based priority queues, and `TimeoutFunction`. |
| `pacman_utils/graphicsDisplay.py` | Tkinter GUI. Initialized when `-q` and `-t` flags are both absent. |
| `pacman_utils/textDisplay.py` | ASCII display for `-t` (text) mode and `NullGraphics` for `-q` (quiet/training) mode. |
| `pacman_utils/pacmanAgents.py` | `GreedyAgent` and other framework-provided Pacman agents. |

### Layout file format (`.lay`)

Maps are ASCII text files where each character encodes a cell type:

| Character | Meaning |
|---|---|
| `%` | Wall |
| `.` | Food pellet |
| `o` | Power capsule |
| `P` | Pacman start position |
| `G` | Ghost start position |
| `1`–`4` | Numbered ghost start positions |
| ` ` | Empty corridor |

## Key Classes at a Glance

| Class | File | Role |
|---|---|---|
| `GameStateFeatures` | `mlLearningAgents.py` | State abstraction for Q-table keys |
| `QLearnAgent` | `mlLearningAgents.py` | Tabular Q-learning agent |
| `GameState` | `pacman.py` | Full authoritative game state passed to agents |
| `Agent` | `pacman_utils/game.py` | Base class — defines `getAction(state)` interface |
| `Directions` | `pacman_utils/game.py` | Enum-style constants: North, South, East, West, Stop |
| `Game` | `pacman_utils/game.py` | Main game loop — solicits actions, applies them, checks win/lose |
| `ClassicGameRules` | `pacman.py` | Controls game start/end, timeout logic |
| `Layout` | `pacman_utils/layout.py` | Static board information (walls, food, start positions) |
| `GhostAgent` | `pacman_utils/ghostAgents.py` | Base class for all ghost implementations |
| `Grid` | `pacman_utils/game.py` | 2D boolean array used for food and wall maps |
