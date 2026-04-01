# Q-Learning Pacman Agent

A tabular Q-learning agent that teaches itself to play Pacman from scratch through trial and error. Built on the [UC Berkeley Pacman AI framework](http://ai.berkeley.edu/reinforcement.html).

## What it does

The agent starts with no knowledge of the game and learns purely through reinforcement — receiving rewards for eating food and penalties for dying. After ~2000 training episodes on a small grid it learns a competent policy without any hand-crafted rules.

Key ML concepts implemented from scratch:

- **Tabular Q-learning** with the Bellman update rule
- **Epsilon-greedy exploration** — random action with probability ε, greedy otherwise
- **Count-based exploration bonus** — unexplored state-action pairs get a high optimistic value (1000.0) until visited `maxAttempts` times, preventing the agent from getting stuck in local optima
- **Reward shaping** — terminal win/loss bonuses (+500 / −500) on top of score delta to guide early learning
- **State abstraction** — raw game state compressed to (Pacman position, ghost positions, capsule locations, food count) for tractable Q-table size

## Algorithm

**Q-learning update (Bellman equation):**

```
Q(s, a) ← Q(s, a) + α [R(s, a, s') + γ · max_a' Q(s', a') − Q(s, a)]
```

**Action selection:**

```
if count(s, a) < maxAttempts:
    value(s, a) = 1000.0   # exploration bonus for under-visited pairs
else:
    value(s, a) = Q(s, a)  # exploit learned values

action = argmax_a value(s, a)   # with ε-greedy random override
```

## Results

Training on `smallGrid` with default hyperparameters (2000 training episodes, 10 test episodes):

- The agent converges to a reliable win rate on `smallGrid` within ~1000 episodes
- Post-training (ε=0, α=0) the agent acts greedily on its learned Q-table

## Hyperparameters

| Parameter | Default | Role |
|---|---|---|
| `alpha` | 0.2 | Learning rate — how much new experience overwrites old |
| `epsilon` | 0.05 | Exploration rate — probability of taking a random action |
| `gamma` | 0.8 | Discount factor — how much future rewards are valued |
| `maxAttempts` | 30 | Visit threshold before exploration bonus is removed |
| `numTraining` | 10 | Training episodes (set via `-x` flag at runtime) |

After training completes, both `alpha` and `epsilon` are set to 0 so the agent exploits its learned policy with no further updates.

## Setup

Python 3.x, no external dependencies.

```bash
git clone https://github.com/YOUR_USERNAME/pacman-qlearning.git
cd pacman-qlearning
```

## Running

### Fast (no GUI) — recommended for experiments

```bash
python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid -q
```

### With GUI

```bash
python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid
```

### Flag reference

| Flag | Meaning |
|---|---|
| `-p QLearnAgent` | Use the Q-learning agent |
| `-x 2000` | 2000 training episodes (output suppressed) |
| `-n 2010` | 2010 total episodes → last 10 are test games |
| `-l smallGrid` | Layout (see below) |
| `-q` | No graphics, minimal output |
| `-a alpha=0.3,...` | Override any hyperparameter |

### Tuning hyperparameters

```bash
python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid -q \
  -a alpha=0.3,epsilon=0.1,gamma=0.9,maxAttempts=20
```

## Available Layouts

| Layout | Description |
|---|---|
| `smallGrid` | Tiny grid, fastest convergence |
| `mediumGrid` | Medium grid |
| `smallClassic` | Small classic Pacman map |
| `mediumClassic` | Standard map |
| `originalClassic` | Full original layout |
| `capsuleClassic` | Includes power capsules |

## Project Structure

```
├── mlLearningAgents.py   # Q-learning implementation (main file)
├── pacman.py             # Game engine and entry point
├── sampleAgents.py       # Baseline agents (greedy, random)
└── pacman_utils/
    ├── game.py           # Core game logic, Agent base class
    ├── util.py           # Helpers (Counter, flipCoin, etc.)
    ├── layout.py         # Map loader
    ├── ghostAgents.py    # Ghost AI
    ├── layouts/          # Map files (.lay)
    └── ...
```

## Attribution

The Pacman AI framework was developed at UC Berkeley by John DeNero, Dan Klein, Brad Miller, Nick Hay, and Pieter Abbeel. Adapted for KCL by Simon Parsons; updated to Python 3 by Dylan Cope and Lin Li.

Licensing: free to use or extend for educational purposes — do not distribute solutions, retain attribution, link to [http://ai.berkeley.edu](http://ai.berkeley.edu).
