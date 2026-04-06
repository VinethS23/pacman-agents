# Pacman Agents — Experimental ML

## Overview
Experimental playground for implementing RL agents in Pacman using different ML techniques. Learn by building: tabular Q-learning, neural networks, and beyond.

## Problem
Understanding how different ML approaches solve the same problem (Pacman control) requires hands-on implementation. This project explores the trade-offs: tabular methods are interpretable but don't scale; neural networks scale but are harder to debug.

## Tech Stack
- **Language:** Python 3.x
- **Framework:** UC Berkeley Pacman AI (read-only infrastructure)
- **Q-Learning:** Standard library only (pure Python)
- **Neural Networks:** PyTorch (for NN agents)
- **Testing:** Command-line interface `-p AgentName` flag

## Architecture
```
pacman.py (game engine)
    ↓
Agent subclass (your implementations)
    ├── getAction(state) → Directions
    └── final(state) → None

Current agents in mlLearningAgents.py:
    ├── QLearnAgent (tabular Q-learning)
    └── NNQAgent (neural network Q-approximation — to be built)

Feature extraction:
    GameState → GameStateFeatures (hashable wrapper)
    or
    GameState → torch.Tensor (fixed-length feature vector)
```

## Features
### Must-Have
- **Tabular Q-Learning Agent** — ✓ Complete. 89% win rate on smallGrid.
- **Neural Network Agent** — In progress. Feedforward network with PyTorch. User implements with guidance.
- **Feature vector design** — Extracts state information for both agents.
- **Hyperparameter tuning interface** — `-a` flag supports both agents.

### Nice-to-Have
- Policy gradient agent (REINFORCE, A2C)
- Convolutional network for larger grids
- Experience replay and target networks (DQN improvements)
- Training visualization (score plots over episodes)
- Model checkpointing and loading

## Data Model
**GameStateFeatures** (for Q-learning):
- Pacman position (x, y)
- Ghost positions (tuple of (x, y) per ghost)
- Capsule locations (tuple)
- Food count (int)
- Hashable for dictionary key storage

**Feature vector** (for NN agent):
- Fixed-length numpy array / torch.Tensor
- Pacman position (2 values)
- Ghost positions relative to pacman (2 per ghost × N ghosts)
- Ghost scared timers (1 per ghost × N ghosts)
- Food remaining (1 value)
- Capsules remaining (1 value)
- Total: ~10–15 features depending on layout

## API / Integration Contracts
**Agent interface** (from `pacman_utils.game.Agent`):
```python
getAction(state: GameState) -> Directions
final(state: GameState) -> None
```

**Reward function** (static method):
```python
computeReward(startState: GameState, endState: GameState) -> float
# Returns: score delta + win/loss bonus
```

**Running agents**:
```bash
python pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid -q
python pacman.py -p NNQAgent -x 2000 -n 2010 -l smallGrid -q
```

## Auth
N/A — single-player agent implementations.

## Deployment
Run locally in terminal. Optional: save trained Q-tables or NN weights for later evaluation.

## Constraints
- **Framework is read-only** — only modify agent code in `mlLearningAgents.py`
- **No external grading** — focus on learning, not optimization for test cases
- **UC Berkeley license** — educational use only; retain attribution

## Development Phases

### Phase 1 — Neural Network Agent Implementation
Goal: Build a PyTorch-based Q-value approximator and compare with tabular Q-learning.

- [ ] **Feature extraction design**
  - `get_features(state: GameState) -> torch.Tensor` — extract fixed-length feature vector
  - Include: pacman position, ghost positions (relative), food count, capsule count, scared timers
  - Decide: simple features vs. rich (distance to food, wall indicators)

- [ ] **PyTorch network class**
  - `QNetwork(nn.Module)` — 3-layer feedforward
  - Input: feature vector size (TBD after feature design)
  - Hidden: 64 units, ReLU activation
  - Output: 5 Q-values (one per action: N, S, E, W, STOP)
  - `forward(x)` method returns Q-values

- [ ] **NNQAgent class**
  - Extends `Agent`
  - Stores: network, optimizer (Adam), last state/action/reward
  - `getAction(state)` — epsilon-greedy action selection with network
  - `learn(state, action, reward, next_state)` — backprop Q-learning update
  - `final(state)` — terminal state update, episode tracking
  - Hyperparameters: alpha (learning rate), epsilon, gamma, numTraining

- [ ] **Test and tune**
  - Run on smallGrid: 2000 training, 10 test episodes
  - Measure: win rate, average score, convergence speed
  - Adjust hyperparameters (hidden size, learning rate, exploration decay)
  - Compare results with QLearnAgent baseline

Verification: NN agent runs without errors and achieves >70% win rate on smallGrid after 2000 training episodes.

### Phase 2 — Analysis & Comparison
Direction: Understand how NN agent differs from Q-learning — speed, stability, scalability.

Likely includes:
- Side-by-side training curves (Q-learning vs NN)
- Hyperparameter sensitivity analysis
- Performance on medium-sized grids
- Failure mode analysis (where does NN struggle vs Q-learning?)

### Phase 3 — Advanced Neural Network Variants
Direction: Extend beyond simple feedforward — experiment with architectural improvements.

Likely includes:
- Dueling networks (value + advantage streams)
- Convolutional networks for larger grids
- Experience replay buffer
- Target network (for stability)

### Phase 4 — Alternative ML Approaches
Direction: Explore non-neural RL methods for contrast.

Likely includes:
- Policy gradient (REINFORCE)
- Actor-critic methods
- Imitation learning (behavioral cloning from Q-learning)

## Open Questions
None — scope is clear. Claude will ask for design decisions (feature vector scope, network size) before implementation begins.
