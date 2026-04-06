# PROJECT — Pacman Agents

## What This Is
Experimental playground for RL agents in Pacman. Currently implements tabular Q-learning (achieved 89% win rate); expanding to neural network-based agents and beyond. Personal learning project — not graded.

## Problem
Understanding ML approaches requires hands-on implementation. This project explores: tabular Q-learning (interpretable, doesn't scale), neural networks (scalable, harder to debug), policy gradients, and more.

## Core Value
Learn different RL/ML techniques by implementing agents that solve the same problem — Pacman control.

## Tech Stack
- **Language:** Python 3.x
- **Framework:** UC Berkeley Pacman AI (read-only)
- **Q-Learning:** Standard library (numpy not required)
- **Neural Networks:** PyTorch (backprop, optimization)
- **Testing:** CLI with `-p AgentName` flag

## Architecture
```
Game engine (pacman.py)
    ↓
Agent subclass implementations
    ├── getAction(state) → action
    └── final(state) → learn from terminal state

Current: QLearnAgent (tabular Q-learning)
In progress: NNQAgent (PyTorch feedforward network)
Future: policy gradient, actor-critic, etc.
```

## Key Decisions
| Decision | Rationale |
|---|---|
| Extend `Agent` base class | Pluggable interface; run with `-p AgentName` flag |
| State abstraction (features) | Q-learning uses hashable wrapper; NN uses fixed-length tensor |
| PyTorch for NN | Modern, clean syntax; automatic differentiation (backprop) |
| Feedforward first, conv later | Start simple; scale up to larger grids after validating approach |
| Compare, don't replace | Keep Q-learning intact; side-by-side comparison is the goal |

## Requirements — Validated
- ✓ Q-Learning Agent (QLearnAgent) — 89% win rate on smallGrid, 2000 training episodes
- ✓ State abstraction (GameStateFeatures) — hashable for Q-table keys
- ✓ Framework integration — command-line flags working (`-p`, `-x`, `-n`, `-l`, `-a`)

## Requirements — Active
- [ ] Neural Network Agent (NNQAgent) — PyTorch feedforward, achieves >70% win rate after 2000 episodes on smallGrid
- [ ] Feature extraction — fixed-length vector from game state for NN input
- [ ] Training loop — backprop Q-learning updates, epsilon-greedy action selection
- [ ] Hyperparameter interface — `-a` flag works for both agents
- [ ] Performance comparison — side-by-side analysis with Q-learning

## Requirements — Out of Scope
- GPU training (CPU is fine)
- Model persistence (saving/loading weights)
- GUI improvements
- Framework modifications (UC Berkeley code is read-only)
- Production deployment

## Context
- **Solo project** — personal learning
- **Ungraded** — freedom to experiment
- **Educational goal** — understand ML trade-offs through implementation
- **UC Berkeley Pacman** — provided framework; retain attribution
- **PyTorch** — added dependency for NN experiments

## Constraints
1. Do not modify Q-Learning Agent without explicit permission
2. Framework code is read-only (focus on agents only)
3. Maintain UC Berkeley attribution and licensing terms
