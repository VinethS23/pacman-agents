# REQUIREMENTS

## v1 Requirements — Active

### Neural Network Agent
- **REQ-NN-1:** User can run a PyTorch-based Q-learning agent with `-p NNQAgent` flag
  - Category: Agent Implementation
  - **Acceptance:** Command `python pacman.py -p NNQAgent -x 100 -n 110 -l smallGrid -q` runs without errors

- **REQ-NN-2:** Neural Network Agent extracts fixed-length feature vector from GameState
  - Category: State Representation
  - **Acceptance:** Feature vector is consistent length across all states; includes pacman position, ghost positions, food/capsule counts

- **REQ-NN-3:** Neural Network Agent achieves >70% win rate on smallGrid after 2000 training episodes
  - Category: Performance
  - **Acceptance:** Running with default hyperparameters achieves ≥70 wins out of 100 test episodes

- **REQ-NN-4:** Neural Network Agent accepts hyperparameters via `-a` flag (alpha, epsilon, gamma, numTraining)
  - Category: Interface Compatibility
  - **Acceptance:** `-a alpha=0.001,epsilon=0.1,gamma=0.99,numTraining=2000` works and affects learning

- **REQ-NN-5:** Backpropagation training updates network weights based on Q-learning Bellman equation
  - Category: Algorithm Correctness
  - **Acceptance:** Agent learns; performance improves over episodes; loss decreases

### Comparison & Analysis
- **REQ-CMP-1:** User can run both QLearnAgent and NNQAgent on same layout for comparison
  - Category: Flexibility
  - **Acceptance:** Both agents run and produce scores; can be logged side-by-side

- **REQ-CMP-2:** NNQAgent and QLearnAgent share reward function and episode structure
  - Category: Fairness
  - **Acceptance:** Same hyperparameters mapped to both; same layouts; same episode counts

## v1 Requirements — Already Shipped

- ✓ Tabular Q-Learning Agent (QLearnAgent) — 89% win rate on smallGrid
- ✓ GameStateFeatures state abstraction — hashing and equality working
- ✓ Framework integration — command-line flags (`-p`, `-x`, `-n`, `-l`, `-a`, `-q`)
- ✓ Reward function (computeReward static method)
- ✓ Exploration strategies (epsilon-greedy, count-based bonus)

## v2 Requirements — Nice-to-Have

- [ ] Experience replay buffer for Neural Network Agent
- [ ] Target network (separate frozen network for stability)
- [ ] Convolutional neural network for larger grids
- [ ] Policy gradient agent (REINFORCE)
- [ ] Actor-Critic agent
- [ ] Training visualization (score plots over episodes)
- [ ] Model checkpointing (save/load trained weights)
- [ ] Imitation learning from Q-learning policy

## Out of Scope

- GPU acceleration
- Distributed training
- Web UI
- Production deployment
- Modifications to UC Berkeley framework code
