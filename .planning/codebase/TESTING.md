# Testing Patterns

**Analysis Date:** 2026-04-03

## Test Framework

**Status:** No automated testing framework configured

**Finding:** This is an educational ML project with manual testing only. No test runner, no test files (*.test.py, *.spec.py), and no testing configuration (pytest.ini, setup.py, pyproject.toml) found in the repository.

**Testing Approach:** Manual execution with command-line arguments and visual/output verification

## Manual Testing

**Run Commands:**

```bash
# Fast (no GUI) — recommended for experiments
python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid -q

# With GUI
python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid

# Tuning hyperparameters
python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid -q \
  -a alpha=0.3,epsilon=0.1,gamma=0.9,maxAttempts=20
```

**Test Metrics (from README):**
```
| Metric | Value |
|---|---|
| Win rate | **89/100 (89%)** |
| Average score | **391.73** |
| Winning score range | 499 – 505 |
| Losing score range | −513 – −508 |
```

## Verification Strategy

**Manual Testing Points:**
1. Agent training convergence - observe learning curve over 2000 episodes
2. Episode output verification - "Game N just ended!" printed at end of each episode
3. Training completion - both alpha and epsilon set to 0 when training done
4. Test phase accuracy - final 10 episodes run with learned policy (ε=0, α=0)
5. Win/loss conditions - properly trigger reward bonuses (+500/-500)
6. State-action hash consistency - states hashable and usable as Q-table keys

**Visual Verification (with GUI):**
- Pacman moves toward food initially
- Ghosts avoided during learning
- Movement becomes more coherent over episodes (less random)
- Test episodes show minimal random exploration

**Non-GUI Verification (-q flag):**
- Episode completion messages printed
- Training indicator messages output
- No graphics overhead allows faster training

## Testing Patterns Observed

**Agent Testing:**
```python
# From sampleAgents.py - RandomAgent pattern
def getAction(self, state):
    # Get the actions we can try, and remove "STOP" if that is one of them.
    legal = state.getLegalPacmanActions()
    if Directions.STOP in legal:
        legal.remove(Directions.STOP)
    # Random choice between the legal options.
    return random.choice(legal)
```

**State Verification:**
```python
# From mlLearningAgents.py - GameStateFeatures testing pattern
def __hash__(self):
    """
    Allow states to be keys of dictionaries.
    We use a tuple of features that uniquely identify a state.
    """
    return hash((self.pacman_position, tuple(self.ghost_positions), 
                tuple(self.capsules), self.food.count()))

def __eq__(self, otherState):
    """
    Allow states to be compared for equality.
    Two states are equal if all their key features are equal.
    """
    if otherState is None:
        return False
    return (self.pacman_position == otherState.pacman_position and
            self.ghost_positions == otherState.ghost_positions and
            self.capsules == otherState.capsules and
            self.food.count() == otherState.food.count())
```

**Method Warnings:**
- Methods marked with `# WARNING: You will be tested on the functionality of this method` are tested by the framework
- These methods have fixed signatures that cannot be changed:
  - `computeReward(startState: GameState, endState: GameState) -> float`
  - `getQValue(state: GameStateFeatures, action: Directions) -> float`
  - `maxQValue(state: GameStateFeatures) -> float`
  - `learn(state, action, reward, nextState)`
  - `updateCount(state, action)`
  - `getCount(state, action) -> int`
  - `explorationFn(utility, counts) -> float`
  - `getAction(state: GameState) -> Directions`

## Fixtures and Test Data

**Layout Files:**
```
pacman_utils/layouts/
├── smallGrid.lay      # Tiny grid, fastest convergence
├── mediumGrid.lay     # Medium grid
├── smallClassic.lay   # Small classic Pacman map
├── mediumClassic.lay  # Standard map
├── originalClassic.lay # Full original layout
├── capsuleClassic.lay # Includes power capsules
└── testClassic.lay    # Test layout
```

**Hyperparameter Defaults (in mlLearningAgents.py):**
```python
def __init__(self,
             alpha: float = 0.2,      # Learning rate
             epsilon: float = 0.05,   # Exploration rate
             gamma: float = 0.8,      # Discount factor
             maxAttempts: int = 30,   # Exploration bonus threshold
             numTraining: int = 10):  # Training episodes
```

**Test Configuration:**
- `-x 2000`: Training episodes (output suppressed)
- `-n 2010`: Total episodes (2000 training + 10 testing)
- `-l smallGrid`: Use small layout for fast iteration
- `-q`: No graphics mode for speed

## Known Test Patterns

**Episode Lifecycle Testing:**
```python
def final(self, state: GameState):
    """
    Handle the end of episodes.
    This is called by the game after a win or a loss.
    """
    print(f"Game {self.getEpisodesSoFar()} just ended!")
    
    # Perform final learning update
    if self.last_state is not None and self.last_action is not None:
        final_state_features = GameStateFeatures(state)
        reward = self.computeReward(self.last_state.state, state)
        self.learn(self.last_state, self.last_action, reward, final_state_features)
    
    # Reset for next episode
    self.last_state = None
    self.last_action = None
    
    # Track episodes and disable learning at end of training
    self.incrementEpisodesSoFar()
    if self.getEpisodesSoFar() == self.getNumTraining():
        print('Training Done (turning off epsilon and alpha)')
        self.setAlpha(0)
        self.setEpsilon(0)
```

**Reward Computation Testing:**
```python
@staticmethod
def computeReward(startState: GameState, endState: GameState) -> float:
    """Test expected: rewards come from score delta + terminal bonuses"""
    reward = endState.getScore() - startState.getScore()
    
    if endState.isWin():
        reward += 500  # Reward for winning
    elif endState.isLose():
        reward -= 500  # Penalty for losing
        
    return reward
```

## Test Coverage Assessment

**Well-Tested Areas:**
- Q-learning Bellman update (`learn()` method)
- Epsilon-greedy action selection with tie-breaking
- State hashing and equality (`GameStateFeatures`)
- Exploration bonus application
- Reward computation with terminal bonuses
- Episode lifecycle and training completion

**Testing Gaps:**
- No unit tests for mathematical correctness (Bellman update formula)
- No regression tests tracking learning curves across versions
- No performance benchmarks for convergence speed
- No tests for hyperparameter sensitivity
- Edge cases not explicitly tested (e.g., all walls, no food, single tile)

## Integration Testing

**Framework Integration:**
- Tested against UC Berkeley Pacman framework (`pacman.py`, `pacman_utils/game.py`)
- Agents must implement `Agent` base class interface
- Must work with `GameState` objects from framework
- Compatible with built-in ghost agents

**Layout Testing:**
- Tested on 6+ different maze layouts (small to large)
- Success criteria: 89% win rate on smallGrid (2000 training episodes)

---

*Testing analysis: 2026-04-03*
