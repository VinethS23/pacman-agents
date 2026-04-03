# Coding Conventions

**Analysis Date:** 2026-04-03

## Naming Patterns

**Files:**
- Descriptive snake_case: `mlLearningAgents.py`, `sampleAgents.py`
- Some files use camelCase mixing: `pacman_utils/pacmanAgents.py`, `graphicsDisplay.py`
- Purpose: Class/module names in filenames are descriptive

**Functions:**
- camelCase for methods and functions: `getAction()`, `getQValue()`, `maxQValue()`, `incrementEpisodesSoFar()`
- Accessor methods follow getter/setter pattern: `getEpisodesSoFar()`, `setEpsilon()`, `setAlpha()`
- Private/internal methods use camelCase: `computeReward()`, `updateCount()`
- Static methods use camelCase: `computeReward()` (marked with `@staticmethod`)

**Variables:**
- camelCase for instance variables: `self.q_values`, `self.action_counts`, `self.last_state`, `self.last_action`
- camelCase for local variables: `legal_actions`, `new_q_value`, `old_q_value`, `best_value`, `best_actions`
- camelCase for parameters: `alpha`, `epsilon`, `gamma`, `maxAttempts`, `numTraining`
- Underscores for multi-word: `pacman_position`, `ghost_positions`, `final_state_features`
- CONSTANTS in UPPER_CASE: `NORTH`, `SOUTH`, `EAST`, `WEST`, `STOP` (in `Directions` class)

**Types and Classes:**
- PascalCase for class names: `GameStateFeatures`, `QLearnAgent`, `RandomAgent`, `LeftTurnAgent`, `GreedyAgent`
- PascalCase for configuration/data classes: `GameState`, `Configuration`, `AgentState`, `Grid`

## Code Style

**Formatting:**
- No explicit formatter configured (no .prettierrc or black config found)
- 4-space indentation (Python standard)
- Line breaks used for logical separation

**Linting:**
- No linter configuration found (.eslintrc, .pylintrc, etc.)
- Code follows PEP 8 conventions by convention

**Type Hints:**
- Type annotations used in function signatures: `state: GameState`, `action: Directions`, `utility: float`, `counts: int`
- Return types annotated: `-> float`, `-> int`, `-> Directions`, `-> bool`
- Keyword argument annotations in `__init__`: `alpha: float = 0.2`, `epsilon: float = 0.05`
- Parameter types in docstrings when not using annotations

## Import Organization

**Order:**
1. Future imports for Python 2/3 compatibility: `from __future__ import absolute_import`, `from __future__ import print_function`
2. Standard library imports: `import random`, `import sys`, `import os`, `import traceback`
3. Relative imports from local packages: `from pacman import Directions, GameState`, `from pacman_utils.game import Agent`

**Path Aliases:**
- No explicit path aliases configured
- Relative imports use dot notation: `from .game import Agent`, `from .util import manhattanDistance`
- Absolute imports from package root: `from pacman import Directions`

## Error Handling

**Patterns:**
- Exceptions raised with descriptive messages: `raise Exception('Can\'t generate a successor of a terminal state.')`
- Try/finally blocks for resource cleanup: `try: return Layout(...) finally: f.close()` (in `layout.py`)
- Null checks with `is None`: `if other is None: return False`, `if self.last_state is not None`
- Silent defaults: `self.q_values.get((state, action), 0.0)` returns 0.0 if key not found
- Boolean checks: `if not legal_actions: return 0.0`

## Logging

**Framework:** Console print statements

**Patterns:**
- Direct print() calls: `print(f"Game {self.getEpisodesSoFar()} just ended!")`
- String formatting with f-strings (Python 3.6+): `f"Game {self.getEpisodesSoFar()} just ended!"`
- Old-style formatting in some utility code: `'%s\n%s' % (msg, '-' * len(msg))`
- Minimal logging - only used for episode milestones and training completion
- No structured logging or log levels

## Comments

**When to Comment:**
- Method purpose documented in docstrings, not inline
- Algorithm explanation at method level: "Q-learning update rule: Q(s,a) = Q(s,a) + alpha * [R(s) + gamma * max_a' Q(s',a') - Q(s,a)]"
- Implementation notes on non-obvious logic: "# Break ties for instances with equal q-values"
- Inline comments for key steps in conditional blocks

**JSDoc/TSDoc:**
- Python docstrings (triple quotes) used for all classes and public methods
- Standard format with description, Args, and Return sections:
```python
def getQValue(self,
              state: GameStateFeatures,
              action: Directions) -> float:
    """
    Args:
        state: A given state
        action: Proposed action to take

    Returns:
        Q(state, action)
    """
```
- Warnings for tested methods: `# WARNING: You will be tested on the functionality of this method`

## Function Design

**Size:** 
- Small, focused methods (10-30 lines typical)
- Accessors are 1-3 lines
- Core logic methods like `getAction()` are 20-30 lines
- Larger utility functions (game.py) up to 50+ lines for complex state transitions

**Parameters:**
- Limited parameters (2-5 typical)
- Type hints on all parameters
- Default values for hyperparameters: `alpha: float = 0.2`
- Keyword-only arguments in constructors

**Return Values:**
- Single return value per function
- Explicit `return` statements (no implicit None)
- Consistent return types matching annotations

## Module Design

**Exports:**
- No explicit `__all__` declarations
- Public methods named without leading underscore
- Classes exposed at module level

**Barrel Files:**
- `__init__.py` exists in `pacman_utils/` but is empty (no re-exports)
- Each module imported directly by its path

## Special Patterns

**Q-Learning Specific:**
- State-action pairs stored as tuples: `(state, action)` used as dictionary keys
- Exploration function pattern: `explorationFn()` provides count-based optimization bonus
- Count tracking for state-action pairs: `action_counts` dictionary parallel to `q_values`
- Reward shaping with terminal bonuses: `+500` for win, `-500` for loss

**Agent Base Pattern:**
- All agents inherit from `Agent` base class (in `pacman_utils/game.py`)
- Required method `getAction(state)` implemented by all agents
- Optional `final(state)` method called at episode end
- `registerInitialState()` hook available but not used

**State Representation:**
- Two-level state abstraction:
  - `GameState`: Full game state (in `pacman.py`) with accessor methods
  - `GameStateFeatures`: Wrapped state (in `mlLearningAgents.py`) with hashable features for Q-table

---

*Convention analysis: 2026-04-03*
