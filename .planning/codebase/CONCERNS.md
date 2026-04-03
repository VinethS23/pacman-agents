# Codebase Concerns

**Analysis Date:** 2026-04-03

## Tech Debt

**Uncached Food Count Calculation:**
- Issue: `pacman.py` line 386 has a TODO comment indicating that `numFood` should be cached but is recalculated on every food consumption
- Files: `pacman.py:386`
- Impact: In each game, every time Pacman eats food, the entire food grid is iterated to count remaining food. This is O(width*height) per consumption and accumulates significantly over 2000+ training episodes
- Fix approach: Cache numFood in GameStateData and decrement it when food is eaten, then increment when capsules reset it

**Priority Queue Implementation Inconsistency:**
- Issue: `pacman_utils/util.py` lines 177-182 show restored old behavior for debugging, with conflicting FIXME and FIXED comments indicating uncertainty about correct implementation
- Files: `pacman_utils/util.py:177-187`
- Impact: The priority queue uses (priority, count, item) tuples instead of (priority, item), which uses extra memory and may affect heap ordering stability across Python versions
- Fix approach: Document whether stability is required; if not, simplify to (priority, item). If yes, clarify the intent and remove conflicting comments

**Incomplete FIXME in Utility Module:**
- Issue: `pacman_utils/util.py` line 613 has a bare "# FIXME" comment with no description
- Files: `pacman_utils/util.py:613`
- Impact: Developer intent is unclear; context around line 613 may contain unresolved issue
- Fix approach: Add descriptive comment or resolve the issue

**Missing Defaults in Graphics Utilities:**
- Issue: `pacman_utils/graphicsUtils.py` line 48 has `pass  # XXX need defaults here` indicating incomplete initialization
- Files: `pacman_utils/graphicsUtils.py:48`
- Impact: Graphics system may fail or behave unpredictably if this default case is reached
- Fix approach: Define appropriate default values for graphics initialization

## Known Bugs

**State Access Bug in Q-Learning Agent:**
- Symptoms: Line 201 of `mlLearningAgents.py` accesses `self.state.getLegalPacmanActions()` in maxQValue(), but self.state may not be initialized on the first call
- Files: `mlLearningAgents.py:201`
- Trigger: Call maxQValue() before any getAction() call in an episode
- Impact: AttributeError: 'NoneType' object has no attribute 'getLegalPacmanActions' on first state evaluation during training
- Current code: `self.state` is set to None initially (line 49 through implicit __init__) but only set in getAction() at line 314. If maxQValue() is called before getAction(), self.state is None
- Workaround: maxQValue() is only called from learn() which is only called from getAction(), so bug is latent but will trigger if code structure changes

**Static Method Type Checking Flaw:**
- Symptoms: `pacman.py` lines 440 and 646 have TODO comments about type checking ("TODO Check for type of other", "TODO: could this exceed the total time")
- Files: `pacman.py:440,646`
- Trigger: Comparisons with incompatible types or edge cases in game state duration
- Impact: May silently fail or produce incorrect comparisons if wrong types are passed

**Global State Accumulation:**
- Symptoms: `pacman.py` line 85 maintains a static set `GameState.explored` that stores every game state generated
- Files: `pacman.py:85-89,136-137`
- Trigger: Running multiple training episodes (2000+)
- Impact: Memory leak. The explored set grows unbounded across episodes. With 2000 training episodes and potentially thousands of states per episode, this can consume significant memory
- Workaround: getAndResetExplored() is called between games to clear it, but this is only effective if called properly by the display system
- Current status: Commented out on line 98, suggesting previous debugging use

## Performance Bottlenecks

**Q-Table Hashing with GameStateFeatures:**
- Problem: GameStateFeatures uses custom __hash__() and __eq__() for state comparison (lines 57-87 in `mlLearningAgents.py`). The hash includes tuple(self.ghost_positions) and tuple(self.capsules), which create new tuples on every hash call
- Files: `mlLearningAgents.py:57-87`
- Cause: Python will call __hash__ and __eq__ thousands of times per training episode for Q-table lookups. Creating tuples in __hash__ is inefficient
- Impact: Slows down training by 10-30% depending on episode length and state complexity
- Improvement path: Pre-compute hash at GameStateFeatures initialization and cache it; use immutable representations (tuples) in constructor instead of converting on-the-fly

**Food Count Evaluation:**
- Problem: GameStateFeatures.__hash__() calls self.food.count() on every hash operation (line 67). The food grid is potentially large (mediumClassic is 19x21)
- Files: `mlLearningAgents.py:67`
- Cause: count() iterates the entire grid; with thousands of Q-table lookups per episode, this accumulates
- Improvement path: Cache food.count() at state creation time or use bit representation

**Exploration Bonus Applied Every Episode:**
- Problem: explorationFn() applies a 1000.0 bonus to all state-action pairs with count < maxAttempts on every action selection (lines 291-294 in `mlLearningAgents.py`)
- Files: `mlLearningAgents.py:335`
- Cause: This creates a "soft cap" where explored states get the same exploration bonus repeatedly, defeating the purpose of count-based exploration by mid-training
- Impact: After ~1000 episodes, the exploration bonus becomes stale and the agent explores the same suboptimal actions repeatedly instead of refining learned values
- Improvement path: Switch to exploration decay (reduce epsilon over time) or use visit counting with decreasing bonus magnitude

## Fragile Areas

**GameStateFeatures Equality and Hashing:**
- Files: `mlLearningAgents.py:57-87`
- Why fragile: Depends on internal state object structure. If GameState API changes (e.g., capsules representation), equality breaks silently. The __eq__ method compares food.count() equality but not actual food positions, which could cause hash collisions with different food grids
- Safe modification: Add explicit type checks in __eq__; consider using frozenset for capsules instead of list for hashability
- Test coverage: No unit tests for GameStateFeatures equality; only integration tested through Q-learning performance

**State Transition in Learning Loop:**
- Files: `mlLearningAgents.py:318-349` (getAction method)
- Why fragile: Stores last_state and last_action as instance variables. If getAction() is called twice in succession without final() being called, the learning update will use stale state information
- Safe modification: Add assertions that last_state is None at episode start; document contract clearly
- Test coverage: Depends on game framework calling methods in correct order; if test harness changes call sequence, silent bugs occur

**Hardcoded Reward Values:**
- Files: `mlLearningAgents.py:169-171` (win/loss bonuses of 500/-500)
- Why fragile: Reward shaping is tightly coupled to food rewards (10 points per food in `pacman.py:382). If food reward changes, entire learning dynamic breaks. No configuration mechanism to tune separately
- Safe modification: Make reward bonuses configurable parameters; add validation that rewards are consistent
- Test coverage: README documents expected performance (89% win rate) which will regress if rewards change

**Global GameState.explored Set:**
- Files: `pacman.py:85,136-137`
- Why fragile: Static set accumulates across multiple game runs. Clearing it requires calling getAndResetExplored() which is optional and framework-dependent. If not cleared between test runs, state space grows unpredictably
- Safe modification: Use instance variable instead of static; pass display/state manager to agent initialization for cleanup hooks
- Test coverage: Manual testing only; no automated cleanup verification

## Scaling Limits

**Q-Table Memory Growth:**
- Current capacity: With state representation of (pacman_pos, ghost_positions, capsules, food_count), state space is bounded by 19*21 * (4*19*21) * 4 * max_food on mediumClassic ≈ 30+ million possible states
- Limit: After ~10,000 episodes with fully-explored layouts, q_values dictionary could grow to millions of entries, consuming 1-2 GB of memory depending on state complexity
- Scaling path: Implement function approximation (neural network) instead of tabular Q-learning; or use state aggregation/abstraction to reduce state space

**Episode Runtime Growth:**
- Current capacity: Each episode runs ~500-2000 steps on smallGrid. Training 2000 episodes takes ~30-60 minutes
- Limit: Larger layouts (originalClassic) would make training impractical (hours+)
- Scaling path: Implement experience replay and mini-batch updates to decouple training from episode speed; use GPU acceleration if available

## Security Considerations

**Pickle Deserialization (Game Replay):**
- Risk: `pacman.py:609-612` uses pickle.load() to deserialize game history files without validation
- Files: `pacman.py:609-612`
- Current mitigation: User-controlled filename, local file system access only
- Recommendations: Add file format validation before deserialization; use safer serialization formats (JSON, pickle protocol with restrictions)

## Test Coverage Gaps

**GameStateFeatures Hash Collision Testing:**
- What's not tested: Equality correctness between states that should be equal; collision detection for different food layouts with same count
- Files: `mlLearningAgents.py:57-87`
- Risk: Silent equality bugs could cause Q-values to be confused between different food distributions
- Priority: High

**Edge Cases in Q-Learning Update:**
- What's not tested: Learning with zero legal actions; behavior when maxQValue() returns 0.0; final episode step with no successor
- Files: `mlLearningAgents.py:193-236`
- Risk: Division by zero or incorrect updates in corner cases
- Priority: Medium

**Graphics System Initialization:**
- What's not tested: graphicsUtils initialization with missing defaults; behavior when display is None
- Files: `pacman_utils/graphicsUtils.py:48`
- Risk: Silent failures or crashes in text-mode execution
- Priority: Medium

**State Space Exhaustiveness:**
- What's not tested: Whether all layout-specific features (one-way corridors, dead ends) are properly represented in GameStateFeatures
- Files: `mlLearningAgents.py:35-88`
- Risk: Agent may fail to distinguish between strategically different positions
- Priority: Low

---

*Concerns audit: 2026-04-03*
