# STATE

## Project Reference
**Project:** Pacman Agents — Experimental ML
**Source:** plan.md
**Core Value:** Learn different RL/ML techniques by implementing agents that solve the same problem

## Current Position
**Phase:** 1 — Neural Network Agent Implementation
**Status:** Design phase (awaiting architecture decisions)
**Last activity:** 2026-04-06 — Onboarding via /join-project; project scope clarified; Claude guiding user implementation

## Accumulated Context

### What's Working
- Q-Learning Agent (QLearnAgent) — tabular Q-learning, 89% win rate on smallGrid
- Framework integration — command-line flags functional
- State abstraction (GameStateFeatures) — hashing and equality working

### Blockers
None — ready to start Phase 1 design.

### Constraints
- Do not modify QLearnAgent without explicit permission
- Framework code is read-only
- Focus on agent implementations only

## Session Continuity
**Last session:** 2026-04-06 (current)
**Resume file:** None
**Context preserved:** Full project understanding via join-draft.md, plan.md, and this state file

## Next Steps
1. Lock in design decisions (feature vector scope, network size)
2. User implements NNQAgent with Claude guidance
3. Test on smallGrid
4. Compare with QLearnAgent
5. Phase 2: Analysis
