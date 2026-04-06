# ROADMAP

## Milestone 1 — Core NN Agent

### Phase 1 — Neural Network Agent Implementation
**Goal:** Build a PyTorch Q-approximator; validate against Q-learning baseline.

**Depends on:** None (Q-learning already complete)

**Requirements mapped:** REQ-NN-1, REQ-NN-2, REQ-NN-3, REQ-NN-4, REQ-NN-5, REQ-CMP-1, REQ-CMP-2

**Success Criteria:**
- [ ] NNQAgent runs without errors on smallGrid
- [ ] Feature vector is fixed-length and consistent
- [ ] Network trains (loss decreases over episodes)
- [ ] Win rate >70% after 2000 training episodes
- [ ] Hyperparameter flag interface works
- [ ] Side-by-side comparison with QLearnAgent is possible

**Plans:** TBD — design phase in progress (feature vector scope, network size, hyperparameters)

---

### Phase 2 — Analysis & Optimization
**Goal:** Understand how NN agent differs from Q-learning; optimize performance.

**Depends on:** Phase 1 complete

**Likely includes:**
- Training curve comparison (NN vs Q-learning convergence speed)
- Hyperparameter sensitivity (learning rate, network size, exploration decay)
- Performance on mediumGrid (scaling test)
- Failure mode analysis (where does NN struggle vs Q-learning?)
- Bug fixes and stability improvements

**Success Criteria:** TBD after Phase 1

---

### Phase 3 — Advanced NN Architectures
**Goal:** Explore architectural improvements and larger state spaces.

**Depends on:** Phase 2 complete

**Likely includes:**
- Convolutional neural network for larger grids
- Dueling networks (value + advantage)
- Experience replay buffer
- Target network (frozen for stability)

**Success Criteria:** TBD

---

### Phase 4 — Alternative ML Approaches
**Goal:** Validate that NN is one of many approaches; implement contrasting techniques.

**Depends on:** Phase 1 + 2 complete

**Likely includes:**
- Policy gradient agent (REINFORCE)
- Actor-Critic methods
- Comparison with NN and tabular approaches

**Success Criteria:** TBD

---

## Progress

| Phase | Status | Goal | Notes |
|---|---|---|---|
| Phase 1 | Planning | NNQAgent implementation | Design decisions TBD; Claude guides, user writes code |
| Phase 2 | Not started | Analysis & optimization | After Phase 1 validates |
| Phase 3 | Not started | Advanced architectures | Larger grids, improved training |
| Phase 4 | Not started | Alternative approaches | Policy gradients, actor-critic |

---

## Critical Path
1. **Design** (current) — feature vector, network size, hyperparameters
2. **Implement** — user codes NNQAgent with Claude guidance
3. **Validate** — runs, learns, achieves >70% win rate
4. **Analyze** — compare with Q-learning, identify next improvements
5. **Iterate** — Phase 2+ based on Phase 1 findings
