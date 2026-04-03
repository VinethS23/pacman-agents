# External Integrations

**Analysis Date:** 2026-04-03

## APIs & External Services

**Not applicable** - This is a self-contained educational project with no external API integrations.

## Data Storage

**Databases:**
- Not used - All game state and Q-learning data structures stored in-memory (Python dictionaries)

**Persistent Storage:**
- No file-based persistence implemented
- Q-table and action counts stored in memory during runtime only (`mlLearningAgents.py` lines 120-121)

**File Storage:**
- Game layouts: Local `.lay` files in `pacman_utils/layouts/` directory
  - No external storage dependency

**Caching:**
- No caching layer implemented
- In-memory dictionaries used for Q-table management

## Authentication & Identity

**Auth Provider:**
- Not applicable - Single-user educational system

## Monitoring & Observability

**Error Tracking:**
- Not implemented
- Errors propagate via Python stack traces

**Logs:**
- Console-based logging only
  - Game progress printed to stdout: `print(f"Game {self.getEpisodesSoFar()} just ended!")` (line 359 in `mlLearningAgents.py`)
  - Message format: Game event notifications during training/testing
- No persistent logging to files
- Suppressed via `-q` flag for silent execution

## CI/CD & Deployment

**Hosting:**
- None - Educational project intended for local execution

**CI Pipeline:**
- None detected

**Deployment:**
- Manual execution via command line
- No containerization or automated deployment

## Environment Configuration

**Required env vars:**
- None - Application uses only command-line arguments

**Config Files:**
- Game layouts: `pacman_utils/layouts/*.lay` files
  - Available layouts: `smallGrid`, `mediumGrid`, `smallClassic`, `mediumClassic`, `originalClassic`, `capsuleClassic`
- Hyperparameters passed via command-line (`-a` flag)

**Secrets location:**
- Not applicable - No secrets or credentials

## Webhooks & Callbacks

**Incoming:**
- Not applicable

**Outgoing:**
- Not applicable

## Game State & Internal Data Flow

**State Management:**
- Game state: `pacman.GameState` class manages current game board, positions, food, walls
- Agent state: `mlLearningAgents.QLearnAgent` maintains:
  - Q-table: `self.q_values` - Dictionary mapping `(GameStateFeatures, action)` tuples to float values
  - Action counts: `self.action_counts` - Tracks state-action pair visitation frequency
  - Episode tracking: `self.episodesSoFar` - Incremented each game

**Data Structures:**
- `GameStateFeatures` (`mlLearningAgents.py` lines 35-87) - Compressed state representation (Pacman position, ghost positions, capsules, food count)
- Game state serialization not implemented (in-memory only)

## Integration Points Within Codebase

**Agent Interface:**
- Base class: `pacman_utils.game.Agent` - Inherited by `QLearnAgent`
- Required methods: `getAction(state)`, `final(state)` 
- Call chain: Game engine → Agent.getAction() → Q-learning logic → Action returned

**Game Loop:**
- `pacman.py` - Game engine calls agent methods each turn
- Graphics dispatch: `pacman_utils.graphicsDisplay.py` updates UI when `-q` flag absent
- Text display: `pacman_utils.textDisplay.py` provides fallback console output

---

*Integration audit: 2026-04-03*
