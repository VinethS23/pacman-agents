# Technology Stack

**Analysis Date:** 2026-04-03

## Languages

**Primary:**
- Python 3.13.2 - Core implementation language for all agent logic and game engine

**Secondary:**
- Python 2 compatibility layer - Via `six` module for cross-version support (future-proofing for legacy systems)

## Runtime

**Environment:**
- CPython 3.13.2
- Uses `__future__` imports for Python 2/3 compatibility (despite Python 3 being primary)

**Package Manager:**
- pip 25.0.1
- Lockfile: Not present - minimal dependencies approach

## Frameworks

**Core:**
- UC Berkeley Pacman AI Framework - Game engine, agent base classes, game state management (`pacman_utils/`)
- No external ML/RL frameworks - Q-learning algorithm implemented from scratch

**Build/Dev:**
- None detected - Direct Python execution

## Key Dependencies

**Critical:**
- `six` 1.17.0 - Python 2/3 compatibility library (provides `six.moves` for cross-version imports)
  - Used by: `graphicsUtils.py`, `graphicsDisplay.py`, `util.py`, `game.py`, `layout.py`, `ghostAgents.py`
  - Specifically: Cross-version range, zip, map functions via `six.moves`

**Graphics & Display:**
- `tkinter` (Python stdlib) - GUI rendering for game visualization
  - Imported via `six.moves.tkinter` for compatibility
  - Used in: `pacman_utils/graphicsUtils.py`, `pacman_utils/graphicsDisplay.py`

## Standard Library Usage

**Core Modules:**
- `random` - Random action selection, exploration
- `sys` - System operations, platform detection (Windows vs Unix for graphics)
- `pathlib.Path` - File path handling
- `typing.Union` - Type hints
- `heapq` - Heap operations in utilities
- `inspect` - Introspection for function signatures
- `signal` - Signal handling for timeouts
- `time` - Timing and delays
- `math` - Mathematical operations for graphics
- `traceback` - Error handling
- `functools.reduce` - Functional utilities

## Configuration

**Environment:**
- No environment variables required for core functionality
- Command-line argument parsing: `pacman.py` accepts flags (`-p`, `-x`, `-n`, `-l`, `-q`, `-a`)

**Build:**
- No build configuration files (setup.py, pyproject.toml, requirements.txt)
- Virtual environment: `.venv/` directory present (not committed to git)

## Platform Requirements

**Development:**
- Python 3.13.2
- Tkinter (usually included with Python, may require separate install on some systems)
- Cross-platform compatible (Windows/Unix detection in code)

**Production:**
- Python 3.13.2 runtime
- No external services or cloud dependencies
- Headless execution supported via `-q` flag (no graphics)

## Execution

**Entry Points:**
- `pacman.py` - Main game engine executable

**Running:**
```bash
# Without graphics (recommended)
python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid -q

# With graphics
python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid
```

---

*Stack analysis: 2026-04-03*
