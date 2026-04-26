#!/usr/bin/env python3
"""
evaluate.py

Evaluation harness for Pacman ML agents on smallGrid.
Produces reward curves, ablation plot, and a results table.

Usage:
    python evaluate.py

Outputs:
    eval_reward_curves.png
    eval_ablation.png
    eval_results_table.md
    eval_results.json
"""
import sys, os, json, statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pacman
import pacman_utils.layout as layout_module
import pacman_utils.textDisplay as textDisplay
from pacman_utils.ghostAgents import RandomGhost

from mlLearningAgents import QLearnAgent, NNQAgent
from sampleAgents import RandomAgent, RandomishAgent

NUM_TRAINING = 2000
NUM_EVAL = 1000
LAYOUT = 'smallGrid'


# ---------------------------------------------------------------------------
# Tracked agent wrappers
# ---------------------------------------------------------------------------

class TrackedQLearnAgent(QLearnAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_scores = []
        self.train_wins = []
        self._step_count = 0

    def getAction(self, state):
        self._step_count += 1
        return super().getAction(state)

    def final(self, state):
        super().final(state)
        if self.getEpisodesSoFar() <= self.getNumTraining():
            self.train_scores.append(state.getScore())
            self.train_wins.append(state.isWin())
        self._step_count = 0


class AblationQLearnAgent(TrackedQLearnAgent):
    """QLearnAgent with exploration bonus disabled — always returns raw utility."""
    def explorationFn(self, utility, counts):
        return utility


class TrackedNNQAgent(NNQAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_scores = []
        self.train_wins = []
        self._step_count = 0

    def getAction(self, state):
        self._step_count += 1
        return super().getAction(state)

    def final(self, state):
        super().final(state)
        if self.getEpisodesSoFar() <= self.getNumTraining():
            self.train_scores.append(state.getScore())
            self.train_wins.append(state.isWin())
        self._step_count = 0


# ---------------------------------------------------------------------------
# Game runner helpers
# ---------------------------------------------------------------------------

def _null_display():
    return textDisplay.NullGraphics()


def run_games(agent, num_games, num_training):
    the_layout = layout_module.getLayout(LAYOUT)
    ghosts = [RandomGhost(1)]
    display = _null_display()
    return pacman.runGames(
        the_layout, agent, ghosts, display,
        numGames=num_games,
        record=False,
        numTraining=num_training,
    )


def pacman_steps(game):
    """Count Pacman's moves from moveHistory (agent index 0)."""
    return sum(1 for agent_idx, _ in game.moveHistory if agent_idx == 0)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def episodes_to_win_rate(wins, threshold=0.5, window=100):
    """First episode where rolling win rate (window=100) reaches threshold."""
    for i in range(window, len(wins) + 1):
        if sum(wins[i - window:i]) / window >= threshold:
            return i
    return None


def q_coverage(agent):
    """
    Returns (unique_states, avg_actions_per_state) from action_counts.
    avg_actions_per_state is how many distinct actions were tried per state on average.
    """
    counts = getattr(agent, 'action_counts', {})
    if not counts:
        return None, None
    unique_states = len({s for s, _ in counts})
    avg_actions = len(counts) / unique_states
    return unique_states, avg_actions


def smooth(data, window=50):
    out = []
    for i in range(len(data)):
        lo = max(0, i - window + 1)
        out.append(sum(data[lo:i + 1]) / (i - lo + 1))
    return out


# ---------------------------------------------------------------------------
# Evaluate functions
# ---------------------------------------------------------------------------

def evaluate_learning_agent(name, agent):
    print(f"\n{'='*60}")
    print(f"Training {name} — {NUM_TRAINING} episodes")
    print('='*60)
    # All training, data captured in agent.final()
    run_games(agent, num_games=NUM_TRAINING, num_training=NUM_TRAINING)

    print(f"Evaluating {name} — {NUM_EVAL} episodes")
    eval_games = run_games(agent, num_games=NUM_EVAL, num_training=0)

    eval_scores = [g.state.getScore() for g in eval_games]
    eval_wins = [g.state.isWin() for g in eval_games]
    win_steps = [pacman_steps(g) for g in eval_games if g.state.isWin()]

    unique_states, avg_actions = q_coverage(agent)

    result = {
        'name': name,
        'train_scores': agent.train_scores,
        'train_wins': agent.train_wins,
        'eval_scores': eval_scores,
        'eval_wins': eval_wins,
        'win_rate': sum(eval_wins) / len(eval_wins),
        'avg_score': statistics.mean(eval_scores),
        'avg_steps_per_win': statistics.mean(win_steps) if win_steps else None,
        'episodes_to_50pct': episodes_to_win_rate(agent.train_wins),
        'score_std_last200': (
            statistics.stdev(agent.train_scores[-200:])
            if len(agent.train_scores) >= 2 else None
        ),
        'unique_states': unique_states,
        'avg_actions_per_state': avg_actions,
    }

    _print_summary(result)
    return result


def evaluate_baseline(name, agent_class):
    print(f"\n{'='*60}")
    print(f"Evaluating baseline {name} — {NUM_EVAL} episodes")
    print('='*60)
    agent = agent_class()
    eval_games = run_games(agent, num_games=NUM_EVAL, num_training=0)

    eval_scores = [g.state.getScore() for g in eval_games]
    eval_wins = [g.state.isWin() for g in eval_games]
    win_steps = [pacman_steps(g) for g in eval_games if g.state.isWin()]

    result = {
        'name': name,
        'eval_scores': eval_scores,
        'eval_wins': eval_wins,
        'win_rate': sum(eval_wins) / len(eval_wins),
        'avg_score': statistics.mean(eval_scores),
        'avg_steps_per_win': statistics.mean(win_steps) if win_steps else None,
    }

    _print_summary(result)
    return result


def _print_summary(r):
    print(f"  Win rate:        {r['win_rate']*100:.1f}%")
    print(f"  Avg score:       {r['avg_score']:.1f}")
    if r.get('avg_steps_per_win') is not None:
        print(f"  Avg steps/win:   {r['avg_steps_per_win']:.1f}")
    if r.get('episodes_to_50pct') is not None:
        print(f"  Episodes to 50%: {r['episodes_to_50pct']}")
    if r.get('score_std_last200') is not None:
        print(f"  Score std (200): {r['score_std_last200']:.1f}")
    if r.get('unique_states') is not None:
        print(f"  Unique states:   {r['unique_states']}")
        print(f"  Avg actions/st:  {r['avg_actions_per_state']:.2f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_reward_curves(results, path='eval_reward_curves.png'):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in results:
        if 'train_scores' in r and r['train_scores']:
            ax.plot(smooth(r['train_scores'], window=50), label=r['name'])
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Score (smoothed, window=50)')
    ax.set_title(f'Reward Curves — {LAYOUT}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_ablation(ql, abl, path='eval_ablation.png'):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(smooth(ql['train_scores'], 50), label='With exploration bonus')
    ax.plot(smooth(abl['train_scores'], 50), label='No exploration bonus', linestyle='--')
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Score (smoothed)')
    ax.set_title('Ablation: Exploration Bonus — Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    window = 100
    for r, label, ls in [(ql, 'With bonus', '-'), (abl, 'No bonus', '--')]:
        wins = r['train_wins']
        rolling = [
            sum(wins[max(0, i - window):i + 1]) / min(i + 1, window)
            for i in range(len(wins))
        ]
        ax.plot(rolling, label=label, linestyle=ls)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.7, label='50% threshold')
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Rolling win rate (window=100)')
    ax.set_title('Ablation: Exploration Bonus — Win Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

AGENT_ORDER = ['Q-Learner', 'Q-Learner (no bonus)', 'NNQ-Learner', 'RandomishAgent', 'RandomAgent']

def _fmt(v, pct=False, dec=1):
    if v is None:
        return 'n/a'
    if pct:
        return f'{v*100:.{dec}f}%'
    if dec == 0:
        return str(int(round(v)))
    return f'{v:.{dec}f}'


def print_and_save_table(results):
    by_name = {r['name']: r for r in results}

    headers = ['Metric'] + AGENT_ORDER
    rows = [
        ('Win rate (smallGrid)',
         [_fmt(by_name.get(n, {}).get('win_rate'), pct=True) for n in AGENT_ORDER]),
        ('Avg score',
         [_fmt(by_name.get(n, {}).get('avg_score'), dec=0) for n in AGENT_ORDER]),
        ('Avg steps per win',
         [_fmt(by_name.get(n, {}).get('avg_steps_per_win'), dec=0) for n in AGENT_ORDER]),
        ('Episodes to 50% win rate',
         [_fmt(by_name.get(n, {}).get('episodes_to_50pct'), dec=0) for n in AGENT_ORDER]),
        ('Score std (last 200 eps)',
         [_fmt(by_name.get(n, {}).get('score_std_last200'), dec=1) for n in AGENT_ORDER]),
        ('Q-table unique states',
         [_fmt(by_name.get(n, {}).get('unique_states'), dec=0) for n in AGENT_ORDER]),
        ('Avg actions explored/state',
         [_fmt(by_name.get(n, {}).get('avg_actions_per_state'), dec=2) for n in AGENT_ORDER]),
    ]

    lines = []
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('|' + '|'.join(['-' * (len(h) + 2) for h in headers]) + '|')
    for metric, vals in rows:
        lines.append('| ' + metric + ' | ' + ' | '.join(vals) + ' |')

    md = '\n'.join(lines)
    print('\n' + '=' * 80)
    print('RESULTS TABLE')
    print('=' * 80)
    print(md)

    with open('eval_results_table.md', 'w') as f:
        f.write(md + '\n')
    print('\nSaved eval_results_table.md')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ql_agent = TrackedQLearnAgent(
        numTraining=NUM_TRAINING, epsilon=0.05, alpha=0.2, gamma=0.8, maxAttempts=30
    )
    ablation_agent = AblationQLearnAgent(
        numTraining=NUM_TRAINING, epsilon=0.05, alpha=0.2, gamma=0.8, maxAttempts=30
    )
    nnq_agent = TrackedNNQAgent(
        numTraining=NUM_TRAINING, epsilon=0.3, alpha=0.001, gamma=0.9
    )

    ql_result = evaluate_learning_agent('Q-Learner', ql_agent)
    ablation_result = evaluate_learning_agent('Q-Learner (no bonus)', ablation_agent)
    nnq_result = evaluate_learning_agent('NNQ-Learner', nnq_agent)

    randomish_result = evaluate_baseline('RandomishAgent', RandomishAgent)
    random_result = evaluate_baseline('RandomAgent', RandomAgent)

    all_results = [ql_result, ablation_result, nnq_result, randomish_result, random_result]

    plot_reward_curves([ql_result, ablation_result, nnq_result])
    plot_ablation(ql_result, ablation_result)
    print_and_save_table(all_results)

    # Save raw data for reproducibility
    serialisable = {}
    for r in all_results:
        serialisable[r['name']] = {
            k: v for k, v in r.items()
            if isinstance(v, (int, float, str, type(None), list, bool))
        }
    with open('eval_results.json', 'w') as f:
        json.dump(serialisable, f, indent=2)
    print('Saved eval_results.json')
