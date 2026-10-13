"""Development benchmarks (GPQA Diamond, LiveCodeBench v6); no source locks, private samples or architecture inspection.

GPQA prompts and extraction and LiveCodeBench prompts, extraction and comparison are the same
functions the trusted final evaluation uses. The final score also includes held-out benchmarks
that are not part of this module, so a development result is never comparable to it.
"""
import csv
import hashlib
import math
import random
from pathlib import Path
from . import livecodebench
from .text_benchmarks import completion_text, extract_choice, _run

# Development benchmarks and their paths under the public benchmarks directory.
PUBLIC_FILES = {'gpqa_diamond': 'gpqa_diamond.csv', 'livecodebench_v6': 'livecodebench-v6'}


def load_gpqa(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    required = {'Question', 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3'}
    if not rows or any(not required.issubset(row) or not all(row[key] for key in required) for row in rows):
        raise ValueError('Expected complete GPQA questions and choices')
    return rows


def harmonic_mean(*scores: float) -> float:
    """Equal weight per benchmark; any zero forces the result to zero."""
    if not scores or not all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in scores):
        raise ValueError('Benchmark scores must be finite fractions in [0,1]')
    if any(value == 0 for value in scores):
        return 0.0
    return len(scores) / sum(1 / value for value in scores)


def evaluate_gpqa(transport, rows: list[dict], *, max_tokens: int = 32768,
                  timeout: float = 3600, concurrency: int = 1, temperature: float = 1.0) -> dict:
    if not rows:
        raise ValueError('GPQA Diamond cannot be empty')
    def score_one(indexed):
        index, row = indexed
        choices = [row['Correct Answer']] + [row[f'Incorrect Answer {i}'] for i in range(1, 4)]
        order = list(range(4))
        random.Random(hashlib.sha256(row['Question'].encode()).hexdigest()).shuffle(order)
        correct = 'ABCD'[order.index(0)]
        prompt = row['Question'] + '\n\n' + '\n'.join(f'{"ABCD"[i]}. {choices[j]}' for i, j in enumerate(order))
        prompt += '\n\nReason through the question, then put your final answer on its own line as ANSWER: A (or B, C, D).'
        text = completion_text(transport, prompt, max_tokens=max_tokens, timeout=timeout, temperature=temperature)
        predicted = extract_choice(text, 'ABCD')
        return {'index': index, 'correct': predicted == correct, 'predicted_answer': predicted, 'output': text}
    outcomes = _run(list(enumerate(rows)), score_one, concurrency)
    return {'score': sum(row['correct'] for row in outcomes) / len(rows), 'n': len(rows),
            'details': {'metric': 'accuracy', 'shots': 0, 'attempts_per_question': 1, 'temperature': temperature,
                        'outcomes': outcomes}}


# Suite v5 default generation, llm-diffusion's graded settings: the model thinks before answering, so a request
# allows 32,768 generated tokens, is sampled at temperature 1.0 and gets an hour. ternary-ptq overrides these with
# v4's greedy 4,096 tokens in its task config and validate.py.
DEFAULT_GENERATION = {'concurrency': 32, 'max_tokens': 32768, 'timeout': 3600, 'temperature': 1.0}


def _generation_settings(config: dict) -> dict:
    concurrency = config.get('generation_concurrency', DEFAULT_GENERATION['concurrency'])
    max_tokens = config.get('max_tokens', DEFAULT_GENERATION['max_tokens'])
    timeout = config.get('request_timeout_seconds', DEFAULT_GENERATION['timeout'])
    temperature = config.get('temperature', DEFAULT_GENERATION['temperature'])
    if type(concurrency) is not int or not 1 <= concurrency <= 32:
        raise ValueError('generation_concurrency must be an integer from 1 to 32')
    if type(max_tokens) is not int or not 1 <= max_tokens <= 65536:
        raise ValueError('max_tokens must be an integer from 1 to 65536')
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('request_timeout_seconds must be finite and positive')
    if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
        raise ValueError('temperature must be a number from 0 to 2')
    return {'concurrency': concurrency, 'max_tokens': max_tokens, 'timeout': timeout, 'temperature': temperature}


def evaluate_public(transport, benchmarks_dir: str | Path, config: dict) -> dict:
    """Development runner over the benchmark files given to the agent; same prompts and scoring as the final suite.

    LiveCodeBench solutions are executed locally in subprocesses (see
    frontier_ml.validation.livecodebench: this runs model-generated code on this machine).
    `limit` truncates each benchmark, so a partial run is a smoke test. Even a complete run
    covers only the development benchmarks, so its mean is not comparable to the
    final score, which also includes held-out benchmarks.
    """
    root = Path(benchmarks_dir)
    settings = _generation_settings(config)
    names = list(dict.fromkeys(config.get('benchmarks') or PUBLIC_FILES))
    unknown = [name for name in names if name not in PUBLIC_FILES]
    if unknown:
        raise ValueError(f"Unknown benchmark(s): {', '.join(unknown)}; choose from {', '.join(PUBLIC_FILES)}")
    limit = config.get('limit')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive integer')
    results = {}
    for name in names:
        path = root / PUBLIC_FILES[name]
        if name == 'gpqa_diamond':
            rows = load_gpqa(path)
            results[name] = evaluate_gpqa(transport, rows[:limit] if limit else rows, **settings)
        else:
            code_config = {'generation_concurrency': settings['concurrency'], 'max_tokens': settings['max_tokens'],
                           'request_timeout_seconds': settings['timeout'], 'temperature': settings['temperature']}
            results[name] = livecodebench.evaluate_local(transport, path, code_config, limit=limit)
    scores = [results[name]['score'] for name in names]
    return {'score': sum(scores) / len(scores),
            'details': {'metric': 'mean_of_development_benchmarks',
                        'accuracy': {name: results[name]['score'] for name in names},
                        'benchmarks': results, 'limit': limit, 'generation': settings,
                        'comparable_to_final_score': False,
                        'note': ('Mean of the selected development benchmarks only. The final score is the '
                                 'equal-weight mean of four benchmarks, two of which are held out and each normalized '
                                 'between its anchors, so this number is not comparable to it.')}}
