"""Pinned four-benchmark quality suite for ternary PTQ and diffusion conversion.

GPQA Diamond and LiveCodeBench v6 are development benchmarks shared with the agent's validate.py;
MATH-500 and MMLU-Pro are held out from the agent's data and implemented only in the trusted
reasoning_benchmarks module. The contract names neither (operator decision, 2026-09-30): an LLM judge decides
whether a reply's final answer is equivalent to the reference, so the answer format needs no disclosure.

SUITE (task.yaml `verifier.config.language_suite`) names the scored sample. v4 scores GPQA Diamond, MATH-500
and LiveCodeBench v6 complete and the fixed, category-stratified MMLU-Pro sample that select_mmlu_pro draws
here from the complete pinned file (unchanged from v3), so grading a compliant server takes hours rather than
days. v4 replaced AIME 2026 (30 problems, five sampled attempts) with MATH-500 (500 problems, one attempt
each) because the models under evaluation scored near zero on AIME, which carried no signal into the
aggregate. v5 keeps v4's samples and prompts and states the generation settings (max_tokens, temperature,
concurrency, timeout) in each task's config. llm-diffusion lets the model reason: 32,768 generated tokens at
temperature 1.0, an hour per request, 32 concurrent (validation.language.DEFAULT_GENERATION, also the default).
ternary-ptq keeps v4's greedy 4,096 tokens, at which the reasoning base models were mostly cut off before
answering in thinking mode (bf16 Qwen3.5-4B: GPQA 0.03), so its servers answer without thinking.
v6 (2026-10-01) makes MMLU-Pro zero-shot: the request is one instruction ("The following is a multiple choice
question. Think step by step, then output "The answer is (X)" at the end."), the question and its options, like
GPQA's. v3-v5 prepended the five official category-matched validation CoT demonstrations and named the category.
The scored sample, its seed and the other three benchmarks are unchanged; the validation file is no longer read.
Every MMLU-Pro normalization ceiling measured under v5 is stale until re-measured under this prompt.
Since 2026-10-01 (operator decision) the suite's raw score and every language task's normalized reward are the
equal-weight arithmetic MEAN of the four benchmarks, not the harmonic mean: the harmonic mean, dominated by its
smallest term, zeroed any submission at or below one benchmark's floor (GPQA at chance on 198 questions is within one
standard error for many 4B models). The raw harmonic mean is still reported in details as a diagnostic.
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import os
import random
import re
import threading
import time
from pathlib import Path

from .architecture import verify_review
from ..validation.errors import EvaluationError, VerificationFailure
from ..validation.language import harmonic_mean, evaluate_gpqa, _generation_settings
from .sources import download_file, source_lock
from .reasoning_benchmarks import (JUDGE_MODEL, completion_text, extract_choice, evaluate_math500,
                                   evaluate_mmlu_pro, math500_prompt, _run)

SUITE = 'gpqa-math500-lcbv6-mmlupro-v6'
# A task may grade against several identical copies of the submission on separate H100s (verifier.config
# inference_replicas; llm-diffusion runs 4). The client concurrency the contract states is per copy, so the suite sends
# concurrency x replicas requests at once through a round-robin ReplicaTransport; LiveCodeBench, whose every request
# also costs a CPU sandbox, is capped at lcb_max_concurrency (default 64) to spare the sandbox provider.
MAX_REPLICAS = 8
DEFAULT_LCB_MAX_CONCURRENCY = 64
BENCHMARKS = ('gpqa_diamond', 'math_500', 'livecodebench_v6', 'mmlu_pro')
# Complete pinned sources; the loaders verify each byte for byte before anything is scored.
SOURCE_COUNTS = dict(zip(BENCHMARKS, (198, 500, 1055, 12032), strict=True))
# Since v3 the suite scores MMLU_PRO_PER_CATEGORY questions from each of MMLU-Pro's 14 categories (1,400 of
# 12,032), chosen by select_mmlu_pro from the complete file at load time. Scored at up to 4,096 generated tokens
# per question, the whole split was about 12,000 requests and roughly 10M generated tokens, 80 % of the suite's
# grading wall-clock; the fixed sample keeps all four components of the aggregate while bounding grading.
MMLU_PRO_PER_CATEGORY = 100
MMLU_PRO_CATEGORIES = 14
# The selection seed is a fixed string, not derived from SUITE: v4, v5 and v6 keep the v3 seed verbatim so the scored
# MMLU-Pro sample (ids digest 1592cb62...) is identical to v3's. A new sample is a deliberate new seed, not a side effect of renaming the suite.
MMLU_PRO_SEED = 'gpqa-aime2026-lcbv6-mmlupro-v3/mmlu-pro'
# Scored sample sizes; evaluate_suite refuses a result that does not cover exactly these.
EXPECTED_COUNTS = {**SOURCE_COUNTS, 'mmlu_pro': MMLU_PRO_PER_CATEGORY * MMLU_PRO_CATEGORIES}
# Chat requests per benchmark: exactly one completion per scored item in every benchmark.
REQUEST_COUNTS = dict(EXPECTED_COUNTS)




def prepare_gpqa(destination: str | Path, token: str | None = None) -> Path:
    """Requires an HF token whose user has accepted the source dataset terms."""
    lock = source_lock()['gpqa']
    path = Path(destination) / lock['file']
    download_file(f"https://huggingface.co/datasets/{lock['repo']}/resolve/{lock['revision']}/{lock['file']}", path, token=token)
    load_gpqa(path)
    return path


def fetch_qwen(destination: str | Path) -> str:
    """Explicit operator command: downloads ~9.34 GB before offline evaluation."""
    from huggingface_hub import snapshot_download
    lock = source_lock()['qwen']
    return snapshot_download(lock['repo'], revision=lock['revision'], local_dir=str(destination))


def fetch_diffusion_base(destination: str | Path) -> str:
    """Download the explicitly selected pinned diffusion base and tokenizer."""
    from huggingface_hub import snapshot_download
    lock = source_lock()['diffusion']
    return snapshot_download(lock['repo'], revision=lock['revision'], local_dir=str(destination))


def load_gpqa(path: str | Path, expected_rows: int = 198) -> list[dict]:
    path = Path(path)
    payload = path.read_bytes()
    # HF stores this CSV as a Git blob, so bind the complete original bytes.
    digest = hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
    if path.is_symlink() or len(payload) != 1373492 or digest != '7589e3e467d69a1dceb126a60c4108d6d4f1d166':
        raise ValueError('GPQA Diamond does not match the exact pinned upstream CSV')
    with path.open(newline='') as file:
        rows = list(csv.DictReader(file))
    required = {'Question', 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3'}
    if len(rows) != expected_rows or any(not required.issubset(row) for row in rows):
        raise ValueError('Expected official GPQA Diamond CSV: 198 complete questions')
    if len({row['Question'] for row in rows}) != len(rows) or any(not all(row[key].strip() for key in required) for row in rows):
        raise ValueError('GPQA questions/choices must be complete and unique')
    return rows


def measure_single_stream(transport, tokenizer, prompts: list[str], *, minimum_tps: float,
                          max_tokens: int = 1024, temperature: float = 1.0,
                          clock=time.monotonic, warmup: bool = True, retries: int = 1) -> dict:
    """End-to-end single request at a time; count decoded text with trusted tokenizer.

    The decoded text is the reply's reasoning_content (when the server returns one) followed by its content:
    a reasoning model spends most of the 1,024 tokens thinking, and the gate measures decoding speed, not
    answer length. A reply cut off at max_tokens counts, since the gate never scores the answer.

    No server-reported usage or token IDs are trusted. Includes prefill and HTTP
    time, excludes verifier tokenization time. Every prompt must exceed X TPS. There is no minimum reply
    length: the fixed prefill and HTTP time already make a short reply slower per token, and an empty reply
    scores zero.

    The contract promises end-to-end request time, so each request is timed by the clock nearest the server:
    a transport offering `request_timed` (HarborHTTPTransport, the trusted HTTP client inside the GPU sandbox)
    reports the HTTP exchange's own elapsed seconds, and the grader's control-plane RPCs around every request
    (upload, exec and an interpreter start: 0.35-0.5 s on Modal, which zeroed servers at 393-413 tok/s that
    decoded at 500-530 tok/s in the sandbox) are not charged to the server. Any other transport (the public
    validator's direct HTTP client, which carries no such overhead) is timed with the host `clock`. Every
    record names the clock it used: a sandbox-timed record keeps the host time as `host_seconds`, and an attempt
    recorded as an error (no sandbox time to report) is timed with the host clock.

    With `warmup`, the first prompt is sent once untimed so first-request kernel compilation and cache capture are
    not charged to the server, and the remaining prompts are measured. The warm-up is drawn from the same pool of
    scored requests as the measured prompts (speed_prompts), so no request of the gate is recognisable by its
    content; until 2026-09-30 it was a fixed synthetic paragraph prompt. A prompt that misses the gate is re-sent
    once before the gate fails: a measured requirement should not hinge on a single cold or hiccuping request.
    Every attempt is recorded.
    """
    if not prompts or not math.isfinite(minimum_tps) or minimum_tps <= 0:
        raise ValueError('Diffusion needs nonempty trusted speed prompts and positive X TPS')
    prompts = list(prompts)
    if warmup and len(prompts) < 2:
        raise ValueError('The speed gate needs a warm-up prompt followed by at least one measured prompt')

    # A server that is merely too slow must fail the gate with a measurement, not a transport error:
    # allow several times the passing duration before giving up on a request, and record a request
    # that errors, times out or returns no text as a failed attempt.
    request_timeout = max(30.0, 4 * max_tokens / minimum_tps)
    request_timed = getattr(transport, 'request_timed', None)

    def send(messages):
        payload = {'model': 'submission', 'messages': messages, 'temperature': temperature,
                   'max_tokens': max_tokens, 'stream': False}
        start = clock()
        try:
            if request_timed is None:
                response = transport.request('POST', '/v1/chat/completions', payload, timeout=request_timeout)
                elapsed, timing = clock() - start, {'clock': 'host'}
            else:
                response, elapsed = request_timed('POST', '/v1/chat/completions', payload, timeout=request_timeout)
                timing = {'clock': 'sandbox', 'host_seconds': clock() - start}
            message = response['choices'][0]['message']
            reasoning = message.get('reasoning_content')
            text = message['content']
            if isinstance(reasoning, str) and (isinstance(text, str) or text is None):
                text = reasoning + (text or '')
        except Exception as error:  # noqa: BLE001 - the gate records the failure and moves on
            return {'tokens': 0, 'seconds': clock() - start, 'clock': 'host', 'tokens_per_second': 0.0, 'passed': False,
                    'error': f'{type(error).__name__}: {str(error)[:200]}'}
        if not isinstance(text, str) or elapsed <= 0:
            return {'tokens': 0, 'seconds': elapsed, **timing, 'tokens_per_second': 0.0, 'passed': False,
                    'error': 'response had no text content'}
        decoded = len(tokenizer.encode(text, add_special_tokens=False))
        # Re-tokenizing a reply that stopped at max_tokens can land slightly above it (the reasoning parser strips
        # its think markers), so the count is capped rather than rejected: extra text never raises the speed.
        tokens = min(decoded, max_tokens)
        tps = tokens / elapsed
        return {'tokens': tokens, 'decoded_tokens': decoded, 'seconds': elapsed, **timing, 'tokens_per_second': tps,
                'passed': tokens > 0 and tps > minimum_tps}

    def messages_of(prompt):
        # A string is one user turn; a message list (a sampled scoring request) is sent as-is.
        return [{'role': 'user', 'content': prompt}] if isinstance(prompt, str) else prompt

    used = 'sandbox' if request_timed is not None else 'host'
    results = []
    if warmup:
        results.append({**send(messages_of(prompts.pop(0))), 'warmup': True, 'passed': None})
    for prompt in prompts:
        messages = messages_of(prompt)
        for attempt in range(1 + retries):
            result = send(messages)
            results.append({**result, 'attempt': attempt + 1})
            if result['passed']:
                break
        else:
            return {'passed': False, 'clock': used, 'requests': results}
    return {'passed': True, 'clock': used, 'requests': results}




def _mmlu_rank(row: dict, seed: str = MMLU_PRO_SEED) -> str:
    return hashlib.sha256(f'{seed}\0{row["id"]}'.encode()).hexdigest()


def select_mmlu_pro(rows: list[dict], *, per_category: int = MMLU_PRO_PER_CATEGORY,
                    seed: str = MMLU_PRO_SEED) -> list[dict]:
    """The scored MMLU-Pro sample: from each category, the `per_category` questions of the complete
    official test split whose sha256(seed, id) is smallest, returned in official question order.

    The rule uses only the suite seed and the official question IDs, so the sample is fixed for the suite
    version, the same on every machine, independent of file order, operator and submission, and
    reproducible from the pinned file. A partial source is refused: the sample is always drawn from the
    whole verified split, never from a file that already contains only the sample."""
    if type(per_category) is not int or per_category < 1 or not isinstance(seed, str) or not seed:
        raise ValueError('MMLU-Pro selection needs a positive per-category count and a seed')
    if len(rows) != SOURCE_COUNTS['mmlu_pro'] or len({row['id'] for row in rows}) != len(rows):
        raise ValueError(f"MMLU-Pro selection requires the complete official split of {SOURCE_COUNTS['mmlu_pro']} unique questions")
    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(row['category'], []).append(row)
    if len(by_category) != MMLU_PRO_CATEGORIES or any(len(group) < per_category for group in by_category.values()):
        raise ValueError(f'MMLU-Pro selection requires {MMLU_PRO_CATEGORIES} categories of at least {per_category} questions')
    selected = [row for category in sorted(by_category)
                for row in heapq.nsmallest(per_category, by_category[category], key=lambda row: _mmlu_rank(row, seed))]
    return sorted(selected, key=lambda row: row['source_id'])


def mmlu_pro_selection(rows: list[dict] | None = None) -> dict:
    """The selection rule as recorded by preflight and in results, with a digest of the selected IDs."""
    record = {'rule': 'per category, the questions with the smallest sha256(seed + NUL + id)',
              'seed': MMLU_PRO_SEED, 'per_category': MMLU_PRO_PER_CATEGORY, 'categories': MMLU_PRO_CATEGORIES,
              'source_questions': SOURCE_COUNTS['mmlu_pro'], 'scored_questions': EXPECTED_COUNTS['mmlu_pro']}
    if rows is not None:
        record['ids_sha256'] = hashlib.sha256('\n'.join(row['id'] for row in rows).encode()).hexdigest()
    return record


def _check_suite(config: dict) -> None:
    """A rendered package names the suite it was built for; stale packages must not grade under another sample."""
    named = config.get('language_suite', SUITE)
    if named != SUITE:
        raise ValueError(f'Trusted config names language suite {named!r}; this verifier implements {SUITE}')


# Request resilience. One slow or dropped request must not zero an eight-hour trial: transport failures
# (connection errors, HTTP error statuses, bridge/exec failures, malformed or non-JSON responses) are retried
# on the identical payload; a request that still fails, or that times out, scores as an incorrect answer for
# its item and is recorded in details.failed_requests. More than
# FAILED_REQUEST_PERCENT of a benchmark's requests failing is a broken server, not a hiccup: the evaluation
# stops there and fails closed, unless the task sets failed_request_limit_percent (null: no limit, every failed
# request only scores wrong). The speed gate uses the raw transport and is never retried.
REQUEST_ATTEMPTS = 3            # the request plus two retries
RETRY_BACKOFF_SECONDS = (1, 3)  # before the second and the third attempt
FAILED_REQUEST_PERCENT = 5
RECORDED_ERRORS = 5             # error strings kept per benchmark
_TIMEOUT_TEXT = re.compile(r'time(?:d\s*)?out|deadline', re.IGNORECASE)


def _is_timeout(error: BaseException) -> bool:
    """Socket/exec timeouts, the sandbox bridge's TimeoutError name, and transport deadline messages."""
    return isinstance(error, TimeoutError) or bool(_TIMEOUT_TEXT.search(str(error)))


def _malformed(response) -> str | None:
    """Why the scorers' completion_content would reject this response, or None; same rules as that function."""
    choices = response.get('choices') if isinstance(response, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return 'response lacks exactly one choice'
    if choices[0].get('finish_reason') == 'length':
        return None
    message = choices[0].get('message')
    if not isinstance(message, dict) or not isinstance(message.get('content'), str) or message.get('tool_calls'):
        return 'choice has no text content'
    return None


def _failed_completion() -> dict:
    """What a failed request yields to the scorers: an empty text completion, which every scorer marks wrong."""
    return {'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': ''}, 'finish_reason': 'request_failed'}]}


def _describe(payload) -> str:
    messages = payload.get('messages') if isinstance(payload, dict) else None
    content = messages[-1].get('content') if isinstance(messages, list) and messages and isinstance(messages[-1], dict) else None
    return repr(content[:60]) if isinstance(content, str) else '<no prompt>'


def failed_request_limit(config: dict) -> int | None:
    """The percent of one benchmark's requests that may fail or time out before the evaluation fails closed;
    None (`failed_request_limit_percent: null`) scores every failed request as wrong and never stops."""
    limit = config.get('failed_request_limit_percent', FAILED_REQUEST_PERCENT)
    if limit is not None and (type(limit) is not int or not 0 <= limit <= 100):
        raise ValueError('failed_request_limit_percent must be an integer from 0 to 100, or null for no limit')
    return limit


def scoring_deadline(config: dict) -> float | None:
    """Seconds of quality scoring (from the first scored request on) after which requests still unsent score wrong
    without being sent, and requests in flight are cut off; `scoring_deadline_seconds`, None by default (no deadline).
    A task whose per-request timeout times its request count could outlast the verifier sets one, so a slow server
    still gets a grade, with its unanswered items scored zero, instead of the verifier timing out with none."""
    seconds = config.get('scoring_deadline_seconds')
    if seconds is not None and (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError('scoring_deadline_seconds must be a positive number of seconds, or null for no deadline')
    return seconds


def _limit_clause(limit: int | None) -> str:
    """How grading treats failed requests beyond scoring each wrong, for the self-check's notes."""
    return f'zero above {limit}% of a benchmark' if limit is not None else 'with no limit on how many'


def inference_replicas(config: dict) -> int:
    replicas = config.get('inference_replicas', 1)
    if type(replicas) is not int or not 1 <= replicas <= MAX_REPLICAS:
        raise ValueError(f'inference_replicas must be an integer from 1 to {MAX_REPLICAS}')
    return replicas


def _effective_concurrency(config: dict, settings: dict) -> dict:
    """Per-replica settings scaled to the whole pool: concurrency x replicas for the chat benchmarks, capped for LCB."""
    replicas = inference_replicas(config)
    lcb_max = config.get('lcb_max_concurrency', DEFAULT_LCB_MAX_CONCURRENCY)
    if type(lcb_max) is not int or not 1 <= lcb_max <= MAX_REPLICAS * 32:
        raise ValueError(f'lcb_max_concurrency must be an integer from 1 to {MAX_REPLICAS * 32}')
    total = settings['concurrency'] * replicas
    return {'replicas': replicas, 'per_replica_concurrency': settings['concurrency'], 'concurrency': total,
            'lcb_concurrency': min(total, lcb_max), 'lcb_max_concurrency': lcb_max}


class ReplicaTransport:
    """One logical server over several identical copies of the submission: each request goes to the next copy
    (round-robin, thread-safe), so a client sending concurrency x replicas requests at once loads every copy with the
    per-copy concurrency the contract states. Only chat requests pass through here; the speed gate measures one copy."""

    def __init__(self, transports):
        self.transports = list(transports)
        if not self.transports:
            raise ValueError('ReplicaTransport needs at least one server')
        self._next = 0
        self._lock = threading.Lock()

    def __len__(self):
        return len(self.transports)

    def request(self, method, path, payload=None, **kwargs):
        with self._lock:
            index = self._next
            self._next = (index + 1) % len(self.transports)
        return self.transports[index].request(method, path, payload, **kwargs)


class ResilientTransport:
    """The suite's view of the server: retries transport failures, scores requests that still fail or time out
    as wrong instead of raising, and fails the evaluation closed once a benchmark passes the failure limit.

    `begin(benchmark, requests)` attributes the following requests to a benchmark; the evaluators run one
    after another, so one current benchmark suffices while their threads share this object.

    `deadline` (a `clock()` time, see scoring_deadline) caps every request's timeout at the time left; once it has
    passed, requests are not sent and score wrong, recorded as timeouts and counted in `unsent_at_deadline`."""

    def __init__(self, transport, *, sleep=None, limit_percent=FAILED_REQUEST_PERCENT, deadline=None, clock=None):
        self.transport, self.sleep, self.limit_percent = transport, sleep or time.sleep, limit_percent
        self.deadline, self.clock = deadline, clock or time.monotonic
        self.lock = threading.Lock()
        self.benchmark: str | None = None
        self.failures: dict[str, dict] = {}
        self.deadline_reached = False

    def begin(self, benchmark: str, requests: int) -> None:
        if type(requests) is not int or requests < 1:
            raise ValueError('A benchmark sends a positive number of requests')
        with self.lock:
            self.benchmark = benchmark
            self.failures[benchmark] = {'requests': requests, 'failed': 0, 'timeouts': 0, 'retried': 0,
                                        'unsent_at_deadline': 0, 'errors': []}

    def summary(self) -> dict:
        with self.lock:
            return {'limit_percent': self.limit_percent, 'attempts': REQUEST_ATTEMPTS,
                    'backoff_seconds': list(RETRY_BACKOFF_SECONDS), 'deadline_reached': self.deadline_reached,
                    'total': sum(record['failed'] for record in self.failures.values()),
                    'per_benchmark': {name: {**record, 'errors': list(record['errors'])} for name, record in self.failures.items()}}

    def _record(self, error: str, *, timeout: bool) -> None:
        with self.lock:
            record = self.failures[self.benchmark]
            record['failed'] += 1
            record['timeouts'] += timeout
            if len(record['errors']) < RECORDED_ERRORS:
                record['errors'].append(error)
            if self.limit_percent is not None and record['failed'] * 100 > self.limit_percent * record['requests']:
                raise VerificationFailure(
                    f"{self.benchmark}: {record['failed']} of {record['requests']} requests failed or timed out, more than "
                    f"{self.limit_percent}%; the server is not serving reliably. First errors: {record['errors']}")

    def request(self, method, path, payload=None, **kwargs):
        if self.benchmark is None:
            raise RuntimeError('ResilientTransport.begin() must name the benchmark before requests')
        errors = []
        for attempt in range(REQUEST_ATTEMPTS):
            if attempt:
                self.sleep(RETRY_BACKOFF_SECONDS[attempt - 1])
            if self.deadline is not None:
                remaining = self.deadline - self.clock()
                if remaining <= 0:
                    with self.lock:
                        self.deadline_reached = True
                        self.failures[self.benchmark]['unsent_at_deadline'] += 1
                    self._record(f'{_describe(payload)} not answered by the scoring deadline', timeout=True)
                    return _failed_completion()
                timeout = kwargs.get('timeout')
                kwargs = {**kwargs, 'timeout': remaining if timeout is None else min(timeout, remaining)}
            try:
                response = self.transport.request(method, path, payload, **kwargs)
            except Exception as error:  # the request layer only; the scorers' own errors never pass through here
                text = f'{type(error).__name__}: {str(error)[:200]}'
                if _is_timeout(error):
                    # A slow request is not retried: it would cost the timeout again, and the item is scored wrong.
                    self._record(f'{_describe(payload)} timed out on attempt {attempt + 1}: {text}', timeout=True)
                    return _failed_completion()
                errors.append(text)
                continue
            problem = _malformed(response)
            if problem is None:
                if attempt:
                    with self.lock:
                        self.failures[self.benchmark]['retried'] += 1
                return response
            errors.append(f'malformed response: {problem}')
        self._record(f'{_describe(payload)} failed {REQUEST_ATTEMPTS} attempts: ' + ' | '.join(errors), timeout=False)
        return _failed_completion()


def _benchmark_paths(config: dict, data_path: str | Path) -> dict[str, Path]:
    paths = dict(config.get('benchmark_paths', {}))
    # GPQA remains --data for the shared verifier CLI, but other three benchmarks
    # are mandatory; no fallback to a smaller suite or former Terminal-Bench.
    paths.setdefault('gpqa_diamond', config.get('gpqa_path', data_path))
    if any(not paths.get(name) for name in BENCHMARKS):
        raise ValueError('benchmark_paths must provide all four required quality benchmarks')
    return {name: Path(paths[name]) for name in BENCHMARKS}




def _code_settings(config: dict, settings: dict) -> dict:
    return {'grading_resources': config.get('grading_resources'), 'grading_images': config.get('grading_images'),
            **config.get('livecodebench', {}),
            'generation_concurrency': settings['concurrency'], 'temperature': settings['temperature'],
            'max_tokens': settings['max_tokens'], 'request_timeout_seconds': settings['timeout']}


def _load_reasoning_data(config: dict, data_path: str | Path) -> tuple[dict, dict]:
    from .benchmark_data import load_prepared
    _check_suite(config)
    paths = _benchmark_paths(config, data_path)
    rows = {'gpqa_diamond': load_gpqa(paths['gpqa_diamond']),
            'math_500': load_prepared('math_500', paths['math_500']),
            'mmlu_pro': load_prepared('mmlu_pro', paths['mmlu_pro'])}
    for name in ('gpqa_diamond', 'math_500', 'mmlu_pro'):
        if len(rows[name]) != SOURCE_COUNTS[name]:
            raise ValueError(f'Incomplete {name}: requires the complete pinned source of {SOURCE_COUNTS[name]} samples')
    # The scored MMLU-Pro sample is drawn here, in trusted code, from the complete verified file.
    rows['mmlu_pro'] = select_mmlu_pro(rows['mmlu_pro'])
    for name in ('gpqa_diamond', 'math_500', 'mmlu_pro'):
        if len(rows[name]) != EXPECTED_COUNTS[name]:
            raise ValueError(f'Incomplete {name}: requires {EXPECTED_COUNTS[name]} samples')
    return paths, rows


def suite_judge(config: dict):
    """The trusted MATH-500 answer judge and its model name.

    validate_suite_config binds the judge into config['judge'] during preflight, so a missing credential fails
    before any GPU is provisioned; a config that arrives here without one is bound the same way, from the
    verifier's own environment (runtime.judge.judge_from_config: judge_model, judge_timeout_seconds,
    judge_api_key_env, judge_endpoint). Tests pass a callable of their own."""
    model = config.setdefault('judge_model', JUDGE_MODEL)
    judge = config.get('judge')
    if not callable(judge):
        from ..runtime.judge import judge_from_config
        judge = config['judge'] = judge_from_config(config)
    return judge, model


def validate_suite_config(config: dict, data_path: str | Path) -> dict:
    """Preflight, before provisioning GPUs or calling model APIs: checks the settings and every data file and binds
    the MATH-500 judge (its credential must exist now, not after a GPU is running)."""
    from .livecodebench import preflight
    settings = _generation_settings(config)
    _, judge_model = suite_judge(config)
    paths, rows = _load_reasoning_data(config, data_path)
    code = preflight(paths['livecodebench_v6'], _code_settings(config, settings))
    return {'suite': SUITE, 'benchmarks': dict(EXPECTED_COUNTS), 'source_counts': dict(SOURCE_COUNTS),
            'requests': dict(REQUEST_COUNTS), 'failed_request_limit_percent': failed_request_limit(config),
            'scoring_deadline_seconds': scoring_deadline(config),
            'mmlu_pro_selection': mmlu_pro_selection(rows['mmlu_pro']),
            'livecodebench': code, 'generation': _generation_settings(config), 'math_500_judge': judge_model,
            'replication': _effective_concurrency(config, settings)}


def evaluate_suite(transport, data_path: str | Path, config: dict) -> dict:
    from .livecodebench import evaluate as evaluate_livecodebench, preflight
    paths, rows = _load_reasoning_data(config, data_path)
    settings = _generation_settings(config)
    pool = _effective_concurrency(config, settings)
    # The evaluators see the pooled concurrency (per-replica concurrency x replicas); the request settings are unchanged.
    settings = {**settings, 'concurrency': pool['concurrency']}
    code_settings = {**settings, 'concurrency': pool['lcb_concurrency']}
    judge, judge_model = suite_judge(config)
    # Fail before any completions if coding inputs or safe execution are absent.
    preflight(paths['livecodebench_v6'], _code_settings(config, code_settings))
    # Every scoring request goes through the resilient wrapper (the speed gate does not).
    limit = failed_request_limit(config)
    seconds = scoring_deadline(config)
    server = ResilientTransport(transport, limit_percent=limit,
                                deadline=None if seconds is None else time.monotonic() + seconds)
    results = {}
    server.begin('gpqa_diamond', REQUEST_COUNTS['gpqa_diamond'])
    results['gpqa_diamond'] = evaluate_gpqa(server, rows['gpqa_diamond'], **settings)
    server.begin('math_500', REQUEST_COUNTS['math_500'])
    results['math_500'] = evaluate_math500(server, rows['math_500'], judge=judge, judge_model=judge_model, **settings)
    server.begin('mmlu_pro', REQUEST_COUNTS['mmlu_pro'])
    results['mmlu_pro'] = evaluate_mmlu_pro(server, rows['mmlu_pro'], **settings)
    code_config = _code_settings(config, code_settings)
    server.begin('livecodebench_v6', REQUEST_COUNTS['livecodebench_v6'])
    results['livecodebench_v6'] = evaluate_livecodebench(server, paths['livecodebench_v6'], code_config)
    for name in BENCHMARKS:
        if results[name].get('n') != EXPECTED_COUNTS[name]:
            raise ValueError(f'{name} evaluator did not score every required sample')
    failed = server.summary()
    for name, record in failed['per_benchmark'].items():
        if limit is not None and record['failed'] * 100 > limit * record['requests']:
            raise VerificationFailure(f"{name}: more than {limit}% of requests failed; see failed_requests")
    scores = [results[name]['score'] for name in BENCHMARKS]
    return {'score': sum(scores) / len(scores),
            'details': {'metric': 'four_benchmark_mean', 'suite': SUITE, 'harmonic_mean': harmonic_mean(*scores),
                        'benchmark_weights': dict.fromkeys(BENCHMARKS, 0.25),
                        'sample_sizes': dict(EXPECTED_COUNTS), 'source_counts': dict(SOURCE_COUNTS),
                        'requests': dict(REQUEST_COUNTS), 'failed_requests': failed,
                        'scoring_deadline_seconds': seconds,
                        'mmlu_pro_selection': mmlu_pro_selection(rows['mmlu_pro']),
                        'generation': _generation_settings(config), 'replication': pool,
                        'benchmarks': {name: results[name] for name in BENCHMARKS}}}







def validate_diffusion_config(config: dict) -> dict:
    base = source_lock()['diffusion']
    if config.get('base_model', base['repo']) != base['repo'] or config.get('base_revision', base['revision']) != base['revision']:
        raise ValueError(f"Diffusion must start with pinned {base['repo']}")
    speed = config.get('minimum_tokens_per_second')
    if type(speed) not in (int, float) or not math.isfinite(speed) or speed <= 0:
        raise ValueError('Diffusion X TPS is required; task is not runnable until configured')
    prompts = config.get('speed_prompts')
    if prompts is not None and (not isinstance(prompts, list) or len(prompts) < 2 or not all(map(_speed_prompt, prompts))):
        raise ValueError('speed_prompts must list the warm-up prompt and at least one measured prompt or chat message list')
    if not config.get('trusted_tokenizer_path'):
        raise ValueError('Diffusion requires trusted_tokenizer_path (fetch_diffusion_tokenizer)')
    _check_tokenizer(Path(config['trusted_tokenizer_path']), base)
    return base


def _speed_prompt(prompt) -> bool:
    if isinstance(prompt, str):
        return bool(prompt)
    return (isinstance(prompt, list) and bool(prompt) and all(
        isinstance(message, dict) and message.keys() == {'role', 'content'} and message['role'] in ('system', 'user')
        and isinstance(message['content'], str) and message['content'] for message in prompt))


def _check_tokenizer(path: Path, base: dict) -> None:
    for name, expected in base['tokenizer_sha256'].items():
        file = path / name
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError('Trusted tokenizer does not match the pinned diffusion base')


# Files AutoTokenizer loads for the pinned base (tokenizer_sha256 names the two pinned by SHA-256).
DIFFUSION_TOKENIZER_FILES = ('special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json')


def fetch_diffusion_tokenizer(destination: str | Path, *, client=None, endpoint: str | None = None) -> Path:
    """Download only the pinned diffusion base's tokenizer files into verifier storage and check the pin.

    Each file is verified against the digest Hugging Face records for the pinned commit, then the
    SHA-256 pins in sources.json are checked. Nothing comes from the submission or agent artifacts."""
    import httpx
    from spin.transfer import HF_ENDPOINT, _fetch_one, hf_files
    base, endpoint = source_lock()['diffusion'], endpoint or HF_ENDPOINT
    names = sorted({*DIFFUSION_TOKENIZER_FILES, *base['tokenizer_sha256']})
    destination = Path(destination)
    own = client is None
    client = client or httpx.Client(timeout=120, follow_redirects=True)
    try:
        listed = {entry['path']: entry for entry in hf_files(
            {'repo': base['repo'], 'revision': base['revision'], 'repo_type': 'model', 'prefix': ''},
            client=client, endpoint=endpoint)}
        missing = [name for name in names if name not in listed]
        if missing:
            raise ValueError(f"{base['repo']}@{base['revision'][:12]} lacks tokenizer files: {', '.join(missing)}")
        for name in names:
            _fetch_one(f"{endpoint}/{base['repo']}/resolve/{base['revision']}/{name}", destination / name,
                       listed[name], client)
    finally:
        if own:
            client.close()
    _check_tokenizer(destination, base)
    return destination


# Speed-gate requests are scored requests, drawn at random for every grade and built by the scorers themselves so
# their messages match scoring byte for byte: a server cannot tell a gate request from a scored one by its content
# (until 2026-09-30 the eight measured prompts were a fixed seeded sample and the warm-up a synthetic paragraph
# prompt). The gate uses the suite temperature and max_tokens=1024, and counts reasoning and answer tokens alike.
# The pools favour prompts that draw long replies, so the gate measures sustained decoding: GPQA, level-5 MATH-500
# and hard LiveCodeBench. MMLU-Pro was left out when its prompt carried five CoT demonstrations (about 30-190 words)
# that encourage short replies; the pools were kept unchanged when v6 made it zero-shot, so the gate's draw did not
# move. SPEED_WARMUP extra GPQA draws join the sample; after shuffling, the first request of the
# list is the untimed warm-up (measure_single_stream) and the rest are measured.
SPEED_SAMPLE = {'gpqa_diamond': 3, 'math_500': 3, 'livecodebench_v6': 2}
SPEED_WARMUP = 1
SPEED_MATH_LEVEL = 5


class _RecordingTransport:
    """Captures each chat request a scorer sends (messages, and the whole payload with its timeout) and answers with
    empty text."""
    def __init__(self):
        self.messages = []
        self.requests = []

    def request(self, method, path, payload=None, **kwargs):
        self.messages.append(payload['messages'])
        self.requests.append({'payload': payload, 'timeout': kwargs.get('timeout')})
        return {'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': ''}, 'finish_reason': 'stop'}]}


def _speed_pick(benchmark: str, rows, count: int, rng) -> list:
    """`count` rows drawn without replacement by the grade's own random source."""
    rows = list(rows)
    if len(rows) < count:
        raise ValueError(f'Too few {benchmark} rows for the speed sample')
    return rng.sample(rows, count)


def _speed_draw(config: dict, gpqa_rows, math_rows, problems, sample: dict, rng) -> list[list[dict]]:
    """`sample` (benchmark: measured draws) plus SPEED_WARMUP GPQA draws from these rows, built by the scorers and
    shuffled; the first is the warm-up. speed_prompts and the self-check's public_speed_prompts share it."""
    from .livecodebench import _evaluate_rows
    settings = {**_generation_settings(config), 'concurrency': 1}
    recorder = _RecordingTransport()
    gpqa = _speed_pick('gpqa_diamond', gpqa_rows, sample['gpqa_diamond'] + SPEED_WARMUP, rng)
    evaluate_gpqa(recorder, gpqa, **settings)
    if sample.get('math_500'):
        hardest = (row for row in math_rows if row.get('level') == SPEED_MATH_LEVEL)
        for row in _speed_pick('math_500', hardest, sample['math_500'], rng):
            recorder.messages.append([{'role': 'user', 'content': math500_prompt(row)}])
    hard = (row for row in problems if row['difficulty'] == 'hard')
    picked = _speed_pick('livecodebench_v6', hard, sample['livecodebench_v6'], rng)
    _evaluate_rows(recorder, picked, None, {**_code_settings(config, settings), 'generation_concurrency': 1})
    prompts = recorder.messages
    rng.shuffle(prompts)
    return prompts


def speed_prompts(config: dict, data_path: str | Path, *, rng=None) -> list[list[dict]]:
    """SPEED_WARMUP + sum(SPEED_SAMPLE) chat requests of the scored suite, exactly as the suite sends them, in random
    order; the first is the warm-up. `rng` defaults to the operating system's randomness (a new draw per grade);
    tests pass a seeded random.Random."""
    from .livecodebench import iter_problems
    rng = rng or random.SystemRandom()
    paths, rows = _load_reasoning_data(config, data_path)
    return _speed_draw(config, rows['gpqa_diamond'], rows['math_500'], iter_problems(paths['livecodebench_v6']),
                       SPEED_SAMPLE, rng)


def verify_diffusion_review(submission, review_path, base: dict, *, required_checks=None) -> dict:
    review = verify_review(submission, review_path, 'llm-diffusion',
                           required_checks if required_checks is not None else
                           ['pinned_base_start', 'diffusion_generation', 'actual_inference_uses_reviewed_weights'])
    # Submission bytes alone are insufficient when the benchmark's starting
    # checkpoint changes. A previous Ling or Maple review must not remain valid.
    policy = review.get('policy', {})
    if policy.get('base_model') != base['repo'] or policy.get('base_revision') != base['revision']:
        raise ValueError('Architecture review belongs to a different diffusion starting checkpoint')
    expected_digest = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    if review.get('policy_digest') != expected_digest:
        raise ValueError('Architecture review policy digest is missing or inconsistent')
    return review


# Self-checks (frontier_ml.selfcheck) of the language tasks send public inputs only, built by the same scorers as
# graded requests: GPQA Diamond and LiveCodeBench v6 from the agent's public benchmarks folder; MATH-500 and MMLU-Pro
# are held out. They fetch the GPQA CSV (1.4 MB) and the smallest LiveCodeBench file (test6.jsonl: 175 problems, 80
# hard, 134 MB), not the whole 4.2 GB folder. Nothing here is scored, judged or executed.
SELFCHECK_PUBLIC = ('benchmarks/gpqa_diamond.csv', 'benchmarks/livecodebench-v6/test6.jsonl')
SELFCHECK_LCB_FILES = ('test6.jsonl',)
# The speed gate on public inputs: the three level-5 MATH-500 draws become GPQA draws, the public prompt closest in
# shape (one user turn asking for worked reasoning); the warm-up and the hard LiveCodeBench draws are unchanged.
PUBLIC_SPEED_SAMPLE = {'gpqa_diamond': SPEED_SAMPLE['gpqa_diamond'] + SPEED_SAMPLE['math_500'],
                       'livecodebench_v6': SPEED_SAMPLE['livecodebench_v6']}
# A self-check retry starts only if it can end within this many request timeouts of the request's first attempt, so a
# server that fails slowly cannot hold the self-check's GPU past its budget.
SELFCHECK_RETRY_WINDOW = 1.5


def _public_rows(public) -> tuple[list[dict], list[dict]]:
    """GPQA questions and hard LiveCodeBench problems from a self-check's fetched public folder."""
    from ..validation import livecodebench
    from ..validation.language import PUBLIC_FILES, load_gpqa as load_public_gpqa
    root = Path(public) / 'benchmarks'
    problems = livecodebench.iter_problems(root / PUBLIC_FILES['livecodebench_v6'], SELFCHECK_LCB_FILES)
    return (load_public_gpqa(root / PUBLIC_FILES['gpqa_diamond']),
            [row for row in problems if row.get('difficulty') == 'hard'])


def public_speed_prompts(config: dict, public, *, rng=None) -> list[list[dict]]:
    """speed_prompts on public inputs (PUBLIC_SPEED_SAMPLE): the same count, construction and shuffle."""
    gpqa, problems = _public_rows(public)
    return _speed_draw(config, gpqa, (), problems, PUBLIC_SPEED_SAMPLE, rng or random.SystemRandom())


def public_requests(config: dict, public, *, gpqa: int, livecodebench: int = 0, rng=None) -> list[dict]:
    """Scored chat requests on public inputs exactly as the suite's scorers send them under `config` (messages,
    sampling settings, per-request timeout): `gpqa` GPQA questions, then `livecodebench` hard LiveCodeBench problems,
    each drawn without replacement. Items are {'benchmark', 'payload', 'timeout'}."""
    from .livecodebench import _evaluate_rows
    rng = rng or random.SystemRandom()
    gpqa_rows, problems = _public_rows(public)
    settings = {**_generation_settings(config), 'concurrency': 1}
    recorder = _RecordingTransport()
    if gpqa:
        evaluate_gpqa(recorder, _speed_pick('gpqa_diamond', gpqa_rows, gpqa, rng), **settings)
    if livecodebench:
        _evaluate_rows(recorder, _speed_pick('livecodebench_v6', problems, livecodebench, rng), None,
                       {**_code_settings(config, settings), 'generation_concurrency': 1})
    return [{'benchmark': 'gpqa_diamond' if index < gpqa else 'livecodebench_v6', **request}
            for index, request in enumerate(recorder.requests)]


def parse_reply(request: dict, response) -> dict:
    """What the scorers read from one reply to a public_requests item: ValueError where grading counts it as a failed
    request (ResilientTransport's malformed-response rule), else the finish reason, the extracted GPQA letter or the
    size of the LiveCodeBench code block, and the end of the reply. Never a score."""
    from ..validation.livecodebench import extract_code
    from ..validation.text_benchmarks import completion_content
    problem = _malformed(response)
    if problem is not None:
        raise ValueError(problem)
    choice = response['choices'][0]
    text = completion_content(response)
    reply = {'finish_reason': choice.get('finish_reason')}
    if choice.get('finish_reason') == 'length':
        reply['note'] = 'cut off at max_tokens: grading scores such a reply zero'
    elif request['benchmark'] == 'gpqa_diamond':
        reply['answer'] = extract_choice(text, 'ABCD')
        if reply['answer'] is None:
            reply['note'] = 'no answer letter found (ANSWER: X on its own line): grading scores it wrong'
    else:
        reply['code_characters'] = len(extract_code(text))
        if not reply['code_characters']:
            reply['note'] = 'no fenced code block found: grading scores it wrong'
    message = choice.get('message')
    reasoning = message.get('reasoning_content') if isinstance(message, dict) else None
    reply['content_characters'] = len(text)
    if isinstance(reasoning, str):
        reply['reasoning_characters'] = len(reasoning)
    reply['content_end'] = text[-120:]
    return reply


def _send(transport, payload, timeout, *, retry_window=None, stop=None, clock=time.monotonic, sleep=None):
    """One chat request as ResilientTransport sends it: a transport failure or malformed reply is re-sent after
    RETRY_BACKOFF_SECONDS (REQUEST_ATTEMPTS in all), a timeout never, and a retry starts only if it can end within
    `retry_window` seconds of the first attempt and `stop` (a threading.Event) is not set. Returns (the last reply
    received or None, a record: attempts, seconds of the last attempt on the host clock, and finish_reason or error,
    with timed_out for a timeout and retry_refused when the window ruled a retry out).
    EvaluationError, the grader's own transport failing, propagates."""
    sleep = sleep or time.sleep
    started, record, errors, response = clock(), {}, [], None
    for attempt in range(REQUEST_ATTEMPTS):
        if attempt:
            backoff = RETRY_BACKOFF_SECONDS[attempt - 1]
            if retry_window is not None and clock() + backoff + timeout > started + retry_window:
                errors.append('no time left in the self-check for a retry')
                record['retry_refused'] = True
                break
            sleep(backoff)
            if stop is not None and stop.is_set():
                errors.append('the self-check stopped before the retry')
                break
        begun = clock()
        record['attempts'] = attempt + 1
        try:
            reply = transport.request('POST', '/v1/chat/completions', payload, timeout=timeout)
        except EvaluationError:
            raise
        except Exception as error:  # noqa: BLE001 - the request layer, as ResilientTransport sees it
            record['seconds'] = clock() - begun
            text = f'{type(error).__name__}: {str(error)[:200]}'
            if _is_timeout(error):
                return response, {**record, 'timed_out': True, 'error': text}
            errors.append(text)
            continue
        record['seconds'], response = clock() - begun, reply
        problem = _malformed(reply)
        if problem is None:
            return reply, {**record, 'finish_reason': reply['choices'][0].get('finish_reason')}
        errors.append(f'malformed response: {problem}')
    return response, {**record, 'error': ' | '.join(errors)}


class _Unanswered(dict):
    """The sample's _send record when grading would have scored it wrong and gone on: it timed out, or the
    self-check's window ruled out a retry grading would have made."""


class _Resilient:
    """The self-check sample's transport: _send's retries; the last reply received goes to the parser (which rejects a
    malformed one with its text), a request that timed out or could not be retried yields _Unanswered, and one that
    failed every attempt raises its errors."""

    def __init__(self, transport, retry_window):
        self.transport, self.retry_window = transport, retry_window

    def request(self, method, path, payload=None, *, timeout, **kwargs):
        response, record = _send(self.transport, payload, timeout, retry_window=self.retry_window)
        if record.get('timed_out') or record.get('retry_refused'):
            return _Unanswered(record)
        if response is None:
            raise RuntimeError(record['error'])
        return response


def measure_concurrent(transport, requests: list[dict], *, concurrency: int, retry_window: float | None = None,
                       clock=time.monotonic, sleep=None) -> list[dict]:
    """The suite's request pattern on public_requests items: up to `concurrency` in flight, each with its own timeout
    and ResilientTransport's retries (_send). One record per request, with its benchmark.

    Interrupted (selfcheck.main's SIGTERM raises SystemExit) or failed, it raises at once rather than wait for the
    requests in flight as text_benchmarks._run does: the self-check must leave its launch context within the plugin's
    100 s grace to stop the H100, and stopping the server fails those requests. None is retried after that."""
    from concurrent.futures import ThreadPoolExecutor
    stop = threading.Event()

    def send(request):
        _, record = _send(transport, request['payload'], request['timeout'], retry_window=retry_window, stop=stop,
                          clock=clock, sleep=sleep)
        return {'benchmark': request['benchmark'], **record}

    pool = ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(requests))))
    try:
        futures = [pool.submit(send, request) for request in requests]
        records = [future.result() for future in futures]
    except BaseException:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    return records


def selfcheck_inputs(config: dict, public, *, wave: bool = True, rng=None) -> list[dict]:
    """What selfcheck_requests sends, built before the GPU is up: one GPQA sample, then with `wave` one request per
    concurrent slot (generation_concurrency), half GPQA and half hard LiveCodeBench."""
    size = _generation_settings(config)['concurrency'] if wave else 0
    return public_requests(config, public, gpqa=1 + size - size // 2, livecodebench=size // 2, rng=rng)


def selfcheck_requests(check, server, requests: list[dict], config: dict, *, sample_timeout: float | None = None):
    """A language task's self-check requests on a launched copy (frontier_ml.selfcheck.SelfCheck): the first of
    `requests` (selfcheck_inputs) end to end and read by the scorers' rules (parse_reply), then the rest at once as
    the suite's per-request timeout under its concurrency. Grading sends every benchmark that way for hours, scores a
    request that times out or still fails after its retries as wrong, and stops with zero once more than
    failed_request_limit (FAILED_REQUEST_PERCENT by default) of a benchmark has; the wave is its first minutes, held to
    that share of the wave. Without a limit, failed wave requests warn instead, since grading only scores them wrong. A
    sample that timed out, or that the self-check had no time to retry, passes with a note, since grading only scores it
    wrong. `sample_timeout` caps the sample when grading's timeout exceeds the self-check's budget."""
    settings = _generation_settings(config)
    limit_percent = failed_request_limit(config)
    sample, wave = requests[0], requests[1:]
    timeout = min(sample['timeout'], sample_timeout or sample['timeout'])
    name = 'GPQA request as graded'
    if timeout < sample['timeout']:
        name += f" (self-check cap {timeout:.0f} s; grading allows {sample['timeout']:g} s)"

    def parse(request, response):
        if not isinstance(response, _Unanswered):
            return parse_reply(request, response)
        if not response.get('timed_out'):
            note = ('grading would retry it, but the self-check had no time left; grading scores a request that still '
                    f'fails as wrong ({_limit_clause(limit_percent)})')
        elif timeout < request['timeout']:
            note = (f"no reply within the self-check's {timeout:.0f} s cap; grading allows {request['timeout']:g} s, "
                    'so this was not measured')
        else:
            note = (f'no reply within {timeout:g} s: grading scores a timed-out request as wrong '
                    f'({_limit_clause(limit_percent)})')
        return {'note': note, 'attempts': response['attempts'], 'error': response['error']}

    # _Resilient retries as ResilientTransport does (a full timeout per attempt, malformed replies re-sent), which
    # check.request's shared policy does not; on top of it that policy would send a failing sample nine times.
    resilient = {**server, 'transport': _Resilient(server['transport'], SELFCHECK_RETRY_WINDOW * timeout)}
    check.request(resilient, '/v1/chat/completions', sample['payload'], name=name, parse=parse, sample=sample,
                  timeout=timeout, retry=False)
    if not wave:
        return
    check.stage('gates')
    check.note(f"One wave of {len(wave)} concurrent public requests stood for the suite's "
               f"{sum(REQUEST_COUNTS.values()):,} graded ones at that concurrency; MATH-500 and MMLU-Pro are held out.")
    limit = settings['timeout']
    records = measure_concurrent(server['transport'], wave, concurrency=len(wave),
                                 retry_window=SELFCHECK_RETRY_WINDOW * limit)
    seconds = sorted(record['seconds'] for record in records)
    timed_out = sum(bool(record.get('timed_out')) for record in records)
    failed = [record for record in records if record.get('error') and not record.get('timed_out')]
    # Grading's limit on requests lost to timeouts and failures, as a share of the wave: 0 of 16, 1 of 32 (all without one).
    allowed = limit_percent * len(wave) // 100 if limit_percent is not None else len(wave)
    kinds = {kind: sum(record['benchmark'] == kind for record in records)
             for kind in ('gpqa_diamond', 'livecodebench_v6')}
    check.record(f'slowest of {len(wave)} concurrent requests', timed_out <= allowed, measured=round(seconds[-1], 1),
                 required=limit, unit='s', higher_is_better=False, detail=(
                     f"{len(wave)} public requests at once ({kinds['gpqa_diamond']} GPQA, {kinds['livecodebench_v6']} "
                     f"hard LiveCodeBench) as graded (max_tokens {settings['max_tokens']}, temperature "
                     f"{settings['temperature']:g}, {limit:g} s each): slowest {seconds[-1]:.1f} s, median "
                     f"{seconds[len(seconds) // 2]:.1f} s; {sum(r.get('finish_reason') == 'length' for r in records)} "
                     f"stopped at max_tokens, {timed_out} timed out. Grading sends every benchmark {len(wave)} at a "
                     f"time, scores a request over {limit:g} s or still failing as wrong "
                     + (f"and stops with zero above {limit_percent}% of a benchmark, which here allows {allowed} of "
                        f"{len(wave)}" if limit_percent is not None else "with no limit on how many")
                     + "; held-out requests may run longer."))
    errors = ' || '.join(record['error'] for record in failed[:3])
    if failed and timed_out + len(failed) > allowed:
        check.fail('concurrent requests answered', f'{len(failed)} of {len(wave)} requests failed after up to '
                   f'{REQUEST_ATTEMPTS} attempts' + (f' and {timed_out} timed out' if timed_out else '')
                   + f', more than the {allowed} that grading\'s {limit_percent}% allows of {len(wave)} '
                   f'(it scores each as wrong): {errors}', kind='smoke')
    lost = timed_out + len(failed)
    if limit_percent is None and lost:
        check.warn('concurrent requests answered', (
            f'{len(wave) - lost} well-formed replies' + (f'; {timed_out} timed out after {limit:g} s' if timed_out else '')
            + (f'; {len(failed)} failed after up to {REQUEST_ATTEMPTS} attempts: {errors}' if failed else '')
            + f'. Grading scores each such request as wrong, so {lost} of {len(wave)} here would score zero.'))
        return
    check.record('concurrent requests answered', True, kind='smoke', detail=(
        f'{len(wave) - lost} well-formed replies' + (f'; {timed_out} timed out' if timed_out else '')
        + (f'; {len(failed)} failed after up to {REQUEST_ATTEMPTS} attempts, within grading\'s '
           f'{limit_percent}% but scored as wrong: {errors}' if failed else '')))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepare exact pinned language task dependencies')
    parser.add_argument('source', choices=['gpqa', 'reasoning-data', 'qwen', 'diffusion-base'])
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    if args.source == 'gpqa':
        print(prepare_gpqa(args.destination, os.environ.get('HF_TOKEN')))
    elif args.source == 'qwen':
        print(fetch_qwen(args.destination))
    elif args.source == 'diffusion-base':
        print(fetch_diffusion_base(args.destination))
    else:
        from .benchmark_data import prepare
        print(json.dumps(prepare(args.destination), indent=2))
