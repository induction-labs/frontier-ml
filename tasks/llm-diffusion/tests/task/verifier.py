"""LLM to diffusion conversion of pinned MiniCPM5-2B: four-benchmark mean behind review and speed gates.

Official grading (verify_isolated) starts `inference_replicas` identical copies of the submission on separate H100s in
parallel, like cot-reconstruction's four reconstruction servers: the review runs once, the single-stream speed gate is
measured on the first copy, and the suite sends its requests round-robin over all copies at the contract's per-copy
concurrency (frontier_ml.scoring.language.ReplicaTransport). `verify` is the one-server path for operator tools."""
import math
import tempfile
from pathlib import Path
from frontier_ml.taskspec import sibling

from frontier_ml.runtime.server import VerificationFailure
from frontier_ml.scoring import language
from frontier_ml.scoring.sources import source_lock
from frontier_ml.validation.errors import EvaluationError

# The self-check's public files: the GPQA CSV and one LiveCodeBench file (language.SELFCHECK_PUBLIC), not 4.2 GB.
SELFCHECK_PUBLIC = language.SELFCHECK_PUBLIC


def default_config():
    """Build-time defaults; task.yaml verifier.config sets minimum_tokens_per_second (450), None forces a choice."""
    base = source_lock()['diffusion']
    return {'base_model': base['repo'], 'base_revision': base['revision'], 'minimum_tokens_per_second': None,
            'inference_replicas': 1}


def preflight(config, data_path):
    review = config.get('architecture_review_path')
    if not review or not Path(review).is_file():
        raise VerificationFailure('Trusted architecture_review_path is required before provisioning')
    minimum_tps = config.get('minimum_tokens_per_second')
    if not isinstance(minimum_tps, (float, int)) or not math.isfinite(minimum_tps) or minimum_tps <= 0:
        raise VerificationFailure('Diffusion minimum_tokens_per_second must be configured before provisioning')
    language.validate_suite_config(config, data_path)
    language.inference_replicas(config)
    # The speed gate sends scored requests drawn at random for this grade, the first as the untimed warm-up
    # (speed_prompts overrides the draw), and counts tokens with the base tokenizer, fetched here from the pinned
    # upstream commit, never from the submission.
    if config.get('speed_prompts') is None:
        config['speed_prompts'] = language.speed_prompts(config, data_path)
    _fetch_tokenizer(config)
    language.validate_diffusion_config(config)


def _fetch_tokenizer(config):
    """trusted_tokenizer_path unless the operator set it: the base tokenizer fetched from the pinned upstream commit."""
    if not config.get('trusted_tokenizer_path'):
        directory = tempfile.mkdtemp(prefix='frontier-tokenizer-')
        config['trusted_tokenizer_path'] = str(language.fetch_diffusion_tokenizer(directory))


def _tokenizer(config):
    """The speed gate's token counter, loaded from the verifier-fetched files only."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config['trusted_tokenizer_path'], local_files_only=True, trust_remote_code=False)


def _gates(server, data_path, config):
    """Review receipt and the single-stream speed gate on one server; (review, speed)."""
    base = language.validate_diffusion_config(config)
    review = language.verify_diffusion_review(server['snapshot'], config['architecture_review_path'], base,
                                             required_checks=list(sibling(__file__, 'review.py').POLICY['checks']))
    tokenizer = _tokenizer(config)
    prompts = config.get('speed_prompts') or language.speed_prompts(config, data_path)
    speed = language.measure_single_stream(server['transport'], tokenizer, prompts,
                                           minimum_tps=float(config['minimum_tokens_per_second']),
                                           temperature=language._generation_settings(config)['temperature'])
    return review, speed


def verify_isolated(launch, data_path, config):
    """inference_replicas copies of the submission on separate H100s, started in parallel; the speed gate on the first,
    the suite spread over all of them."""
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import ExitStack
    replicas = language.inference_replicas(config)
    with ExitStack() as stack:
        contexts = [launch() for _ in range(replicas)]
        with ThreadPoolExecutor(max_workers=replicas) as pool:
            futures = [pool.submit(context.__enter__) for context in contexts]
            servers, errors = [], []
            for context, future in zip(contexts, futures, strict=True):
                try:
                    servers.append(future.result())
                    stack.push(context.__exit__)
                except BaseException as error:  # noqa: BLE001 - stop the copies that did start, then report the first failure
                    errors.append(error)
        if errors:
            raise errors[0]
        review, speed = _gates(servers[0], data_path, config)
        replication = {'count': replicas, 'speed_gate_replica': 0,
                       'startup_seconds': [server.get('startup_elapsed') for server in servers],
                       'sandboxes': [server.get('sandbox_id') for server in servers]}
        if not speed['passed']:
            return {'score': 0.0, 'details': {'failure': 'throughput', 'speed': speed, 'replication': replication}}
        pooled = language.ReplicaTransport([server['transport'] for server in servers])
        result = language.evaluate_suite(pooled, data_path, config)
    result['details']['gates'] = {'architecture': review, 'speed': speed}
    result['details']['replication'] = {**result['details'].get('replication', {}), **replication}
    return result


def verify(server, data_path, config):
    """One-server path (operator tools and tests); official grading uses verify_isolated."""
    review, speed = _gates(server, data_path, config)
    if not speed['passed']:
        return {'score': 0.0, 'details': {'failure': 'throughput', 'speed': speed}}
    result = language.evaluate_suite(server['transport'], data_path, {**config, 'inference_replicas': 1})
    result['details']['gates'] = {'architecture': review, 'speed': speed}
    return result


class _Watched:
    """The grading transport as the self-check's speed gate sees it, keeping the first EvaluationError (the grader's
    own bridge failing), which measure_single_stream would otherwise record as the server's failed attempt. Every later
    request raises it at once, so the gate's remaining attempts do not wait on a broken bridge."""

    def __init__(self, transport):
        self.transport, self.error = transport, None
        if callable(getattr(transport, 'request_timed', None)):  # the sandbox clock, when the grading transport has one
            self.request_timed = lambda *args, **kwargs: self._call(transport.request_timed, *args, **kwargs)

    def _call(self, method, *args, **kwargs):
        if self.error is not None:
            raise self.error
        try:
            return method(*args, **kwargs)
        except EvaluationError as error:
            self.error = error
            raise

    def request(self, *args, **kwargs):
        return self._call(self.transport.request, *args, **kwargs)


class _Counted:
    """The pinned tokenizer as the speed gate uses it, keeping the first exception: measure_single_stream counts a
    reply's tokens outside its error handling, so a reply the tokenizer cannot encode (a lone UTF-16 surrogate, say)
    ends grading as a failed grade with score 0, the submission's fault."""

    def __init__(self, tokenizer):
        self.tokenizer, self.error = tokenizer, None

    def encode(self, text, **kwargs):
        try:
            return self.tokenizer.encode(text, **kwargs)
        except Exception as error:
            self.error = self.error or error
            raise


def _per_prompt(speed):
    """The best attempt of every measured prompt (a miss is re-sent once), in order."""
    prompts = []
    for record in speed['requests']:
        if record.get('warmup'):
            continue
        if record['attempt'] == 1:
            prompts.append(record)
        elif record['tokens_per_second'] > prompts[-1]['tokens_per_second']:
            prompts[-1] = record
    return prompts


# The self-check caps its sample at four times max_tokens at the gate's speed, never under this (the gate's own floor
# per request); grading allows request_timeout_seconds, an hour.
SAMPLE_CAP_FLOOR_SECONDS = 30.0


def selfcheck(check, public, config):
    """One copy, launched as grading launches each of its inference_replicas identical copies (whose speed gate is
    measured on one copy too), then the single-stream speed gate as graded on public prompts (nine requests of at
    most 30 s, run in full) and one public GPQA request read by the scorers' rules. No static constraint applies
    beyond the review prerequisites selfcheck.run checks first: preflight checks only the review receipt, the
    operator's settings and the private data. The suite's concurrent requests are not sent: at 3,600 s each they
    could hold the GPU for an hour."""
    settings = dict(config)
    _fetch_tokenizer(settings)
    base = language.validate_diffusion_config(settings)
    tokenizer = _Counted(_tokenizer(settings))
    generation = language._generation_settings(settings)
    minimum = float(settings['minimum_tokens_per_second'])
    prompts = language.public_speed_prompts(settings, public)
    requests = language.selfcheck_inputs(settings, public, wave=False)
    check.note(f"The suite's {generation['concurrency']} concurrent requests per copy, up to "
               f"{generation['timeout']:g} s each, were not sent; one request at its settings was.")
    with check.launch() as server:
        # Grading measures the gate first on a fresh copy, its warm-up absorbing first-request compilation.
        check.stage('gates')
        transport = _Watched(server['transport'])
        try:
            speed = language.measure_single_stream(transport, tokenizer, prompts, minimum_tps=minimum,
                                                   temperature=generation['temperature'])
        except Exception as error:
            if error is not tokenizer.error:
                raise
            check.fail('single-stream speed', f"the pinned {base['repo']} tokenizer could not count a reply "
                       f"({type(error).__name__}: {str(error)[:200]}); grading stops here with score 0",
                       required=settings['minimum_tokens_per_second'], unit='tokens/s')
        if transport.error is not None:
            raise transport.error
        measured = _per_prompt(speed)
        slowest = min(measured, key=lambda record: record['tokens_per_second'])
        problem = f"; {slowest['error']}" if slowest.get('error') else ''
        public_mix, graded = language.PUBLIC_SPEED_SAMPLE, language.SPEED_SAMPLE
        detail = (f"{len(measured)} of {len(prompts) - 1} measured prompts, sent one at a time after an untimed "
                  f"warm-up (max_tokens 1024, temperature {generation['temperature']:g}), each reply's reasoning and "
                  f"answer tokens counted with the pinned {base['repo']} tokenizer over end-to-end request time "
                  f"({speed['clock']} clock), a miss re-sent once; slowest {slowest['tokens_per_second']:.0f} tokens/s "
                  f"({slowest['tokens']} tokens in {slowest['seconds']:.2f} s{problem}); per prompt: "
                  + ', '.join(f"{record['tokens_per_second']:.0f}" for record in measured)
                  + f". Public prompts: {public_mix['gpqa_diamond'] + language.SPEED_WARMUP} GPQA and "
                  f"{public_mix['livecodebench_v6']} hard LiveCodeBench; grading draws "
                  f"{graded['gpqa_diamond'] + language.SPEED_WARMUP} GPQA, {graded['math_500']} level-5 MATH-500 "
                  f"(held out) and {graded['livecodebench_v6']} hard LiveCodeBench, so its replies differ.")
        measurement = {'measured': round(slowest['tokens_per_second'], 1),
                       'required': settings['minimum_tokens_per_second'], 'unit': 'tokens/s', 'higher_is_better': True}
        if not speed['passed']:  # grading stops at the first prompt that misses twice, with score 0
            check.fail('single-stream speed', detail + ' Grading stops here with score 0.', **measurement)
        check.record('single-stream speed', True, detail=detail, **measurement)
        check.stage('smoke')
        language.selfcheck_requests(check, server, requests, settings, sample_timeout=max(
            SAMPLE_CAP_FLOOR_SECONDS, 4 * generation['max_tokens'] / minimum))
