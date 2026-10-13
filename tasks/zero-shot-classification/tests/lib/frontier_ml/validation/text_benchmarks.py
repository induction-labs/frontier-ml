"""Chat completion requests, answer-letter extraction and bounded concurrency."""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

LETTERS = 'ABCD'


def completion_text(transport, prompt: str, *, max_tokens: int, timeout: float, temperature: float) -> str:
    response = transport.request('POST', '/v1/chat/completions', {
        'model': 'submission', 'messages': [{'role': 'user', 'content': prompt}],
        'temperature': temperature, 'max_tokens': max_tokens, 'stream': False,
    }, timeout=timeout)
    return completion_content(response)


def completion_content(response) -> str:
    """Read one completion; a length-truncated answer receives no credit."""
    choices = response.get('choices') if isinstance(response, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError('Expected exactly one model completion per question')
    # Even a correct answer before the cutoff is invalid. A reasoning model may
    # exhaust its budget before producing content, returning null instead.
    if choices[0].get('finish_reason') == 'length':
        return ''
    message = choices[0].get('message', {})
    text = message.get('content')
    if not isinstance(text, str) or message.get('tool_calls'):
        raise ValueError('Text benchmark requires a text completion without tools')
    return text


def extract_choice(text: str, valid_letters: str = LETTERS, marker_letters: str | None = None) -> str | None:
    """Letter from the last explicit answer marker (ANSWER: X, "the answer is X", boxed X), else the bare reply.

    Markers naming a letter from `marker_letters` (default: the valid letters) are candidates; the
    last one wins and must be a valid letter, so incidental capital letters in reasoning are never guessed.
    """
    letters = re.escape(marker_letters or valid_letters)
    candidates = []
    cleaned = text.replace('**', '')
    patterns = [rf'(?im)^\s*ANSWER:\s*\(?([{letters}])\)?[.\s]*$',
                rf'(?i)\bthe answer is\s*\(?([{letters}])\)?(?=[\s.,:;!?)]|$)',
                rf'\\boxed\{{\s*([{letters}])\s*\}}']
    for pattern in patterns:
        candidates.extend((match.start(), match.group(1).upper()) for match in re.finditer(pattern, cleaned))
    answer = max(candidates)[1] if candidates else cleaned.strip()
    return answer if len(answer) == 1 and answer in valid_letters else None


def _run(rows, score_one, concurrency: int) -> list[dict]:
    # Up to 8 server copies x 32 requests each (frontier_ml.scoring.language.MAX_REPLICAS); a single copy gets 32.
    if type(concurrency) is not int or not 1 <= concurrency <= 256:
        raise ValueError('Evaluation concurrency must be an integer from 1 to 256')
    if concurrency == 1:
        return [score_one(row) for row in rows]
    # map preserves input order. Exceptions fail the benchmark rather than
    # silently dropping unanswered examples from its denominator.
    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        result = list(pool.map(score_one, rows))
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return result
