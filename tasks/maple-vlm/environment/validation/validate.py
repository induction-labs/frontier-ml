#!/usr/bin/env python3
"""Evaluate your vision-language server on the public validation questions.

    python /task/validate.py --server-url http://127.0.0.1:8000 [--judge-base-url URL]

Sends each question to POST /v1/chat/completions with the images as base64 data URLs,
exactly as the trusted final client does. The score is accuracy, and every question is
decided by gpt-6-luna as a judge: it reads the question, the answer key and your complete
reply and returns correct or incorrect, for multiple-choice and open questions alike (the
rules it applies are the system prompt in answers.py). Nothing else parses your reply.
Pass --judge-base-url to run the same judge locally and get a score; without it this
script returns your raw replies and no score. This script measures quality only; the
requirements stated in /task/CONTRACT.md are enforced by the trusted grader.
"""
from concurrent.futures import ThreadPoolExecutor
from statistics import mean
import threading

from frontier_ml.validation.evaluation import chat_answer, image_url, run_http_evaluation
from frontier_ml.validation.errors import EvaluationError
from answers import JUDGE_MODEL, answer_key, judge_correct, question_options
from frontier_ml.validation.cli import public_main

TASK = "maple-vlm"
DEFAULT_DATA = "/task/public/validation.jsonl"
JUDGE_WORKERS = 16  # bounded pool of independent judge calls, one per answered question (900 in the held-out set)


def build_request(sample, data_root):
    question = sample["question"]
    options = question_options(sample)
    if options:
        question += "\n" + "\n".join(f"{chr(65 + i)}. {v}" for i, v in enumerate(options))
        question += "\nReturn only the correct option letter."
    else:
        question += "\nReturn only the answer."
    images = sample.get("images", {})
    refs = [images[key] for key in sorted(images)] if isinstance(images, dict) else images
    content = [{"type": "text", "text": question}]
    content.extend({"type": "image_url", "image_url": {"url": image_url(ref, data_root)}} for ref in refs)
    payload = {"model": "submission", "messages": [{"role": "user", "content": content}],
               "max_tokens": 256, "temperature": 0, "stream": False}
    return "/v1/chat/completions", payload


def parse_response(sample, response):
    return {"id": sample["id"], "answer": chat_answer(response)}


def failed_prediction(sample, failure):
    """A question whose request failed after the shared retries, timed out or returned a malformed completion
    (see CONTRACT.md) counts as an empty, incorrect answer, so the accuracy and its per-question records stay
    over every question."""
    return {"id": sample["id"], "answer": ""}


def score(samples, predictions, config=None):
    """Accuracy over every question, each decided by the judge in `config["judge"]` (answers.judge_correct) from
    the question, its answer key and the reply, whatever the question type. An empty reply, including the
    evaluator's failed_prediction, is wrong without a judge call. Every row's answer key is checked before the
    first paid call. Judge faults raise EvaluationError (no score imputed), never a wrong answer."""
    config = config or {}
    judge = config.get("judge")
    if not callable(judge):
        raise RuntimeError("The gpt-6-luna answer judge is the only scorer; configure it (validate.py --judge-base-url)")
    model = config.get("judge_model", JUDGE_MODEL)
    if not isinstance(model, str) or not model:
        raise ValueError("judge_model must identify the answer judge deployment")
    if not samples or len(samples) != len(predictions):
        raise ValueError("Expected exactly one prediction per question")
    records, jobs = [None] * len(samples), []
    for index, (row, prediction) in enumerate(zip(samples, predictions, strict=True)):
        if prediction.get("id") != row["id"] or not isinstance(prediction.get("answer"), str):
            raise ValueError("Invalid prediction ID or answer")
        answer_key(row)
        reply = prediction["answer"].strip()
        record = {"id": row["id"], "subject": row.get("subject"), "question_type": row["question_type"]}
        if reply:
            jobs.append((index, row, reply))
        else:
            record.update(correct=False, judged=False, reason="empty reply")
        records[index] = record

    abort = threading.Event()  # one question out of judge attempts stops the others before they spend more

    def judged(job):
        index, row, reply = job
        try:
            _, info = judge_correct(judge, model, row, reply, abort=abort)
        except EvaluationError:
            abort.set()
            raise
        return index, {**records[index], "judged": True, **info}

    if jobs:
        with ThreadPoolExecutor(max_workers=JUDGE_WORKERS, thread_name_prefix="maple-judge") as pool:
            for index, record in pool.map(judged, jobs):
                records[index] = record
    return {"score": mean(int(r["correct"]) for r in records),
            "details": {"samples": len(records), "metric": "accuracy", "answer_judge": model,
                        "judged": len(jobs), "open_questions": sum(row["question_type"] == "open" for row in samples),
                        "documents": records}}


def run(transport, data_path, config):
    config = dict(config)
    if config.get("validation_mode") is True and not callable(config.get("judge")):
        config["predictions_only"] = True  # the judge is the only scorer: without one, the raw replies are the report
    return run_http_evaluation(transport, data_path, config, build_request=build_request,
                               parse_response=parse_response, score=score, failed_prediction=failed_prediction)


def main(argv=None):
    return public_main(TASK, run, argv, default_data=DEFAULT_DATA, description=__doc__, judge=True,
                       default_judge_model=JUDGE_MODEL)


if __name__ == "__main__":
    raise SystemExit(main())
