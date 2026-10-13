"""Operator-only GPU tooling for ai-text-detection baselines (never shipped). Requires torch and transformers.

    python tasks/ai-text-detection/ai_text_detection_models.py zeroshot --rows ROWS.jsonl --out-dir SCORES
    python tasks/ai-text-detection/ai_text_detection_models.py finetune --train TRAIN.jsonl --rows ROWS.jsonl --out SCORES.json
    python tasks/ai-text-detection/ai_text_detection_models.py pretrained --model HF_ID --rows ROWS.jsonl --out SCORES.json [--ai-label N]
    python tasks/ai-text-detection/ai_text_detection_models.py serve --kind classifier|llm --port P ...

ROWS/TRAIN rows: {"id", "text"} (+ "label" for TRAIN). Score files: {"name", "scores": {id: float}}, higher = more AI.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time


def _rows(path):
    return [json.loads(line) for line in open(path)]


def zeroshot(rows_path, out_dir, *, observer="Qwen/Qwen3-4B-Base", performer="Qwen/Qwen3-4B", batch=8, max_len=1024):
    """Binoculars, Fast-DetectGPT (analytic) and log-likelihood over the whole passage."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = _rows(rows_path)
    tok = AutoTokenizer.from_pretrained(observer)
    obs = AutoModelForCausalLM.from_pretrained(observer, dtype=torch.bfloat16).cuda().eval()
    perf = AutoModelForCausalLM.from_pretrained(performer, dtype=torch.bfloat16).cuda().eval()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    encoded = [[bos] + tok(r["text"], add_special_tokens=False)["input_ids"][: max_len - 1] for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(encoded[i]))
    result = {m: {} for m in ("binoculars", "fastdetectgpt", "loglik")}
    started = time.time()
    with torch.no_grad():
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            length = max(len(encoded[i]) for i in idx)
            ids = torch.full((len(idx), length), tok.pad_token_id or 0)
            mask = torch.zeros((len(idx), length), dtype=torch.long)
            for j, i in enumerate(idx):
                ids[j, :len(encoded[i])] = torch.tensor(encoded[i])
                mask[j, :len(encoded[i])] = 1
            ids, mask = ids.cuda(), mask.cuda()
            logp = torch.log_softmax(obs(input_ids=ids, attention_mask=mask).logits[:, :-1].float(), -1)
            q = torch.softmax(perf(input_ids=ids, attention_mask=mask).logits[:, :-1].float(), -1)
            y, t = ids[:, 1:], mask[:, 1:].bool()
            ll = logp.gather(-1, y[..., None])[..., 0]
            mean_ref = (q * logp).sum(-1)
            var_ref = (q * logp.pow(2)).sum(-1) - mean_ref.pow(2)
            n = t.sum(-1).clamp(min=1)
            nll = -(ll * t).sum(-1) / n
            xppl = (-mean_ref * t).sum(-1) / n
            fd = ((ll - mean_ref) * t).sum(-1) / (var_ref * t).sum(-1).clamp(min=1e-6).sqrt()
            for j, i in enumerate(idx):
                rid = rows[i]["id"]
                result["binoculars"][rid] = float(-(nll[j] / xppl[j]))
                result["fastdetectgpt"][rid] = float(fd[j])
                result["loglik"][rid] = float(-nll[j])
            del logp, q
    rate = len(rows) / (time.time() - started)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for method, scores in result.items():
        (out / f"{method}.json").write_text(json.dumps({"name": f"{method} ({observer} / {performer})", "scores": scores, "messages_per_second": rate}))


def finetune(train_path, rows_path, out, *, model_name="FacebookAI/xlm-roberta-base", max_len=768, epochs=2.0, lr=3e-5, batch=32):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    train, rows = _rows(train_path), _rows(rows_path)
    humans = sum(r["label"] == 0 for r in train)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).cuda()
    enc = lambda rs: tok([r["text"] for r in rs], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    weight = torch.tensor([(len(train) - humans) / humans, 1.0]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = math.ceil(len(train) / batch * epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / (0.06 * steps)) * max(0.0, (steps - s) / steps))
    rng, step, started = random.Random(0), 0, time.time()
    model.train()
    while step < steps:
        order = list(range(len(train)))
        rng.shuffle(order)
        for b in range(0, len(order), batch):
            chunk = [train[i] for i in order[b:b + batch]]
            x = {k: v.cuda() for k, v in enc(chunk).items()}
            y = torch.tensor([r["label"] for r in chunk]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**x).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), y, weight=weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 200 == 0:
                print(f"step {step}/{steps} loss {loss.item():.4f} {time.time() - started:.0f}s", flush=True)
            if step >= steps:
                break
    train_seconds = time.time() - started
    scores, rate = _classify(model, tok, rows, max_len, lambda logits: (logits[:, 1] - logits[:, 0]).tolist())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"name": f"{Path(model_name).name} fine-tune", "scores": scores, "messages_per_second": rate,
                                     "train_seconds": train_seconds, "train_rows": len(train)}))
    save = Path(out).with_suffix("")
    model.save_pretrained(save)
    tok.save_pretrained(save)
    print("scored", len(rows), f"{rate:.0f} msg/s; model saved to {save}", flush=True)


def _classify(model, tok, rows, max_len, head):
    import torch
    model.eval()
    scores, started = {}, time.time()
    with torch.no_grad():
        for b in range(0, len(rows), 64):
            chunk = rows[b:b + 64]
            x = {k: v.cuda() for k, v in tok([r["text"] for r in chunk], truncation=True, max_length=max_len, padding=True, return_tensors="pt").items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**x).logits.float()
            for r, s in zip(chunk, head(logits)):
                scores[r["id"]] = s
    return scores, len(rows) / (time.time() - started)


def pretrained(model_name, rows_path, out, *, ai_label=None, max_len=512):
    """An off-the-shelf open detector: the score is the logit margin (or single logit) of its AI class."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if "desklib" in model_name:
        # Desklib's published architecture: DeBERTa-v3 encoder, mean pooling, one linear logit (AI).
        from transformers import AutoConfig, AutoModel, PreTrainedModel

        class DesklibAIDetectionModel(PreTrainedModel):
            config_class = AutoConfig

            def __init__(self, config):
                super().__init__(config)
                self.model = AutoModel.from_config(config)
                self.classifier = torch.nn.Linear(config.hidden_size, 1)
                self.post_init()

            def forward(self, input_ids=None, attention_mask=None, **_):
                hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                return type("Out", (), {"logits": self.classifier(pooled)})()

        model = DesklibAIDetectionModel.from_pretrained(model_name).to(torch.bfloat16).cuda()
        labels = {0: "AI"}
        ai_label = 0
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=torch.bfloat16).cuda()
        labels = {int(k): v for k, v in model.config.id2label.items()}
    if ai_label is None:
        ai_label = next((k for k, v in labels.items() if any(w in v.lower() for w in ("ai", "machine", "fake", "generated", "gpt"))), 1)

    def head(logits):
        if logits.shape[1] == 1:
            return logits[:, 0].tolist()
        return (logits[:, ai_label] - torch.logsumexp(torch.cat([logits[:, :ai_label], logits[:, ai_label + 1:]], 1), 1)).tolist()

    scores, rate = _classify(model, tok, _rows(rows_path), max_len, head)
    Path(out).write_text(json.dumps({"name": f"{model_name} (off the shelf, AI label {labels.get(ai_label, ai_label)})", "scores": scores,
                                     "messages_per_second": rate}))
    print("scored", len(scores), f"{rate:.0f} msg/s")


def serve(kind, port, *, model_dir=None, upstream=None, upstream_model=None, max_batch=64, wait_ms=4):
    """Contract-obeying /detect servers used to calibrate the throughput gate.
    classifier: a saved sequence classifier with dynamic micro-batching.
    llm: one forward pass of an LM per request through an OpenAI-compatible completions upstream (vLLM)."""
    import queue
    import threading
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    if kind == "classifier":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda().eval()
        jobs = queue.Queue()

        def worker():
            while True:
                batch = [jobs.get()]
                deadline = time.time() + wait_ms / 1000
                while len(batch) < max_batch and time.time() < deadline:
                    try:
                        batch.append(jobs.get(timeout=max(0.0, deadline - time.time())))
                    except queue.Empty:
                        break
                x = tok([b[0] for b in batch], truncation=True, max_length=768, padding=True, return_tensors="pt")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    probs = torch.softmax(model(**{k: v.cuda() for k, v in x.items()}).logits.float(), -1)[:, 1].tolist()
                for (_, slot), p in zip(batch, probs):
                    slot.append(p)
                    slot[0].set()

        threading.Thread(target=worker, daemon=True).start()

        def score(text):
            slot = [threading.Event()]
            jobs.put((text, slot))
            slot[0].wait()
            return slot[1]
    else:
        def score(text):
            prompt = "Decide whether the following text was written by a human or generated by an AI model.\n\n" + text + "\n\nAnswer (human or AI):"
            body = {"model": upstream_model, "prompt": prompt, "max_tokens": 1, "temperature": 0, "logprobs": 20}
            request = urllib.request.Request(upstream.rstrip("/") + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=600) as response:
                top = json.load(response)["choices"][0]["logprobs"]["top_logprobs"][0]
            p_ai = sum(math.exp(v) for k, v in top.items() if k.strip().lower() == "ai")
            p_h = sum(math.exp(v) for k, v in top.items() if k.strip().lower() in ("human", "hum"))
            return p_ai / (p_ai + p_h) if p_ai + p_h > 0 else 0.5

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200 if self.path == "/health" else 404, {})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.reply(200, {"p_ai": min(1.0, max(0.0, float(score(payload["text"]))))})

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 256

    Server(("127.0.0.1", port), Handler).serve_forever()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    z = sub.add_parser("zeroshot")
    z.add_argument("--rows", required=True)
    z.add_argument("--out-dir", required=True)
    f = sub.add_parser("finetune")
    f.add_argument("--train", required=True)
    f.add_argument("--rows", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    q = sub.add_parser("pretrained")
    q.add_argument("--model", required=True)
    q.add_argument("--rows", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--ai-label", type=int)
    v = sub.add_parser("serve")
    v.add_argument("--kind", choices=["classifier", "llm"], required=True)
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--model-dir")
    v.add_argument("--upstream")
    v.add_argument("--upstream-model")
    args = parser.parse_args(argv)
    if args.command == "zeroshot":
        zeroshot(args.rows, args.out_dir)
    elif args.command == "finetune":
        finetune(args.train, args.rows, args.out, model_name=args.model)
    elif args.command == "pretrained":
        pretrained(args.model, args.rows, args.out, ai_label=args.ai_label)
    else:
        serve(args.kind, args.port, model_dir=args.model_dir, upstream=args.upstream, upstream_model=args.upstream_model)


if __name__ == "__main__":
    main()
