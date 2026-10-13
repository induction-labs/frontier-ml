"""Example draft server (operator fixture): 
- a small causal LM, int8 on disk, greedy 16-token drafts in batches.
"""
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import threading

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
DRAFT_TOKENS = 16
STOP = (151645, 151643)
PAD = 151643
MAX_BATCH = 32
GATHER_SECONDS = 0.004


def load_model():
    config = AutoConfig.from_pretrained(HERE / "drafter")
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    packed = load_file(HERE / "drafter/weights.safetensors")
    state = {}
    for name, tensor in packed.items():
        if name.endswith(".qweight"):
            base = name[:-len(".qweight")]
            state[base] = (tensor.to(torch.float32) * packed[base + ".scale"].to(torch.float32)).to(torch.bfloat16)
        elif not name.endswith(".scale"):
            state[name] = tensor
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or [m for m in missing if m != "lm_head.weight"]:
        raise RuntimeError(f"weights do not match the config: missing={missing} unexpected={unexpected}")
    model.tie_weights()
    return model.to("cuda").eval()


MODEL = load_model()
REQUESTS = queue.Queue()


@torch.inference_mode()
def draft_batch(prefixes):
    width = max(len(p) for p in prefixes)
    ids = torch.full((len(prefixes), width), PAD, dtype=torch.long)
    mask = torch.zeros((len(prefixes), width), dtype=torch.long)
    for row, prefix in enumerate(prefixes):
        ids[row, width - len(prefix):] = torch.tensor(prefix)
        mask[row, width - len(prefix):] = 1
    output = MODEL.generate(input_ids=ids.cuda(), attention_mask=mask.cuda(), max_new_tokens=DRAFT_TOKENS,
                            do_sample=False, pad_token_id=PAD, eos_token_id=list(STOP))
    drafts = []
    for row in output[:, width:].tolist():
        draft = []
        for token in row:
            draft.append(token)
            if token in STOP:
                break
        drafts.append(draft)
    return drafts


def worker():
    while True:
        batch = [REQUESTS.get()]
        try:
            while len(batch) < MAX_BATCH:
                batch.append(REQUESTS.get(timeout=GATHER_SECONDS))
        except queue.Empty:
            pass
        try:
            for (_, future), draft in zip(batch, draft_batch([tokens for tokens, _ in batch])):
                future.set_result(draft)
        except Exception as error:  # noqa: BLE001 - every waiting request gets the failure
            for _, future in batch:
                future.set_exception(error)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def respond(self, status, body):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self.respond(200 if self.path == "/health" else 404, {"ok": self.path == "/health"})

    def do_POST(self):
        if self.path != "/draft":
            return self.respond(404, {"error": "not found"})
        try:
            tokens = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["tokens"]
            future = Future()
            REQUESTS.put((tokens, future))
            self.respond(200, {"draft": future.result(timeout=120)})
        except Exception as error:  # noqa: BLE001
            self.respond(500, {"error": str(error)[:200]})


if __name__ == "__main__":
    draft_batch([[151644, 872, 198, 9707]])  # warm up before /health answers
    threading.Thread(target=worker, daemon=True).start()
    ThreadingHTTPServer.request_queue_size = 256
    ThreadingHTTPServer((os.environ.get("HOST", "0.0.0.0"), int(os.environ.get("PORT", "8000"))), Handler).serve_forever()
