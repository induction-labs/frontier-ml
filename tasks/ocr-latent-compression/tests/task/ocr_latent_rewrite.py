"""Paraphrase the fetched corpus with a local LLM so rendered text exists nowhere online.

The agent has Internet access during development. If page text were verbatim arXiv
text, a decoder fine-tuned on scraped recent abstracts could memorize held-out pages
and reconstruct them from a payload that merely identifies the paper. Every title,
abstract and full-text paragraph is therefore rewritten by a pinned open model with a
private sampling seed; only the rewritten text is rendered. Author names are
recombined at render time and are not rewritten.

    python tasks/ocr-latent-compression/ocr_latent_rewrite.py IN.jsonl OUT.jsonl --model DIR --seed HEX

Requires vLLM and a GPU; operator-only, never imported by the verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from ocr_latent_sources import clean, plain

PROMPTS = {
    "title": ("Rewrite this research paper title so it keeps the topic but uses different wording. "
              "Plain text only, no quotes, no markdown, no LaTeX, no symbols. Output only the new title.\n\nTitle: {text}"),
    "abstract": ("Rewrite the following research abstract in your own words. Keep the meaning, the technical register "
                 "and roughly the same length, but change the wording and sentence structure substantially. Use plain "
                 "ASCII prose only: no markdown, no LaTeX, no math symbols, no lists, no citations, no URLs. Output only "
                 "the rewritten abstract.\n\n{text}"),
    "paragraph": ("Rewrite the following paragraph from a research paper in your own words. Keep the meaning, the "
                  "technical register and roughly the same length, but change the wording and sentence structure "
                  "substantially. Use plain ASCII prose only: no markdown, no LaTeX, no math symbols, no lists, no "
                  "citations in brackets, no URLs. Output only the rewritten paragraph.\n\n{text}"),
    "heading": ("Rewrite this section heading from a research paper with different wording of similar length. Plain "
                "text only, no numbering, no markdown. Output only the heading.\n\nHeading: {text}"),
}


def units(records, max_papers, rng):
    """Texts to rewrite: every usable paper's title/abstract, plus full text where present."""
    usable = [r for r in records if r.get("title_plain") and r.get("abstract_plain") and len(r["abstract"]) >= 300]
    full = [r for r in usable if r.get("full_text")]
    rest = [r for r in usable if not r.get("full_text")]
    rng.shuffle(rest)
    chosen = full + rest[:max(0, max_papers - len(full))]
    out = []
    for record in chosen:
        out.append((record["arxiv_id"], "title", None, record["title"]))
        out.append((record["arxiv_id"], "abstract", None, record["abstract"]))
        for index, block in enumerate(record.get("full_text", [])):
            out.append((record["arxiv_id"], block["kind"], index, block["text"]))
    return chosen, out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", required=True, help="local directory of the pinned instruct model")
    parser.add_argument("--seed", required=True, help="private sampling seed (hex)")
    parser.add_argument("--max-papers", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args(argv)
    from vllm import LLM, SamplingParams
    records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    rng = random.Random(f"{args.seed}:select")
    chosen, work = units(records, args.max_papers, rng)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85, seed=0)
    tokenizer = llm.get_tokenizer()
    prompts, params = [], []
    for arxiv_id, kind, index, text in work:
        messages = [{"role": "user", "content": PROMPTS[kind].format(text=text)}]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        unit_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{arxiv_id}:{kind}:{index}".encode()).digest()[:4], "big")
        params.append(SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=1024, seed=unit_seed))
    outputs = llm.generate(prompts, params)
    rewritten = {}
    for (arxiv_id, kind, index, original), output in zip(work, outputs, strict=True):
        text = clean(output.outputs[0].text).strip('"')
        ok = plain(text) and 0.5 * len(original) <= len(text) <= 1.8 * len(original) + 20
        rewritten[(arxiv_id, kind, index)] = text if ok else None
    kept = 0
    with args.output.open("w") as stream:
        for record in chosen:
            title = rewritten.get((record["arxiv_id"], "title", None))
            abstract = rewritten.get((record["arxiv_id"], "abstract", None))
            if not title or not abstract:
                continue
            blocks = []
            for index, block in enumerate(record.get("full_text", [])):
                text = rewritten.get((record["arxiv_id"], block["kind"], index))
                if text:
                    blocks.append({"kind": block["kind"], "text": text})
            out = {"arxiv_id": record["arxiv_id"], "created": record["created"], "license": record["license"],
                   "open_license": record.get("open_license"), "authors": record["authors"],
                   "title": title, "abstract": abstract, "full_text": blocks,
                   "title_plain": True, "abstract_plain": True, "rewritten": True}
            stream.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1
    rejected = sum(v is None for v in rewritten.values())
    print(json.dumps({"papers": kept, "units": len(work), "rejected_units": rejected, "model": args.model,
                      "temperature": args.temperature}))


if __name__ == "__main__":
    main()
