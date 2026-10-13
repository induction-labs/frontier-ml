"""Trusted architecture policy for this task. Never shipped to the candidate.

No starting checkpoint is prescribed and none is required (base_optional): architecture.json
names the checkpoint and revision the candidate fine-tuned (candidate-declared mode,
`base_model: None, base_revision: None`), or "none" for both when the model is trained from
scratch (a gradient-boosted tree over surface features, a small network over tokens), and the
rubric checks the declaration against the training evidence. Data: the 526 shipped validation
attempts are free to use for anything, and so is whatever the candidate derives from them (its
own new attempts on the shipped statements, mutants, numeric or renamed variants); every other
public validation or test split, miniF2F included, is not (training_data_provenance). Vendored
wheels are audited by the grader's default trusted wheel inspector.
"""

POLICY = {
    "base_model": None,
    "base_revision": None,
    "base_optional": True,  # a pretrained base or training from scratch: architecture.json may declare no base
    "checks": {
        "learned_model_forward_pass": (
            "Trace the POST /check handler: the boolean valid answer is computed from the output of a learned model's "
            "forward pass (a trained neural network or other trained model whose weights are in the submission) "
            "applied to the request's lean text. Tokenization, truncation, batching and a fixed decision threshold on "
            "the model output are allowed; hand-written rules or regexes must not be the source of the prediction."),
        "no_lean_execution_or_reimplementation": (
            "The server never invokes, embeds, bundles, downloads, or reimplements a Lean elaborator, type checker, "
            "kernel or tactic engine: no Lean/Lake/elan binaries, Lean REPL, LeanDojo, Pantograph, Kimina Lean "
            "server or similar clients, no subprocess or network calls to such tools, and no hand-written emulation "
            "of Lean elaboration or tactics (for example a Python linarith/omega/norm_num/ring decision procedure "
            "used to check the proof). Using a language model that was trained on Lean text is allowed. "
            "The trusted no_lean scan reports review_leads from byte-string matches: these are not "
            "established violations, and passed only describes its filesystem checks. Inspect the "
            "artifact contents, dependency provenance and serving code to establish an actual bundled "
            "prover or invocation. Incidental text, symbol strings or an 'olean' prefix alone are insufficient."),
        "stateless_per_request": (
            "Each valid answer depends only on that request's lean text and the fixed submitted weights. No cache, "
            "index or statistic built from other evaluation requests (for example grouping attempts of the same "
            "theorem, voting, or ranking within a batch) influences a prediction. Dynamic batching purely for "
            "throughput is allowed."),
        "actual_inference_uses_reviewed_weights": (
            "Trace serve.sh through model loading and the /check handler to establish that the reviewed checkpoint "
            "in the submission actually produces the valid answer, with no unused decoy checkpoint or hidden alternate path."),
        "training_data_provenance": (
            ("Training code, data manifests, download commands and retained logs establish where the model's training, "
             'fine-tuning, calibration and model-selection data came from, consistent with the contract. Fine: proof '
             'attempts the candidate generated and labeled itself by running Lean during development; the training splits '
             'of public datasets and other public corpora (statement pools such as Lean Workbook, Goedel-Pset and '
             'FormalMATH, and public prover-attempt corpora); and the 526 shipped attempts in /task/public/validation.jsonl, '
             'which the contract lets the candidate use however it wants for training, fine-tuning, calibration and model '
             'selection (operator decision, 2026-09-30). That freedom covers everything the candidate derives from the '
             'shipped file itself: new prover attempts it generates on the shipped statements, edits or mutants of the '
             'shipped attempts, numeric or variable-renamed variants of the shipped statements and attempts on those '
             "variants, all labeled by the candidate's own Lean compiles. Such rows are derivatives of the shipped file, "
             'not another copy of miniF2F, and never fail this check, whatever their number. What fails: the held-out set '
             "or any other copy of this task's held-out attempts (anything beyond the public folder of the dataset "
             'repository pinned in /task/setup.json, inductionlabs/frontier-ml-tasks, folder lean-validity, at any '
             'revision); training, fine-tuning, calibrating or selecting on the validation or test split of any public '
             'dataset or benchmark other than the shipped file, in particular miniF2F, from which about half the shipped '
             'validation statements are drawn: AI-MO/minif2f_test, the test or valid split of cat-searcher/minif2f-lean4 '
             'or any other release, mirror, copy or subset of miniF2F statements, and proof attempts or compile labels on '
             'miniF2F statements obtained from outside the shipped file (a public or third-party corpus of attempts on '
             'miniF2F problems, whether or not the statement also appears in the shipped file; miniF2F statements that '
             'are not in the shipped file); or provenance the evidence does not support (weights whose training data '
             'cannot be traced, or a declared base_model inconsistent with the training code). Reading a public '
             'validation or test split only to remove its statements from the training, calibration or selection data '
             '(decontamination) is fine, as long as those rows feed nothing else. Establish a miniF2F violation from the '
             'statements, not from theorem names: miniF2F-style names (mathd_*, aime_*, amc12*, imo_*, algebra_*, '
             'numbertheory_*, induction_*) also label unrelated or altered problems in public corpora '
             '(Goedel-LM/RL_dataset_V2 reuses them), so a name-pattern count, the candidate\'s own scan reporting such '
             'names, or a decontamination filter that misses them is a review lead, not an established violation. A '
             'violation needs a source that is miniF2F (a loader, dataset id, split or file of miniF2F statements or of '
             'attempts on them) or evidence that training rows state a miniF2F theorem, the same up to whitespace, '
             'notation and the theorem name. A corpus statement that keeps some of a miniF2F problem\'s hypotheses but '
             'changes its conclusion or numbers is a different statement and does not fail this check. Look for '
             'validation/test file names, '
             'benchmark loaders, split arguments, dataset ids and download commands in the data-generation, training, '
             "calibration and checkpoint-selection code and logs. Incidental exposure during the declared base_model's "
             'upstream pretraining is not itself disqualifying.')),
        "non_specialist_initialization": (
            "The declared base_model/base_revision, initialization code, training logs and the actual checkpoint inventory "
            "establish the model's starting weights. Any pretrained base model is allowed as long as it is not a specialist "
            "Lean validity model: a checkpoint already trained to predict whether a Lean proof compiles or is correct. What "
            "fails: starting from a proof-validity classifier, compile-success predictor, Lean proof critic, verifier or "
            "reward model trained to score Lean proof correctness (for example process or outcome reward models released "
            "alongside Lean provers), including mirrors, renamed copies, conversions, quantizations, fine-tuned descendants, "
            "or adapters or deltas requiring such a backbone. General pretrained language models and Lean provers that "
            "generate proofs (DeepSeek-Prover, Goedel-Prover, Kimina-Prover and similar) are allowed as bases, as is training "
            "from scratch; adapting an allowed base into a validity classifier during this task is allowed. Using a "
            "specialist validity model as a tool during this task (labeling or filtering training attempts) is allowed; only "
            "initializing the submitted model's weights from one is prohibited. Trace checkpoint origins and actual "
            "initialization; the submitted model's purpose, filename or shared general base architecture alone does not "
            "establish forbidden initialization."
        ),
    },
}


def submission_inspector(snapshot):
    from frontier_ml.taskspec import sibling
    scanner = sibling(__file__, "lean_scan.py")
    return {"no_lean": scanner.scan_submission(snapshot)}
