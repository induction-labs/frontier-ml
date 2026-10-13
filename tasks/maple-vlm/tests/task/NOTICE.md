# Vendored scoring provenance

`mmmu_open_answer_eval.py` originates from MMMU-Benchmark/MMMU, commit
`268471d0d488258990025331c7528359c324aa25`, `mmmu/utils/eval_utils.py`.
Upstream: https://github.com/MMMU-Benchmark/MMMU.
The upstream Apache-2.0 license text is in `frontier_ml/licenses/Apache-2.0.txt`.
Changes: replace NumPy's argmax with equivalent standard-library first-maximum
selection and remove the global random seed side effect.

Since 2026-09-30 (review F3) the module is not called by any scorer or test. This
benchmark's scorer is gpt-6-luna as a judge: for every question, multiple-choice and
open, it reads the question, the answer key and the candidate's complete reply and
returns correct or incorrect (`answers.py`, the task's own implementation, shipped to
the agent and cited by the contract). No option-letter parsing, number extraction or
string comparison happens in code. The upstream open-answer matcher accepted a reply
that merely contained the reference ("Not Paris" for Paris) or any one of several
numbers in it, and its multiple-choice parser fell back to a random option; the copy
is kept for reference only.
