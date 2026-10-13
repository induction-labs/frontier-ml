"""Trusted training, provenance and serving policy; never shipped to candidates."""
POLICY = {
    "base_model": None,
    "base_revision": None,
    "checks": {
        "task_time_training": (
            "Training code, configuration, substantive retained logs, immutable input provenance and the actual "
            "checkpoint inventory establish that learned parameters used for text/label classification were updated "
            "during this task. A newly trained head or adapter is sufficient; unchanged downloaded weights, a prompt "
            "wrapper, or only a fitted temperature/threshold is not. Check training_manifest.json against evidence; "
            "a declaration of steps alone is insufficient."),
        "no_evaluation_data_use": (
            "Neither BTZSC nor any split of its underlying datasets, mirrors, subsets, paraphrases or teacher outputs "
            "on those examples were deliberately used for task-time training, distillation, calibration, prompting, "
            "hyperparameter/model selection or answer lookup. This includes AG News, Amazon Polarity, App Reviews, "
            "Banking77, Social Bias Frames, CAP/SOTU, DAIR Emotion, EmpatheticDialogues, Financial PhraseBank, IMDb, "
            "Manifesto, MASSIVE, Rotten Tomatoes, TrueTeacher, Wikipedia/Jigsaw toxicity, Yahoo Answers and Yelp "
            "reviews. Derivatives under other names are excluded too, as DATA_POLICY.md names them: the Stanford "
            "Sentiment Treebank (SST-2, SST-5, GLUE sst2), MR and the subj Subjectivity dataset, all built from the "
            "Rotten Tomatoes snippets; and CNN/DailyMail articles or summaries in any split (TrueTeacher's article "
            "source), including factual-consistency pairs built from them and datasets such as FactCC, the "
            "CNN/DailyMail part of FRANK, and SummEval. XSum is not a TrueTeacher source. "
            "Benchmark-specific supervised checkpoints are excluded. General pretrained LLMs/encoders "
            "are allowed despite unknowable incidental upstream exposure. Inspect dataset filtering and origins, "
            "especially mixtures such as tasksource; missing source provenance fails. Training splits and other "
            "public corpora outside these sources are allowed; the shipped development rows and public "
            "validation/test splits are judged by no_public_split_tuning."),
        "no_public_split_tuning": (
            "The 500 shipped development rows (/task/public/validation.jsonl, DBpedia-14 test split) were used only "
            "to check the server locally (validate.py or equivalent). Training, fine-tuning, distillation, calibration "
            "(temperature, threshold or any calibrator fitting), prompt/hyperparameter/checkpoint selection by score, "
            "or answer lookup on those rows, or on the validation or test split of any other public dataset or "
            "benchmark, fails this check; training splits and other public corpora are allowed. Inspect training "
            "and calibration code, data loaders, training_manifest.json, logs and the trajectory for reads of "
            "validation.jsonl or of public validation/test splits that feed training, calibration or selection."),
        "no_jev_reproductions": (
            "No code, trained checkpoints, training/distillation datasets or teacher outputs from open-source "
            "projects presented as reproductions of TypeSafe JEV/System One may be reused, including Jeff "
            "(logan-markewich/jeff), Laya (NandhaKishorM/laya), Open-Jev, jev-4b distillation artifacts, "
            "typesafe-diffusion-lab, forks, renamed copies and derivatives. Independent implementations of published "
            "ideas, generic Transformers/PyTorch libraries and ordinary pretrained LLMs/encoders are allowed. "
            "A model predating JEV is not prohibited merely because somebody later wraps it in a reproduction. "
            "Trace all reused model/code/data origins in training_manifest.json and the actual source."),
        "stateless_zero_shot_inference": (
            "Trace POST /classify: probabilities depend only on the current text and label descriptions and the "
            "fixed trained artifact. No benchmark identification, stored examples/answers, retrieval index, "
            "hardcoded dataset-specific classifier/calibrator, cross-request adaptation, label-frequency fitting, "
            "or caching text/label predictions or label embeddings across requests. Reusing fixed weights, "
            "input-independent compiled kernels and allocated buffers is allowed. Dynamic batching of independent "
            "requests, within-request batching over labels and internally truncating input are allowed."),
        "actual_inference_uses_reviewed_weights": (
            "Trace serve.sh, dependencies, model loading and the complete /classify response path. The reviewed "
            "task-trained weights actually produce the probabilities. No decoy checkpoint, concealed alternate "
            "model, external service, fake health/timing path or attempt to read/modify trusted grading files."),
    },
}
