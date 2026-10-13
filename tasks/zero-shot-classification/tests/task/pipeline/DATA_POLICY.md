# Training and data policy

BTZSC is reserved for final evaluation. Do not access or use its examples, labels,
prediction files or scores during task development. Do not train, distill,
calibrate, fit thresholds, select prompts, choose checkpoints or tune
hyperparameters on **any split** of the following underlying sources, their
mirrors, subsets or derivatives:

- AG News; Amazon Polarity; App Reviews; Banking77.
- Social Bias Frames (intent, offensiveness and sexual-content variants).
- Comparative Agendas Project / State of the Union (CAP/SOTU).
- DAIR Emotion; EmpatheticDialogues; Financial PhraseBank; IMDb.
- Manifesto; MASSIVE; Rotten Tomatoes; TrueTeacher.
- Wikipedia/Jigsaw toxicity (toxic, obscene, threat and insult variants).
- Yahoo Answers Topics; Yelp reviews.

This includes paraphrases, augmentations and teacher predictions derived from
those examples. Filter these sources out of multi-dataset training mixtures.

Some derivatives go by other names. These are excluded too, in every split and mirror:

- Rotten Tomatoes: the Stanford Sentiment Treebank in any form (SST-2, SST-5, GLUE
  `sst2`; for example `stanfordnlp/sst`, `stanfordnlp/sst2`, `SetFit/sst2`,
  `SetFit/sst5`), the MR / movie-review sentence-polarity data
  (`cornell-movie-review-data/rotten_tomatoes`) and the Subjectivity dataset
  (`subj`, `SetFit/subj`). All of them are built from the same Rotten Tomatoes
  review snippets.
- TrueTeacher: CNN/DailyMail (`cnn_dailymail`, `abisee/cnn_dailymail`), the source
  of TrueTeacher's articles. Do not use its articles or summaries, including to
  build factual-consistency pairs, and do not use datasets built from its articles
  (for example FactCC, the CNN/DailyMail part of FRANK, or SummEval). Other
  summarization corpora, such as XSum, are not TrueTeacher sources.
Do not initialize from a checkpoint deliberately fine-tuned on these classification
datasets. General pretrained LLMs and encoders are allowed: incidental, unknowable
upstream pretraining exposure is not the same as deliberately using evaluation
data during this task.

The supplied public validation examples (`validation.jsonl`) are 500 rows from the
DBpedia-14 **test split**, topic classification only. They are for local validation:
run `validate.py` against them to check that your server answers correctly and fast
enough. Do not train, fine-tune, distill, calibrate, fit thresholds or temperatures,
select prompts, choose checkpoints or tune hyperparameters on them. The same rule
covers the validation or test split of any other public dataset or benchmark;
training splits and other public corpora are fine. The 500 rows are not a
representative sample of the final evaluation distribution. Train on independent
data, or synthesize new examples without copying or deriving them from excluded sources.

Do not reuse code, checkpoints, training/distillation datasets or teacher outputs
from open-source projects presented as reproductions of TypeSafe JEV/System One.
This includes Jeff (`logan-markewich/jeff`), Laya (`NandhaKishorM/laya`), Open-Jev,
JEV-4B distillation artifacts, `typesafe-diffusion-lab`, and their forks, renamed
copies and derivatives. Independently implementing published ideas is allowed,
as are general libraries and ordinary pretrained encoders/LLMs. An existing model
does not become prohibited merely because somebody later wraps it in a JEV
reproduction.

Record all initialization checkpoints, data sources, teachers and reused model
code in `training_manifest.json`, with immutable revisions or content hashes.
Keep training code, configurations and logs sufficient to review how the final
weights were produced. A trained classification head or adapter is sufficient;
changing only a prompt, decision threshold or temperature is not training the
classifier for this task.
