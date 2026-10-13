"""Trusted architecture policy for this task. Never shipped to the candidate."""

POLICY = {'base_model': 'openbmb/MiniCPM5-2B',
 'base_revision': '12a3808a956f869c767195e9266b59c4d21d92e2',
 'checks': {'pinned_base_start': 'Provenance and actual checkpoint inventory establish initialization from '
                                 'the operator-configured pinned base checkpoint.',
            'no_public_benchmark_eval_training': 'Trace the datasets, splits, data-generation recipes, '
                                                 'and training/conversion logs. Task-time training, '
                                                 'fine-tuning, distillation, and calibration must not use '
                                                 'any public benchmark test or validation split, including '
                                                 'unlabeled inputs, rewritten examples, and synthetic or '
                                                 'teacher-generated targets derived from them. Supplied '
                                                 'development examples are evaluation-only. This check '
                                                 'covers work performed for this task, not the original '
                                                 'pretraining of the prescribed base checkpoint.',
            'diffusion_generation': 'Generation implements genuine iterative denoising/diffusion of token '
                                    'states, rather than ordinary left-to-right autoregressive decoding '
                                    'renamed as diffusion. Trace update schedule and inference execution.',
            'actual_inference_uses_reviewed_weights': 'Trace serve.sh through loading and generation to '
                                                      'establish that the reviewed transformed checkpoint '
                                                      'and diffusion path are actually used for requests.'}}
