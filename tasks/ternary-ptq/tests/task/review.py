"""Trusted architecture policy for this task. Never shipped to the candidate."""

POLICY = {'base_model': 'Qwen/Qwen3.5-4B',
 'base_revision': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
 'checks': {'pinned_qwen_start': 'Training/init and quantization provenance establish post-training '
                                 'quantization from the pinned Qwen main language model. The vision tower '
                                 '(model.visual.*) and MTP head (mtp.*) may be omitted or kept unchanged in '
                                 'full precision; neither is counted towards the ternary parameter ratio. '
                                 'All main language model tensors remain required. The output head may be untied '
                                 '(an added lm_head.weight, which may be separately trained).',
            'text_only_inference': 'The serving path uses the main autoregressive language model. '
                                   'The vision tower and MTP head may remain as unused full-precision '
                                   'weights, but neither participates in inference or evaluation.',
            'no_public_benchmark_eval_training': 'Trace the datasets, splits, data-generation recipes, '
                                                 'and training/quantization logs. Task-time training, '
                                                 'fine-tuning, distillation, and calibration, including '
                                                 'fitting quantization scales or weights, must not use '
                                                 'any public benchmark test or validation split, including '
                                                 'unlabeled inputs, rewritten examples, and synthetic or '
                                                 'teacher-generated targets derived from them. Supplied '
                                                 'development examples are evaluation-only. This check '
                                                 'covers work performed for this task, not the original '
                                                 'pretraining of the prescribed base checkpoint.',
            'actual_inference_uses_reviewed_weights': 'Trace serve.sh through loading and inference to '
                                                      'establish actual use of the inspected checkpoint, '
                                                      'with no alternate hidden model. Compare the serving path with '
                                                      'the numerical audit audited_storage references and the full '
                                                      'weight inventory. Unused training/optimizer artifacts and '
                                                      'shared storage are permitted; unreviewed active inference '
                                                      'weights are not.',
            'ternary_execution': 'The active inference path loads the ternary checkpoint: every tensor the '
                                 'manifest declares literal or packed_2bit is served as code times scale and is '
                                 'never replaced by its original full precision values. Tensors the manifest '
                                 'declares float (the audit counts them as nonternary within its 5% allowance) '
                                 'and the input embedding and output head, which the audit excludes from the '
                                 'fraction and lists in trusted_audits.ternary.excluded_parameters, may '
                                 'legitimately be served in full precision. The trusted numerical audit is '
                                 'independently mandatory.'}}


def submission_inspector(snapshot):
    """The reviewer receives the numerical result before deciding architecture compliance."""
    from frontier_ml.scoring.architecture import inspect_ternary
    return {"ternary": inspect_ternary(snapshot)}
