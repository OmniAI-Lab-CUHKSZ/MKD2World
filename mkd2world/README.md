# MKD2World training and inference

## Focus labels

`training/focus_annotator.py` is the frozen annotator. Schema: `v4-focus-v20-hover-compact-global-support`. It imports the markdown parser in `training/focus_parser.py`. 

ADD and UPDATE spans on the next TSM are focus units. A deleted element does not produce a focus unit.

## 35B agent, current TSM

`training/qwen35_35b_current_tsm.py` fine-tunes Qwen3.5-35B-A3B with LoRA to reconstruct the current screenshot as TSM. `training/qwen35_35b_current_tsm_runtime.py` supervises that run. Prompt: `prompts/templatesV3/train_localwm/localwm_cur.md`.

## World model

`training/next_tsm_sft.py` fine-tunes the 9B world model. Prompt: `localwm_next.md`. The loss is element-averaged base CE, plus element-averaged focus CE, plus a changed-only page margin.

## Inference

`local_wm_runtime.py` turns a screenshot and one candidate action into the next TSM.

```bash
PYTHONPATH=. python -c "from mkd2world.training.focus_annotator import FOCUS_LABEL_VERSION; print(FOCUS_LABEL_VERSION)"
```
