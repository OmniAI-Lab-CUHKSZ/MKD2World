# GPT judges

## Annotation quality

It scores four fields from 0 to 100:

- `annotation_object_validity`
- `boundary_alignment`
- `label_semantic_reasonableness`
- `page_complexity`

Image A is used only for page complexity. Image B is the annotated screenshot. The returned JSON contains those four integers and `sample_id`.

## Trajectory success

`trajectory_judge.py` builds the Online-Mind2Web completion prompt (`build_prompt`) and can submit it to an OpenAI-compatible endpoint. Pass `--base-url` and export `OPENAI_API_KEY`.

```bash
python -c "from judges.trajectory_judge import build_prompt; print('success' in build_prompt('demo', {'result': {'task': 'check the weather', 'action_history': []}, 'metadata': {}, 'screenshots': [], 'raw_steps': [], 'execution_errors': []}))"
```
