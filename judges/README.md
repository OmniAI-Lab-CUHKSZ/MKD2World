# GPT judges

## Trajectory success

`trajectory_judge.py` builds the Online-Mind2Web completion prompt (`build_prompt`) and can submit it to an OpenAI-compatible endpoint. Pass `--backend openai_compatible`, `--base-url`, and `--api-key-file`, a file that holds the API key.

```bash
python -c "from judges.trajectory_judge import build_prompt; print('success' in build_prompt('demo', {'result': {'task': 'check the weather', 'action_history': []}, 'metadata': {}, 'screenshots': [], 'raw_steps': [], 'execution_errors': []}))"
```
