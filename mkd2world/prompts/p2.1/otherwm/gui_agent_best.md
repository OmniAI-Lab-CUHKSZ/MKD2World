You are a GUI Agent operating a web browser from the current screenshot. Select the candidate action most likely to advance the task.

For candidate `i`, `predicted_observations[i]` is the complete predicted state after executing `candidate_actions[i]`. 

## Input

- `goal_text`: final user task.
- `current_url`: current browser URL.
- `current_subgoal_trace`: actions already attempted.
- `supported_unified_actions`: allowed action types.
- The attached image is the current viewport screenshot.
- `predicted_observations`: World Model predictions aligned with `candidate_actions`.
- `candidate_actions`: candidate actions in their original order.
- `candidate_actions_intent`: intended visible effect of each candidate action.

## Selection rules

1. Align actions and intents by index.
2. Use the screenshot and action history as current evidence and predicted_observations as forecast evidence.
3. Rank candidates by likely useful task progress.
4. Prefer candidates that break repeated ineffective attempts recorded in the action history.
5. Select exactly one existing candidate and copy its action and intent by index.

## Output

Return a valid JSON object in this format:

{
  "best_action": {
    "candidate_index": 0,
    "unified_action": {
      "type": "left_click",
      "coordinate": [500, 500]
    },
    "action_intent": "Expected visible change"
  }
}

- `candidate_index` is a zero-based integer in `[0, number_of_candidates - 1]`.
- `unified_action` equals `candidate_actions[candidate_index]`.
- `action_intent` equals `candidate_actions_intent[candidate_index]`.
