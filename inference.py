#!/usr/bin/env python3
"""
inference.py — GDPR Compliance Auditor OpenEnv Agent
====================================================
Executes an LLM-based agent across 6 simulated data privacy tasks.
This script acts as both the environment state holder and the agent,
evaluating fractional rewards based on exact string-matching redactions.
Logging Rules Strictly Enforced:
    1. One [START] line per task.
    2. One [STEP] line per task (single-step episodes).
    3. One [END] line per task.
    4. No internal newlines in the output streams.
"""

import os
import re
import sys
import json
import textwrap
from typing import List, Optional, Tuple, Dict, Any

from openai import OpenAI

# ===========================================================================
# 1. Configuration & Environment Variables
# ===========================================================================

# API routing (Defaults to Hugging Face router, overrideable via .env)
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "gdpr-compliance-auditor"

# LLM generation parameters
TEMPERATURE  = 0.3    # Low temperature to enforce deterministic, valid JSON outputs
MAX_TOKENS   = 1024   # Generous limit to accommodate reasoning models (<think> blocks)

# The exact task IDs defined in openenv.yaml
TASKS = ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6"]


# ===========================================================================
# 2. Agent Prompts & Static Environment Ground Truth
# ===========================================================================

# The system prompt enforces strict adherence to the JSON schema and compliance rules
SYSTEM_PROMPT = textwrap.dedent("""
    You are a strictly compliant GDPR Data Privacy Auditor.
    You will receive a JSON observation detailing a data transfer request.
    You MUST reply with ONLY a valid JSON object matching this exact schema:
    {"decision":"approve"|"block"|"redact","fields_to_redact":["exact_string_1"],"reason_code":"string"}
    
    CRITICAL RULES:
    1. If a rule states a role "must never access" data, or data "CANNOT be transferred", your decision MUST be "block".
    2. If your decision is "redact", the `fields_to_redact` array MUST contain the EXACT text substrings to be scrubbed.
    3. NEVER output generic field keys like "name" or "credit_card". Output the actual exact values.
    4. Do not include markdown blocks or any text outside the JSON.
    5. OBJECT REDACTION: If a rule requires a specific user or record to be entirely removed from an array, you MUST include the EXACT string representation of that entire JSON dictionary object.
""").strip()

# Holds the observations the agent will see, and the exact answers used for grading
TASKS_DATA = {
    "task_1": {
        "observation": {"request_id": "REQ-1", "requester_role": "Marketing", "region": "US_CCPA", "raw_data": "{\"user\": \"John\", \"credit_card\": \"4532-xxxx-xxxx-1234\"}", "policy_rule": "Marketing must never access financial data."},
        "ground_truth": {"decision": "block", "fields_to_redact": [], "reason_code": "Policy_Violation"}
    },
    "task_2": {
        "observation": {"request_id": "REQ-2", "requester_role": "Data_Science", "region": "US_CCPA", "raw_data": "Chat: Hi, my name is Sarah and my phone number is 555-0199. My router is broken.", "policy_rule": "Data Scientists may access support logs only if direct identifiers (Names, Phones) are redacted."},
        "ground_truth": {"decision": "redact", "fields_to_redact": ["Sarah", "555-0199"], "reason_code": "PII_Redacted"}
    },
    "task_3": {
        "observation": {"request_id": "REQ-3", "requester_role": "External_Fraud_Auditor", "region": "Global_Mixed", "raw_data": "[{\"id\": \"U-103\", \"name\": \"Carlos\", \"fraud_score\": 0.2, \"deletion_requested\": true, \"purchases\": [\"Desk\"]}, {\"id\": \"U-104\", \"name\": \"Diana\", \"fraud_score\": 0.88, \"deletion_requested\": true, \"purchases\": [\"Insulin\"]}]", "policy_rule": "1. Default: Redact Names. 2. FCL Rule: Do NOT redact names if fraud_score > 0.80. 3. GDPR Rule: Block user entirely if deletion_requested is true. 4. Health Rule: ALWAYS redact medical purchases like Insulin."},
        "ground_truth": {"decision": "redact", "fields_to_redact": ["{\"id\": \"U-103\", \"name\": \"Carlos\", \"fraud_score\": 0.2, \"deletion_requested\": true, \"purchases\": [\"Desk\"]}", "Insulin"], "reason_code": "Mixed_Compliance"}
    },
    "task_4": {
        "observation": {"request_id": "REQ-4", "requester_role": "Product_Strategy", "region": "Global", "raw_data": "{\"session_length_sec\": 120, \"feature_used\": \"dark_mode\", \"device\": \"iOS\"}", "policy_rule": "Product Strategy may freely access fully anonymized application telemetry. Approve if no PII is present."},
        "ground_truth": {"decision": "approve", "fields_to_redact": [], "reason_code": "Anonymized_Telemetry"}
    },
    "task_5": {
        "observation": {"request_id": "REQ-5", "requester_role": "AI_Training_Cluster", "region": "EU_GDPR", "raw_data": "{\"voice_memo_id\": \"V-8821\", \"audio_transcript\": \"Remind me to buy milk.\", \"biometric_signature\": \"0x8A19F\"}", "policy_rule": "Raw voice or biometric signatures generated in the EU_GDPR region CANNOT be transferred to external AI clusters under any circumstances."},
        "ground_truth": {"decision": "block", "fields_to_redact": [], "reason_code": "Sovereign_Biometric_Rule"}
    },
    "task_6": {
        "observation": {"request_id": "REQ-6", "requester_role": "Third_Party_Vendor", "region": "US_CCPA", "raw_data": "{\"survey_response\": \"The new tool is fast!\", \"user_hash\": \"A7B9\", \"system_metadata\": \"logged_by: john.doe@internal-corp.com\"}", "policy_rule": "Third-party vendors may read user survey responses and hashes, but internal employee email addresses must NEVER be leaked in system metadata."},
        "ground_truth": {"decision": "redact", "fields_to_redact": ["john.doe@internal-corp.com"], "reason_code": "Employee_PII_Redacted"}
    }
}


# ===========================================================================
# 3. Strict Standardized Loggers
# ===========================================================================

def log_start(task_name: str, env: str, model: str) -> None:
    """Emits the environment initialization line."""
    print(f"[START] task={task_name} env={env} model={model}", flush=True)

def log_step(step: int, action: Any, reward: float, done: bool, error: Optional[str]) -> None:
    """Emits the single-step action log, forcefully removing all internal newlines."""
    # Convert dicts to compact JSON strings, or scrub newlines from raw strings
    if isinstance(action, dict):
        action_str = json.dumps(action, separators=(',', ':'))
    else:
        action_str = str(action).replace('\n', ' ')

    error_val = error.replace('\n', ' ') if error else "null"
    done_val = "true" if done else "false" # Convert boolean to lowercase string
    
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emits the final episode summary line, formatting all floats to 2 decimals."""
    success_val = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    
    print(f"[END] success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ===========================================================================
# 4. LLM API Interaction
# ===========================================================================

def get_model_action(client: OpenAI, observation: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Sends the observation to the LLM, sanitizes the response, and extracts JSON."""
    user_prompt = f"Observation: {json.dumps(observation, separators=(',', ':'))}"
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE, 
            max_tokens=MAX_TOKENS, 
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Strip out deep-reasoning blocks (e.g., <think> tags used by Qwen/DeepSeek)
        text_no_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text_no_think = re.sub(r'<think>.*', '', text_no_think, flags=re.DOTALL | re.IGNORECASE)
        
        # Strip standard markdown formatting to safely parse the JSON dictionary
        json_str = text_no_think.replace("```json", "").replace("```", "").strip()
        
        return json.loads(json_str), None
        
    except Exception as exc:
        # Route API errors to stderr so they don't corrupt the stdout parser stream
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        return None, str(exc)


# ===========================================================================
# 5. Main Task Execution Engine
# ===========================================================================

def run_task(client: OpenAI, task_name: str) -> None:
    """Runs a complete 1-step episode for a specific task and calculates fractional rewards."""
    
    # 1. EMIT START (Guaranteed first output)
    log_start(task_name, BENCHMARK, MODEL_NAME)
    
    task_def = TASKS_DATA[task_name]
    obs = task_def["observation"]
    truth = task_def["ground_truth"]
    
    # State tracking variables initialized outside the try-block for safe finally-block logging
    rewards: List[float] = []
    score = 0.0
    steps_taken = 1
    success = False
    last_error: Optional[str] = None
    
    # Fail-safe defaults in case of an early catastrophic crash
    action_data: Any = {}
    final_reward = 0.01 
    
    try:
        # 2a. Fetch LLM Action
        if client:
            action_dict, last_error = get_model_action(client, obs)
            if action_dict:
                action_data = action_dict
            else:
                action_data = {"decision": "block", "fields_to_redact": [], "reason_code": "API_Error"}
        else:
            # Local fallback for debugging without an API key
            action_data = {"decision": truth["decision"], "fields_to_redact": truth.get("fields_to_redact", []), "reason_code": "Local_Mock"}

        raw_reward = 0.01
        
        # 2b. Evaluate Agent Decision (Fractional Reward Math)
        if action_data.get("decision") == truth["decision"]:
            
            # Complex grading: If the correct action is "redact", grade the precision of the fields
            if truth["decision"] == "redact":
                agent_fields = set(action_data.get("fields_to_redact", []))
                truth_fields = set(truth.get("fields_to_redact", []))
                
                correct_redactions = agent_fields.intersection(truth_fields)
                over_redactions = agent_fields - truth_fields
                
                # Base reward for picking 'redact' correctly
                base_reward = 0.40
                
                # Fractional reward based on how many required fields were successfully found
                if truth_fields: 
                    redaction_reward = (len(correct_redactions) / len(truth_fields)) * 0.59 
                else: 
                    redaction_reward = 0.59 if not agent_fields else 0.0
                    
                # Deduct points for redacting things that didn't need to be redacted
                penalty = len(over_redactions) * 0.10
                
                raw_reward = base_reward + redaction_reward - penalty
                
            # Simple grading: If it correctly chose 'approve' or 'block', grant full points
            else:
                raw_reward = 0.99
                
        # Clamp bounds strictly between 0.01 and 0.99 for OpenEnv constraints
        final_reward = max(0.01, min(0.99, raw_reward))
        score = final_reward
        
        # A score of >= 0.80 represents a passing grade for the episode
        success = score >= 0.80

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", file=sys.stderr, flush=True)
        last_error = str(exc)
        
    finally:
        # Finalizers execute regardless of try-block success, ensuring logs never break
        rewards.append(final_reward)
        
        # 3. EMIT STEP (Records the action and the calculated reward)
        log_step(step=steps_taken, action=action_data, reward=final_reward, done=True, error=last_error)
        
        # 4. EMIT END (Closes the episode)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    """Initializes the client and iterates through the defined task sequence."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None
    
    # Loop through each task sequentially, treating each loop as an isolated episode
    for task_name in TASKS: 
        run_task(client, task_name)

if __name__ == "__main__":
    main()