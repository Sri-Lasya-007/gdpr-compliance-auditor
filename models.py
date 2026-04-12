from pydantic import BaseModel, Field
from typing import Literal, List, Optional

# --- MODEL: OBSERVATION ---
class PrivacyObservation(BaseModel):
    request_id: str
    requester_role: str
    region: str
    raw_data: str
    policy_rule: str
    last_action_error: Optional[str] = None

# --- MODEL: ACTION ---
class ComplianceAction(BaseModel):
    decision: Literal["approve", "block", "redact"]
    fields_to_redact: Optional[List[str]] = Field(
        default_factory=list,
        description="List of exact strings to scrub."
    )
    reason_code: str

# --- MODEL: REWARD ---
class StepReward(BaseModel):
    value: float
    reason: str