from typing import Literal

from pydantic import BaseModel, Field, model_validator

class DiagnosisModel(BaseModel):
    problems: list[str] = Field(description="The list of problems to diagnose")
    next_step: Literal["clarification",  "optimization"]

class ClarificationModel(BaseModel):
    questions : list[str] = Field(description="The list of questions to clarify")
    
class OptimizationModel(BaseModel):
    prompts : list[str] = Field(description="The list of prompts have optimized")
    
class EvaluationModel(BaseModel):
    grades : list[float] = Field(description="The list of grades of the prompts")
    reason: str = Field(description="The reason to rewrite the prompt")