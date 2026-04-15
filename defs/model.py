from typing import Literal

from pydantic import BaseModel, Field, model_validator

class DiagnosisReport(BaseModel):
    scene: str = Field(description="The scene of the use of the prompt")
    problems: list[str] = Field(description="The list of problems to diagnose", default_factory=list)
    missing_info: list[str] = Field(description="The list of missing information to clarify", default_factory=list)
    next_step: Literal["clarification",  "optimization"]

class ClarificationReport(BaseModel):
    questions : list[str] = Field(description="The list of questions to clarify")
    answers: list[str] = Field(description="The list of answers to the questions", default_factory=list)
    
class OptimizationReport(BaseModel):
    prompts : list[str] = Field(description="The list of prompts have optimized")
    improved_info: list[str] = Field(description="The list of improved information", default_factory=list)
    
    
class EvaluationReport(BaseModel):
    grades : list[float] = Field(description="The list of grades of the prompts")
    reason: str = Field(description="The reason to rewrite the prompt")
    next_step: Literal["diagnosis" ,"optimization"]
    
class QAReport(BaseModel):
    question: str = Field(description="The question to answer")
    answer: str = Field(description="The answer to the question")
class WorkFlowStateModel(BaseModel):
    original_prompt: str = Field(description="The original prompt")
    
    problems: list[str] = Field(description="The list of problems to diagnose", default_factory=list)
    missing_info: list[str] = Field(description="The list of missing information to clarify", default_factory=list)
    QA: list[QAReport] = Field(description="The list of questions and answers", default_factory=list)
    
    improved_info: list[str] = Field(description="The list of improved information", default_factory=list)
    
    final_prompt: str = Field(description="The final prompt after all steps")
    final_missing_info: list[str] = Field(description="The list of missing information after all steps", default_factory=list)
    
