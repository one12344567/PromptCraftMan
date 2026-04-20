from typing import Literal

from pydantic import BaseModel, Field

WorkflowStep = Literal["diagnosis", "clarification", "optimization", "evaluation", "finalize"]


class DiagnosisReport(BaseModel):
    scene: str = Field(description="提示词的使用场景", min_length=1)
    problems: list[str] = Field(
        description="诊断出的主要问题列表",
        default_factory=list,
        max_length=8,
    )
    missing_info: list[str] = Field(
        description="仍需澄清的关键信息列表",
        default_factory=list,
        max_length=5,
    )
    next_step: Literal["clarification", "optimization"]


class ClarificationReport(BaseModel):
    questions: list[str] = Field(
        description="需要向用户确认的澄清问题列表",
        default_factory=list,
        max_length=5,
    )


class OptimizationReport(BaseModel):
    prompt: str = Field(description="优化后的候选提示词", min_length=1)
    improved_info: list[str] = Field(
        description="本轮优化补强的关键信息列表",
        default_factory=list,
        max_length=8,
    )


class EvaluationReport(BaseModel):
    grade: float = Field(description="对候选提示词的评分", ge=0, le=10)
    evaluation_reason: str = Field(description="评分与流转决策的理由", min_length=1)
    next_step: Literal["diagnosis", "optimization", "finalize"]


class QAReport(BaseModel):
    question: str = Field(description="向用户提出的问题", min_length=1)
    answer: str = Field(description="用户给出的回答", min_length=1)


class WorkFlowStateModel(BaseModel):
    current_step: WorkflowStep = Field(description="当前工作流所处的步骤")
    next_step: WorkflowStep = Field(description="下一步应该进入的步骤")

    original_prompt: str = Field(description="用户最初输入的原始提示词", min_length=1)

    problems: list[str] = Field(
        description="诊断阶段发现的问题列表",
        default_factory=list,
        max_length=8,
    )
    missing_info: list[str] = Field(
        description="当前仍缺失的关键信息列表",
        default_factory=list,
        max_length=5,
    )
    QA: list[QAReport] = Field(
        description="澄清阶段收集到的问答记录",
        default_factory=list,
    )

    candidate_prompt: str = Field(description="优化阶段产出的候选提示词", default="")
    improved_info: list[str] = Field(
        description="优化阶段补强的关键信息列表",
        default_factory=list,
        max_length=8,
    )

    grade: float | None = Field(
        description="评估阶段对候选提示词的评分",
        default=None,
        ge=0,
        le=10,
    )
    evaluation_reason: str = Field(description="评估阶段给出的判断理由", default="")

    final_prompt: str = Field(
        description="最终可直接复制使用的完整提示词，必须是成品，而不是备注、提纲、标签或离散信息。",
        min_length=1,
    )
    final_missing_info: list[str] = Field(
        description="最终仍然缺失、且无法从当前上下文合理推断的信息列表",
        default_factory=list,
        max_length=5,
    )
