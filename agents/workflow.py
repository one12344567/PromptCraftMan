import json

from agents.clarification import ClarificationAgent
from agents.diagnosis import DiagnosisAgent
from agents.evaluation import EvaluationAgent
from agents.optimization import OptimizationAgent
from defs.model import QAReport, WorkFlowStateModel


def _extract_structured_response(result: dict):
    structured_response = result.get("structured_response")
    if structured_response is None:
        raise ValueError("agent 返回中缺少 structured_response")
    return structured_response


def _build_optimization_context(
    original_prompt: str,
    problems: list[str],
    missing_info: list[str],
    qa: list[QAReport],
    evaluation_reason: str = "",
    optimization_feedback: str = "",
) -> str:
    payload = {
        "original_prompt": original_prompt,
        "problems": problems,
        "missing_info": missing_info,
        "QA": [item.model_dump() for item in qa],
    }
    if evaluation_reason:
        payload["evaluation_reason"] = evaluation_reason
    if optimization_feedback:
        payload["optimization_feedback"] = optimization_feedback
    return json.dumps(payload, ensure_ascii=False)


def _build_fallback_final_prompt(original_prompt: str, missing_info: list[str]) -> str:
    if not missing_info:
        return original_prompt

    placeholders = "\n".join(f"- {item}" for item in missing_info)
    return (
        "请基于以下任务生成结果，并在生成前先补全必要信息。\n\n"
        f"任务：{original_prompt}\n\n"
        "请补充的信息：\n"
        f"{placeholders}\n\n"
        "如果暂时无法补全，请使用明确占位符保留缺失项，并输出一个完整、可执行的最终提示词。"
    )


def _collect_answers_once(questions: list[str]) -> tuple[list[QAReport], list[str]]:
    print("需要补充以下信息：")
    for index, question in enumerate(questions, start=1):
        print(f"{index}. {question}")

    print("\n请直接用一段话补充你已知的信息，workflow 会自动识别并继续处理。")
    user_context = input("请输入补充说明：").strip()
    if not user_context:
        return [], questions

    qa_records = [QAReport(question=question, answer=user_context) for question in questions]
    return qa_records, []


class WorkflowAgent:
    def __init__(self):
        self.diagnosis_agent = DiagnosisAgent()
        self.clarification_agent = ClarificationAgent()
        self.optimization_agent = OptimizationAgent()
        self.evaluation_agent = EvaluationAgent()

    def invoke(self, prompt: str) -> dict:
        diagnosis = _extract_structured_response(self.diagnosis_agent.invoke(prompt))

        state = {
            "current_step": "diagnosis",
            "next_step": diagnosis.next_step,
            "original_prompt": prompt,
            "problems": diagnosis.problems,
            "missing_info": diagnosis.missing_info,
            "QA": [],
            "candidate_prompt": "",
            "improved_info": [],
            "grade": None,
            "evaluation_reason": "",
            "final_prompt": _build_fallback_final_prompt(prompt, diagnosis.missing_info),
            "final_missing_info": diagnosis.missing_info,
        }

        if diagnosis.next_step == "clarification" and diagnosis.missing_info:
            clarification = _extract_structured_response(
                self.clarification_agent.invoke(
                    json.dumps({"missing_info": diagnosis.missing_info}, ensure_ascii=False)
                )
            )
            state["current_step"] = "clarification"
            state["next_step"] = "clarification"
            state["missing_info"] = clarification.questions
            state["final_missing_info"] = clarification.questions
            return {"structured_response": WorkFlowStateModel(**state)}

        optimization = _extract_structured_response(
            self.optimization_agent.invoke(
                _build_optimization_context(
                    original_prompt=prompt,
                    problems=diagnosis.problems,
                    missing_info=diagnosis.missing_info,
                    qa=[],
                )
            )
        )
        evaluation = _extract_structured_response(
            self.evaluation_agent.invoke(optimization.prompt)
        )

        state["current_step"] = "finalize" if evaluation.next_step == "finalize" else "evaluation"
        state["next_step"] = "finalize" if evaluation.next_step == "finalize" else evaluation.next_step
        state["candidate_prompt"] = optimization.prompt
        state["improved_info"] = optimization.improved_info
        state["grade"] = evaluation.grade
        state["evaluation_reason"] = evaluation.evaluation_reason
        state["final_prompt"] = optimization.prompt
        return {"structured_response": WorkFlowStateModel(**state)}

    def invoke_interactive(self, prompt: str) -> dict:
        diagnosis = _extract_structured_response(self.diagnosis_agent.invoke(prompt))

        qa_records: list[QAReport] = []
        remaining_missing_info = list(diagnosis.missing_info)

        if diagnosis.next_step == "clarification" and diagnosis.missing_info:
            clarification = _extract_structured_response(
                self.clarification_agent.invoke(
                    json.dumps({"missing_info": diagnosis.missing_info}, ensure_ascii=False)
                )
            )
            qa_records, remaining_missing_info = _collect_answers_once(clarification.questions)

        optimization = _extract_structured_response(
            self.optimization_agent.invoke(
                _build_optimization_context(
                    original_prompt=prompt,
                    problems=diagnosis.problems,
                    missing_info=remaining_missing_info,
                    qa=qa_records,
                )
            )
        )

        evaluation = _extract_structured_response(
            self.evaluation_agent.invoke(optimization.prompt)
        )

        candidate_prompt = optimization.prompt
        improved_info = optimization.improved_info
        grade = evaluation.grade
        evaluation_reason = evaluation.evaluation_reason
        next_step = evaluation.next_step
        current_step = "finalize" if next_step == "finalize" else "evaluation"

        while True:
            print("\n当前准备返回的 final_prompt：\n")
            print(candidate_prompt)
            print("\n是否需要继续优化？输入 y 继续优化，直接回车或输入其他内容结束。")
            refine = input("请输入：").strip().lower()
            if refine != "y":
                break

            feedback = input("请说明你希望继续优化的方向：\n请输入：").strip()
            optimization_retry = _extract_structured_response(
                self.optimization_agent.invoke(
                    _build_optimization_context(
                        original_prompt=prompt,
                        problems=diagnosis.problems,
                        missing_info=remaining_missing_info,
                        qa=qa_records,
                        evaluation_reason=evaluation_reason,
                        optimization_feedback=feedback,
                    )
                )
            )
            candidate_prompt = optimization_retry.prompt
            improved_info = optimization_retry.improved_info

            evaluation_retry = _extract_structured_response(
                self.evaluation_agent.invoke(candidate_prompt)
            )
            grade = evaluation_retry.grade
            evaluation_reason = evaluation_retry.evaluation_reason
            next_step = evaluation_retry.next_step
            current_step = "finalize" if next_step == "finalize" else "evaluation"

        state = WorkFlowStateModel(
            current_step=current_step,
            next_step="finalize" if next_step == "finalize" else next_step,
            original_prompt=prompt,
            problems=diagnosis.problems,
            missing_info=remaining_missing_info,
            QA=qa_records,
            candidate_prompt=candidate_prompt,
            improved_info=improved_info,
            grade=grade,
            evaluation_reason=evaluation_reason,
            final_prompt=candidate_prompt,
            final_missing_info=remaining_missing_info,
        )
        return {"structured_response": state}


if __name__ == "__main__":
    workflow_agent = WorkflowAgent()
    result = workflow_agent.invoke_interactive(input("请输入："))
    structured_response = result.get("structured_response")

    if structured_response is not None:
        print(structured_response.model_dump_json(indent=2, ensure_ascii=False))
    else:
        print(result)
