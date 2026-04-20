import json
import shutil
import sys

from agents.clarification import ClarificationAgent
from agents.diagnosis import DiagnosisAgent
from agents.evaluation import EvaluationAgent
from agents.optimization import OptimizationAgent
from defs.model import QAReport, WorkFlowStateModel


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"


def _supports_color() -> bool:
    return sys.stdout.isatty()


def _style(text: str, *codes: str) -> str:
    if not _supports_color():
        return text
    return f"{''.join(codes)}{text}{RESET}"


def _term_width(default: int = 72) -> int:
    try:
        return max(60, min(100, shutil.get_terminal_size((default, 20)).columns))
    except OSError:
        return default


def _rule(char: str = "─") -> str:
    return char * _term_width()


def _print_header(title: str, subtitle: str | None = None) -> None:
    print(_style(_rule("═"), CYAN, BOLD))
    print(_style(title.center(_term_width()), CYAN, BOLD))
    if subtitle:
        print(_style(subtitle.center(_term_width()), DIM))
    print(_style(_rule("═"), CYAN, BOLD))


def _print_stage(title: str, tone: str = MAGENTA) -> None:
    print()
    print(_style(f"◆ {title}", tone, BOLD))
    print(_style(_rule("─"), DIM))


def _print_kv(label: str, value: str) -> None:
    print(f"{_style(label, BOLD)} {value}")


def _print_panel(title: str, body: str, tone: str = BLUE) -> None:
    width = _term_width()
    print(_style(f"┌{'─' * (width - 2)}┐", tone))
    print(_style(f"│ {title}".ljust(width - 1) + "│", tone, BOLD))
    print(_style(f"├{'─' * (width - 2)}┤", tone))
    for line in (body or "").splitlines() or [""]:
        visible = line[: width - 4]
        print(_style(f"│ {visible}".ljust(width - 1) + "│", tone))
    print(_style(f"└{'─' * (width - 2)}┘", tone))


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
    _print_stage("信息澄清", YELLOW)
    print("需要补充以下信息：")
    for index, question in enumerate(questions, start=1):
        print(f"{index}. {question}")

    print()
    print("请直接用一段话补充你已知的信息，workflow 会自动识别并继续处理。")
    user_context = input(_style("请输入补充说明： ", YELLOW, BOLD)).strip()
    if not user_context:
        return [], questions

    qa_records = [QAReport(question=question, answer=user_context) for question in questions]
    return qa_records, []


def _format_list(items: list[str]) -> str:
    if not items:
        return "无"
    return "\n".join(f"- {item}" for item in items)


def _format_qa(qa_items: list[QAReport]) -> str:
    if not qa_items:
        return "无"
    lines: list[str] = []
    for index, item in enumerate(qa_items, start=1):
        lines.append(f"{index}. 问题：{item.question}")
        lines.append(f"   回答：{item.answer}")
    return "\n".join(lines)


def format_workflow_result(state: WorkFlowStateModel) -> str:
    grade_text = "未评估" if state.grade is None else str(state.grade)
    sections = [
        ("当前步骤", state.current_step),
        ("下一步", state.next_step),
        ("评分", grade_text),
        ("评估说明", state.evaluation_reason or "无"),
        ("原始提示词", state.original_prompt),
        ("诊断问题", _format_list(state.problems)),
        ("缺失信息", _format_list(state.missing_info)),
        ("补充问答", _format_qa(state.QA)),
        ("优化补强点", _format_list(state.improved_info)),
        ("候选提示词", state.candidate_prompt or "无"),
        ("最终提示词", state.final_prompt),
        ("最终仍缺失的信息", _format_list(state.final_missing_info)),
    ]
    blocks = [
        "==============================",
        "PromptCraftMan 处理结果",
        "==============================",
    ]
    for title, content in sections:
        blocks.append(f"【{title}】")
        blocks.append(content)
        blocks.append("")
    return "\n".join(blocks).rstrip()


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
        _print_header("PromptCraftMan", "提示词工作流优化")

        _print_stage("需求输入", CYAN)
        _print_kv("原始需求：", prompt)

        _print_stage("问题诊断", MAGENTA)
        diagnosis = _extract_structured_response(self.diagnosis_agent.invoke(prompt))
        print(_format_list(diagnosis.problems))

        qa_records: list[QAReport] = []
        remaining_missing_info = list(diagnosis.missing_info)

        if diagnosis.next_step == "clarification" and diagnosis.missing_info:
            clarification = _extract_structured_response(
                self.clarification_agent.invoke(
                    json.dumps({"missing_info": diagnosis.missing_info}, ensure_ascii=False)
                )
            )
            qa_records, remaining_missing_info = _collect_answers_once(clarification.questions)

        _print_stage("提示词优化", GREEN)
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

        _print_stage("质量评估", BLUE)
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
            print()
            _print_panel("当前准备返回的 final_prompt", candidate_prompt, GREEN)
            _print_kv("当前评分：", "未评估" if grade is None else str(grade))
            _print_kv("评估说明：", evaluation_reason or "无")
            print()
            refine = input(
                _style("是否需要继续优化？输入 y 继续，其它任意输入结束： ", YELLOW, BOLD)
            ).strip().lower()
            if refine != "y":
                break

            feedback = input(_style("请说明你希望继续优化的方向： ", YELLOW, BOLD)).strip()
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
    user_prompt = input(_style("请输入： ", YELLOW, BOLD))
    workflow_agent = WorkflowAgent()
    result = workflow_agent.invoke_interactive(user_prompt)
    structured_response = result.get("structured_response")

    if structured_response is not None:
        print()
        _print_panel("处理结果", format_workflow_result(structured_response), CYAN)
    else:
        print(result)
