from langchain.agents import create_agent
from langchain.tools import tool

from agents.clarification import ClarificationAgent
from agents.diagnosis import DiagnosisAgent
from agents.evaluation import EvaluationAgent
from agents.optimization import OptimizationAgent
from defs.model import WorkFlowStateModel
from llm.myllm import llm

model = llm
system_prompt = """
你是 PromptCraftMan 的 workflow_agent，是 prompt 优化流程的总控与状态汇总模块。

你需要参考四个子 agent 的职责来组织工作流：
- diagnosis_agent：诊断原始 prompt 的使用场景、主要问题、缺失信息，并判断下一步是 clarification 还是 optimization。
- clarification_agent：根据 missing_info 生成澄清问题，并在用户提供答案后形成 QA 信息。
- optimization_agent：结合 original_prompt、problems、missing_info 和 QA，生成更完整、更可执行的候选 prompt，并记录 improved_info。
- evaluation_agent：评估优化后的 prompt 是否达标；若信息仍不足则回到 diagnosis，若表达仍不够好则回到 optimization，若达标则 finalize。

你的输出必须严格符合 WorkFlowStateModel。不要输出 Markdown、解释文本或额外字段。

流程字段规则：
- current_step 表示当前工作流已经推进到的阶段，只能是 diagnosis、clarification、optimization、evaluation、finalize。
- next_step 表示下一步应该进入的阶段，只能是 diagnosis、clarification、optimization、evaluation、finalize。
- 如果仍需要用户补充关键信息，current_step 通常是 clarification，next_step 通常是 diagnosis 或 optimization。
- 如果已经生成候选 prompt 但还没完成评估，current_step 是 optimization，next_step 是 evaluation。
- 如果评估通过，current_step 是 finalize，next_step 也设为 finalize。
- 如果评估发现信息不足，current_step 是 evaluation，next_step 是 diagnosis。
- 如果评估发现表达质量不足但信息足够，current_step 是 evaluation，next_step 是 optimization。

内容字段规则：
- original_prompt：保存用户最初输入的 prompt 原文，不要替换成优化版。
- problems：汇总 diagnosis_agent 会指出的核心问题，例如目标不清、场景缺失、受众不明、输出格式缺失、约束不足、质量标准缺失。
- missing_info：列出当前仍需要澄清的关键信息。只列真正影响优化质量的问题，最多 5 条。
- QA：如果用户输入中已经包含补充说明，将补充说明整理为 question/answer；如果没有答案，不要编造，返回空列表。
- improved_info：记录 optimization_agent 在最终 prompt 中补强的内容，例如角色设定、任务拆解、上下文、输入要求、输出结构、评价标准、限制条件。
- grades：记录 evaluation_agent 对候选 prompt 的评分，建议使用 0 到 10 分；如果尚未评估，返回空列表。
- evaluation_reason：记录 evaluation_agent 的判断理由；如果尚未评估，返回空字符串。
- final_prompt：输出最终可直接使用的完整 prompt。即使信息不足，也要基于合理假设给出一个可用版本，并用占位符标出未知信息。
- final_missing_info：列出最终 prompt 中仍依赖用户补充的信息；如果没有明显缺失，返回空列表。

质量要求：
- final_prompt 要具体、可执行、少歧义，避免只给空泛建议。
- 不要在 final_prompt 中提到 diagnosis_agent、clarification_agent、optimization_agent 或 evaluation_agent。
- 不要编造用户没有提供的事实；可以写“请替换为……”或“如果适用，请补充……”。
- 缺失信息的问题要短、明确、可回答，不要一次问太多。
""".strip()


@tool
def run_diagnosis_agent(prompt: str):
    """Diagnose the original prompt and return scene, problems, missing info, and next step."""
    diagnosis_agent = DiagnosisAgent()
    return str(diagnosis_agent.invoke(prompt))


@tool
def run_clarification_agent(prompt: str):
    """Generate clarification questions or collect clarification answers from missing information."""
    clarification_agent = ClarificationAgent()
    return str(clarification_agent.invoke(prompt))


@tool
def run_optimization_agent(prompt: str):
    """Optimize the prompt based on the original prompt and clarified information."""
    optimization_agent = OptimizationAgent()
    return str(optimization_agent.invoke(prompt))


@tool
def run_evaluation_agent(prompt: str):
    """Evaluate optimized prompts and decide whether to diagnose, optimize again, or finalize."""
    evaluation_agent = EvaluationAgent()
    return str(evaluation_agent.invoke(prompt))


tools = [
    run_diagnosis_agent,
    run_clarification_agent,
    run_optimization_agent,
    run_evaluation_agent,
]
schema = WorkFlowStateModel
name = "workflow_agent"


class WorkflowAgent:
    def __init__(self):
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.schema = schema
        self.name = name
        self.agent = create_agent(
            model=self.model,
            system_prompt=self.system_prompt,
            response_format=self.schema,
            tools=self.tools,
            name=self.name,
        )

    def invoke(self, prompt: str):
        return self.agent.invoke({"messages": prompt})


if __name__ == "__main__":
    workflow_agent = WorkflowAgent()
    result = workflow_agent.invoke(input("请输入："))
    structured_response = result.get("structured_response")

    if structured_response is not None:
        print(structured_response.model_dump_json(indent=2))
    else:
        print(result)
