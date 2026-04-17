from langchain.agents import create_agent
from langchain.tools import tool

from llm.myllm import llm
from defs.model import DiagnosisReport,WorkFlowStateModel
from agents.mytools.mytools import baidu_search_tool

#参数导入，暂时没写相关模块，占个位置

model=llm
system_prompt="""
你是prompt优化助手的内部agent，你的任务是诊断prompt的问题，根据输出格式要求与用户输入的prompt，输出    scene: str = Field(description="The scene of the use of the prompt")
    problems: list[str] = Field(description="The list of problems to diagnose", default_factory=list)
    missing_info: list[str] = Field(description="The list of missing information to clarify", default_factory=list)
    next_step: Literal["clarification",  "optimization"]
""".strip()

tools=[baidu_search_tool]
schema=DiagnosisReport
name="diagnosis_agent"


class DiagnosisAgent:
    def __init__(self):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.schema = schema
        self.tools = tools
        self.agent = create_agent(
            model=self.model,
            system_prompt=self.system_prompt,
            response_format=self.schema,
            tools=self.tools,
            name=self.name,
        )
    
    def invoke(self, prompt: str):
        return self.agent.invoke({"messages":prompt})
    
if __name__ == "__main__":
    diagnosis_agent = DiagnosisAgent()
    print(diagnosis_agent.invoke("写一篇科技论文"))