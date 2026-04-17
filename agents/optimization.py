from langchain.agents import create_agent
from llm.myllm import llm
from defs.model import OptimizationReport,WorkFlowStateModel
from langchain.tools import tool
from agents.mytools.mytools import baidu_search_tool





#参数导入，暂时没写相关模块，占个位置

model=llm
system_prompt="""
你是prompt优化助手，根据用户的原始prompt和上一模块输出的missing_info的answer，优化prompt，输出优化后的prompt。
""".strip()

tools=[]
schema=OptimizationReport
name="optimization_agent"

msg=""""""


class OptimizationAgent:
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
    optimization_agent = OptimizationAgent()
    print(optimization_agent.invoke(msg))