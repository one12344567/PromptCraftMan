from langchain.agents import create_agent
from llm.myllm import llm
from defs.model import EvaluationReport,WorkFlowStateModel
from langchain.tools import tool






#参数导入，暂时没写相关模块，占个位置

model=llm
system_prompt="""
你是prompt评估助手，根据用户目的，评估prompt是否足够完善，输出评估结果，并决定下一步操作：
- 若信息不完善，返回到diagnosis模块，输出的next_step为diagnosis
- 若信息完善，但prompt不够完善，返回到optimization模块，输出的next_step为optimization
- 若信息完善，且prompt足够完善，下一步到finalize模块，输出的next_step为finalize

""".strip()

tools=[]
schema=EvaluationReport
name="evaluation_agent"

msg=""""""


class EvaluationAgent:
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
    evaluation_agent = EvaluationAgent()
    print(evaluation_agent.invoke(msg))