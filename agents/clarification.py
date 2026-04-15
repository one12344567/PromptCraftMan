from langchain.agents import create_agent
from llm.myllm import llm
from defs.model import ClarificationReport,WorkFlowStateModel
from langchain.tools import tool

@tool
def get_info(info: str) -> str:
    """向用户提问，获取信息"""
    print(info)
    ans = input("请输入：")
    return ans




#参数导入，暂时没写相关模块，占个位置

model=llm
system_prompt="""
你是prompt优化助手的子模块，你的任务是根据上一模块输出的missing_info，向用户提问，补全信息，输出补全后的信息。
""".strip()

tools=[get_info]
schema=ClarificationReport
name="clarification_agent"

msg="""missing_info=[
      '论文的具体主题或研究领域是什么？',
      '论文的类型是什么（综述、研究论文、案例分析等）？',
      '目标读者是谁（学术期刊、会议、学位论文等）？',
      '论文的长度要求是多少？',
      '需要包含哪些具体部分（摘要、引言、方法、结果、讨论等）？',
      '是否有特定的格式要求（APA、IEEE、MLA等）？',
      '研究的目的是什么？',
      '是否有特定的时间范围或资源限制？'
    ]"""


class ClarificationAgent:
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
    diagnosis_agent = ClarificationAgent()
    print(diagnosis_agent.invoke(msg))