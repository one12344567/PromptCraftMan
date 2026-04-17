import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()
baidu_api_key = os.getenv("BAIDU_API_KEY")

@tool
def baidu_search_tool(query: str, top_k: int = 5) -> list[dict]:
    """
    从百度搜索，返回搜索结果
    query: 搜索关键词
    top_k: 返回结果数量，默认5个
    每日额度有限，尽量少用
    """
    print("-----------------\n")
    print(f"搜索了{query}")
    print("\n-----------------")
    return baidu_search(query, top_k)



def baidu_search(query: str, top_k: int = 5) -> list[dict]:
    url = "https://qianfan.baidubce.com/v2/ai_search/web_search"
    headers = {
        "X-Appbuilder-Authorization": f"Bearer {baidu_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [{"content": query, "role": "user"}],
        "search_source": "baidu_search_v2",
        "resource_type_filter": [{"type": "web", "top_k": top_k}],
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    refs = data.get("references", [])
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content", ""),
            "site": item.get("website", ""),
            "date": item.get("date", ""),
        }
        for item in refs
    ]


if __name__ == "__main__":
    print(baidu_search("许家印认罪"))