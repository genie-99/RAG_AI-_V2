from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from config import GMS_KEY, OPENAI_BASE_URL, CHAT_MODEL, PROJECT_ROOT
from model import AgentState
from src.tools.search_tools import search_tools

#사용할 모델 설정
chat_model = ChatOpenAI(
    api_key = GMS_KEY,
    base_url = OPENAI_BASE_URL,
    model = CHAT_MODEL,
)

# 모델에게 사용할 수 있는 도구 목록 전달
model_with_tools = chat_model.bind_tools(
    search_tools,
    parallel_tool_calls=False
)

#에이전트 프롬프트 읽기
prompt_path = PROJECT_ROOT / "prompt" / "agent_prompt.txt"
system_prompt = prompt_path.read_text(encoding="utf-8")

#모델이 도구 사용을 요청하거나 최종 답변을 작성하는 작업
def call_model(state: AgentState):
    messages = [SystemMessage(content = system_prompt)] + state["messages"]

    response = model_with_tools.invoke(messages)

    return {"messages" : [response]}

# 에이전트가 사용할 상태 구조 지정
builder = StateGraph(AgentState)

# 모델 판단과 도구 실행 작업 등록
builder.add_node("agent", call_model)
builder.add_node("tools",ToolNode(search_tools))

#시작하면 먼저 모델 호출
builder.add_edge(START, "agent")

#모델의 응답에 따라 다음 작업 결정
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools":"tools",
        "__end__" : END,
    },
)

# 도구 실행 결과를 모델에게 다시 전달
builder.add_edge("tools", "agent")

#프로그램 실행 중 대화 기록 유지
agent = builder.compile(checkpointer=InMemorySaver())

if __name__ == "__main__":
    run_config = {
        "configurable": {"thread_id": "plant-study"},
        # 한 질문에서 실행할 그래프 단계 수의 상한
        "recursion_limit": 12,
    }

    while True:
        question = input("\n질문 (끝내려면 종료): ").strip()

        if question == "종료":
            break

        if not question:
            continue

        # 각 단계에서 새로 추가된 메시지 출력
        for state in agent.stream(
            {"messages": [HumanMessage(content=question)]},
            config=run_config,
            stream_mode="values",
        ):
            state["messages"][-1].pretty_print()