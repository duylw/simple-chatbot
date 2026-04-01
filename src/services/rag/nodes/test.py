from src.services.rag.state import (
    ThreadState
)

from src.services.rag.context import Context

from src.services.rag.nodes import (
    continue_after_guardrail,
    invoke_query_guardrail,
    invoke_out_of_scope_response,
    invoke_query_rewrite,
    invoke_get_relevant_documents,
    invoke_rerank,
    invoke_generate_answer,
    invoke_grade_answer,
    invoke_response
)


from langchain.messages import HumanMessage, ToolMessage

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    # This is just for testing the individual nodes. The actual orchestration happens in the agent.
    import logging
    import sys

    # Force standard output to use UTF-8 for Vietnamese characters
    sys.stdout.reconfigure(encoding='utf-8')

    class ColorNodeFormatter(logging.Formatter):
        # ANSI color codes
        CYAN = '\033[96m'
        RESET = '\033[0m'

        def format(self, record):
            # Format the log string first
            msg = super().format(record)
            # If the literal "NODE:" is in the message, colorize it
            if "NODE:" in msg:
                msg = msg.replace("NODE:", f"{self.CYAN}NODE:{self.RESET}")
            return msg

    # Create a console handler and attach the custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorNodeFormatter('%(levelname)s - %(message)s'))

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler]
    )


    import asyncio
    from langgraph.graph import START, END, StateGraph
    from langgraph.prebuilt import tools_condition, ToolNode
    from src.services.rag.tools import create_retriever_tool
    from langchain_chroma import Chroma
    from src.services.rag.config import GraphConfig
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from dotenv import load_dotenv
    load_dotenv() # Load environment variables from .env file
    async def test():
        test_md = "src/services/rag/nodes/test.md"
        
        settings = GraphConfig()

        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model
        )

        chroma = Chroma(
            embedding_function=embeddings,
            host=settings.settings.CHROMA_HOST,
            port="8008",
        )
        vectordb_retriever = chroma.as_retriever()

        hybrid_search = create_retriever_tool(vectordb_retriever=vectordb_retriever, top_k=10)
        tools = [hybrid_search]

        workflow = StateGraph(ThreadState, context_schema=Context)
        workflow.add_node("query_guardrail", invoke_query_guardrail)
        workflow.add_node("out_of_scope_response", invoke_out_of_scope_response)
        workflow.add_node("query_rewrite", invoke_query_rewrite)
        workflow.add_node("get_relevant_documents", invoke_get_relevant_documents)
        workflow.add_node("search_tool", ToolNode(tools))
        workflow.add_node("generate_answer", invoke_generate_answer)
        workflow.add_node("grade_answer", invoke_grade_answer)
        workflow.add_node("response", invoke_response)

        workflow.add_edge(START, "query_guardrail")

        workflow.add_conditional_edges(
            "query_guardrail",
            continue_after_guardrail,
            {"continue": "query_rewrite", "out_of_scope": "out_of_scope_response"}
        )

        workflow.add_edge("out_of_scope_response", END)
        workflow.add_edge("query_rewrite", "get_relevant_documents")

        workflow.add_conditional_edges(
            "get_relevant_documents",
            tools_condition,
            {"tools": "search_tool"}
        )
        workflow.add_edge("search_tool", "generate_answer")
        workflow.add_edge("generate_answer", "grade_answer")

        workflow.add_conditional_edges(
            "grade_answer",
            lambda state: state.get("routing_decision", "response"),
            {"response": "response", "rewrite_query": "query_rewrite"}
        )

        workflow.add_edge("response", END)

        agent = workflow.compile()

        context = Context(
            n_iterations=2
        )
        inital_state = {
            "messages": [
                HumanMessage(content="giải thích cơ chế hoạt động của self-attention trong mô hình transformer?"),
            ],
            "n_iterations": 0,
            "n_llm_calls": 0
        }        
        final_state = await agent.ainvoke(inital_state, context=context)

        with open(test_md, "w", encoding="utf-8") as f:
            f.write("# Test Log\n\n")
            for mess in final_state["messages"]:
                if isinstance(mess, ToolMessage):
                    continue
                role = "User" if isinstance(mess, HumanMessage) else "Assistant"
                role = "Tool" if isinstance(mess, ToolMessage) else role

                f.write(f"## {role} Message\n\n")
                if hasattr(mess, "tool_calls") and mess.tool_calls:
                    f.write("### Tool Calls\n\n")
                    for call in mess.tool_calls:
                        f.write(f"- **Tool Name**: {call['name']}\n")
                        f.write(f"  - **ID**: {call['id']}\n\n")
                else:
                    
                    f.write(f"{mess.content}\n\n")

    asyncio.run(test())