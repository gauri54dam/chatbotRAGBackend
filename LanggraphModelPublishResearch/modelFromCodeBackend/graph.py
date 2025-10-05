
import os
from dotenv import load_dotenv
import requests
import datetime

host = "azuredatabricks.net"
os.environ["DATABRICKS_HOST"] = host
load_dotenv()


# Langraph components ---------
from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

memory: MemorySaver = MemorySaver()


# Databricks models and embeddings -------------
from langchain_community.chat_models import ChatDatabricks
from langchain_community.embeddings import DatabricksEmbeddings

# vector search ----------
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch

# MLFLow -------------
from mlflow.models import infer_signature
import mlflow
import langchain
import langchain_core
import langchain_community




## databricks model and embeddings
chat_model = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", max_tokens = 1024)
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

#databricks vector search
index_name="analytics.models.docs_idxs"
VECTOR_SEARCH_ENDPOINT_NAME="document_vector_endpoint"

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host

    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=os.environ['DATABRICKS_HOST'], personal_access_token=os.environ['DATABRICKS_TOKEN'])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="text", embedding=embedding_model
    )
    return vectorstore.as_retriever()


def web_search_tool(query: str) -> str:
    headers = {
    "X-API-KEY": os.environ["SERPAPI_KEY"],
    "Content-Type": "application/json"
        }

    params = {
    "q": query
    }

    res = requests.post("https://google.serper.dev/search", json=params, headers=headers)
    res_json = res.json()
    results = res_json.get("organic", [])
    if results:
        return results[0].get("snippet", "")
    return "No useful web data found."

from langgraph.graph import START, END 

def load_graph() -> CompiledStateGraph:


    # --- Langgraph State Definition
    class State(TypedDict):
        messages: Annotated[List[AnyMessage], add_messages]
        retrieved_context: str
        initial_chatbot_response: str
        transformed_query: str  

    # --- Langgraph Graph Builder
    graph_builder = StateGraph(State)
    retriever = get_retriever()

    def retrieval_node(state: State):
        """
        This node is having intelligent retriver node logic to differentiate between new questions and follow-up questions
        It examines the conversation history and rephrases the last user question into a self-contained query
        to solve the topic-switching and follow-up question problem.
        """
        # Get the conversation history and the latest message
        messages = state['messages']
        last_user_message = messages[-1].content
        conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in messages[:-1]])

        # This prompt asks the LLM to create a self-contained query, effectively handling both new topics and follow-ups in one step.
        query_generation_prompt = f"""Based on the chat history below, formulate a self-contained search query from the "Follow Up Input".
        This query will be used to retrieve relevant documents from a vector database. Make the query specific and detailed.

        Chat History:
        {conversation_history}

        Follow Up Input:
        {last_user_message}

        Optimized Search Query:"""

        # Use the chat model to generate the query
        response = chat_model.invoke(query_generation_prompt)
        search_query = response.content.strip()

        docs = retriever.get_relevant_documents(search_query)
        # Process and return the context as before
        if not docs:
            context = "No relevant HR policy document was found for the query."
        else:
            context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}




    def chatbot_node(state: State):

        prompt = f"""You are an HR Assistant chatbot designed to provide accurate and up-to-date answers to employees HR-related queries. Your responses should be clear, concise, and directly based on the company's official policies. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, say that you don't know; don't try to make up an HR policy answer. Keep the answer as concise as possible. Provide all answers only in English. Use the following pieces of context to answer the HR Policy question:
        {state['retrieved_context']}"""

        llm_messages = [SystemMessage(content=prompt)]

        llm_messages.extend(state['messages']) # Adds all messages from the current conversation

        response = chat_model.invoke(llm_messages)
        return {"messages": [response],"initial_chatbot_response": response.content}
        
    

    def enhancer_node(state: State):
    
        user_query = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)][-1].content
        initial_answer = state["initial_chatbot_response"]
        retrieved_hr_context = state["retrieved_context"]
        current_date_str = datetime.date.today().strftime("%B %d, %Y")

        web_context = web_search_tool(user_query)
        print(f"Web search conducted. Results: {web_context[:100]}...")

        synthesis_prompt = f"""
        You are an HR Assistant chatbot providing comprehensive and accurate answers.
        You have already generated an initial response based on internal HR policies.
        Now, you have additional external information from a web search.

        Your task is to **synthesize** all available information to provide the best possible answer to the user.
        **Do NOT simply copy-paste directly from the web search results.**
        Integrate the web search results with the initial HR policy response, making sure to:
        - Add any missing factual details (like specific dates or external holiday names if relevant and not in HR policy).
        - Clarify or correct information if the web search provides more accurate external facts.
        - Prioritize information from the **internal HR policy** for HR-related questions. Use web search primarily for external facts (e.g., current date-specific holidays, general knowledge).
        - If the web search provides no new relevant information, you can mostly stick to the initial answer, perhaps rephrasing slightly for clarity.
        - If the web search contradicts internal HR policy on a *policy* matter, state that the company policy is the primary source.
        - Maintain a neutral, respectful, and concise tone.
        - Provide all answers only in English.

        --- User Query ---
        {user_query}

        --- Initial HR Policy Response ---
        {initial_answer}

        --- Internal HR Policy Context (from retrieval) ---
        {retrieved_hr_context}

        --- External Web Search Results (if available) ---
        {web_context}

        --- Current Date ---
        {current_date_str}

        Based on the above, provide the final, synthesized answer:
        """
        
      
        llm_messages_for_llm = [SystemMessage(content=synthesis_prompt)]
        
        # Find the last human message for direct context
        last_human_message = next((msg for msg in reversed(state['messages']) if isinstance(msg, HumanMessage)), None)
        if last_human_message:
            llm_messages_for_llm.append(last_human_message)
        else:
            print("Warning: No human message found in state for enhanced_response_node.")
            
        final_response = chat_model.invoke(llm_messages_for_llm)
        
        # LangGraph's 'add_messages' will correctly append this new AIMessage to state['messages']
        # This also effectively replaces the 'initial_chatbot_response' in the conversation history
        return {"messages": [final_response]}


    # --- The Router Node ---
    def should_enhance(state: State) -> str:
        """
        Uses an LLM to decide if the user's query requires a web search for external facts.
        """
        
        user_query = [msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)][-1]
        
        # Use an LLM to make the routing decision
        routing_prompt = f"""You are a routing agent. Your goal is to decide if a user's query requires a web search for external, factual information (like specific dates, holidays, states, current events) or if it can be answered solely by an internal HR policy document.

        User Query: "{user_query}"

        Does this query likely require a web search for up-to-date, external facts? Answer only with the single word 'yes' or 'no'.
        """
        response = chat_model.invoke(routing_prompt)
        decision = response.content.strip().lower()
        print(f"--- [Router] LLM decision: '{decision}' ---")

        if 'yes' in decision:
            print("--- [Router] Path chosen: enhance ---")
            return "enhance"
        else:
            print("--- [Router] Path chosen: end ---")
            return "end"

    # --- Add Nodes to the Graph ---
    graph_builder.add_node("retrieval_node", retrieval_node)
    graph_builder.add_node("chatbot_node", chatbot_node)
    graph_builder.add_node("enhancer_node", enhancer_node)

    # --- Define Graph Edges with Conditional Logic ---
    graph_builder.add_edge(START, "retrieval_node")
    graph_builder.add_edge("retrieval_node", "chatbot_node")

    # After the chatbot node, we call the router to decide the next step
    graph_builder.add_conditional_edges(
        "chatbot_node",
        should_enhance,
        {
            # If the router returns "enhance", we go to the enhancer_node.
            "enhance": "enhancer_node",
            # If the router returns "end", we finish the graph execution.
            "end": END
        }
    )

    graph_builder.add_edge("enhancer_node", END)

    # --- Compile the Graph ---
    graph = graph_builder.compile(checkpointer=memory)




  

    
    return graph

# Set are model to be leveraged via model from code

mlflow.models.set_model(load_graph())



