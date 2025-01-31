import os
import re
import requests
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import numpy as np
from datetime import datetime
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing import Optional

from langchain_openai import ChatOpenAI


from langchain.prompts import PromptTemplate

from langchain_core.runnables import Runnable

from langgraph.graph.message import AnyMessage, add_messages

import shutil
import uuid

from IPython.display import Image, display


from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
# from langchain_groq import ChatGroq


from langgraph.prebuilt import ToolNode


from typing import Annotated
from typing import Optional

from typing_extensions import TypedDict
from pydantic import BaseModel

from langgraph.graph.message import AnyMessage, add_messages

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

import sqlite3

OPENAI_API_KEY = ""
HF_KEY = ""
GROQ_API_KEY=""

EMBEDDING_MODEL_NAME=  "sentence-transformers/all-MiniLM-L6-v2"

# Initialize Hugging Face embeddings
hf_embeddings = HuggingFaceHubEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_KEY
)

# file_path = 'faq.md' 

local_file = "ecommerce.db"
backup_file = "ecommerce.backup.sqlite"

db = local_file

# with open(file_path, 'r') as f:
#     faq_text = f.read()

# docs = [
#     Document(page_content=txt.strip()) for txt in re.split(r"(?=\n##)", faq_text) if txt.strip()
# ]

# persist_dir = 'chroma'

# vectordb = Chroma.from_documents(
#     documents=docs,
#     embedding=hf_embeddings,
#     persist_directory=persist_dir
# )

persist_dir = 'chroma'

# Load the existing ChromaDB
vectordb = Chroma(
    persist_directory=persist_dir, 
    embedding_function=hf_embeddings
)

# Policy Lookup Tool
@tool
def lookup_policy(query):
    """Consult company policies or FAQs to check whether certain options are permitted."""
    results = vectordb.similarity_search(query, k=1)
    return "\n\n".join([doc.page_content for doc in results])


# FAQ Tool
@tool
def fetch_faq(query: str) -> str:
    """Fetch answers to frequently asked questions."""
    results = vectordb.similarity_search(query, k=1)
    return "\n\n".join([doc.page_content for doc in results])

@tool
def fetch_user_order_information(config: RunnableConfig) -> list[dict]:
    """Fetch all orders for the user along with corresponding product information and shipping details.

    Args:
        config (dict): Configuration containing user-specific settings, such as the customer_id.

    Returns:
        A list of dictionaries where each dictionary contains the order details,
        associated product details, and the shipping status for each order belonging to the user.
    """
    configuration = config.get("configurable", {})
    customer_id = configuration.get("customer_id", None)
    if not customer_id:
        raise ValueError("No customer ID configured.")

    conn = sqlite3.connect("ecommerce.db")
    cursor = conn.cursor()

    query = """
    SELECT
        o.order_id, o.status AS order_status,
        p.name AS product_name, p.description, p.price, oi.quantity,
        s.shipping_address, s.shipping_status
    FROM
        Orders o
        JOIN Order_Items oi ON o.order_id = oi.order_id
        JOIN Products p ON oi.product_id = p.product_id
        JOIN Shipping s ON o.order_id = s.order_id
    WHERE
        o.customer_id = ?
    """
    cursor.execute(query, (customer_id,))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return results

@tool
def search_products(
    keyword: Optional[str] = None,
    category: Optional[str] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    limit: int = 20,
) -> str:
    """Search for products based on keywords, category, and price range."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    query = "SELECT * FROM products WHERE 1=1"
    params = []

    if keyword:
        query += " AND name LIKE ?"
        params.append(f"%{keyword}%")
    if category:
        query += " AND category = ?"
        params.append(category)
    if price_min:
        query += " AND price >= ?"
        params.append(price_min)
    if price_max:
        query += " AND price <= ?"
        params.append(price_max)

    query += " LIMIT ?"
    params.append(limit)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()
    return (
        f"Found {len(results)} products:\n"
        + "\n".join([f"- Name : {product['name']} Price : (Rs.{product['price']} Company Name : {product['company_name']} Product Type : {product['category']} Product Description : {product['description']}" for product in results])
        if results
        else "No products found."
    )


# @tool
# def check_order_status(
#     order_id: Optional[str] = None,
#     customer_id: Optional[str] = None,
# ) -> str:
#     """Fetch the status of an order based on various search parameters."""
#     import sqlite3

#     try:

#         conn = sqlite3.connect(db)
#         cursor = conn.cursor()

#         query = "SELECT * FROM orders WHERE 1=1"
#         params = []

#         if order_id:
#             query += " AND order_id = ?"
#             params.append(order_id)
#         if customer_id:
#             query += " AND customer_id = ?"
#             params.append(customer_id)

#         cursor.execute(query, params)
#         rows = cursor.fetchall()

#         column_names = [desc[0] for desc in cursor.description]
#         cursor.close()
#         conn.close()

#         if rows:
#             results = [dict(zip(column_names, row)) for row in rows]
#             response = "Found the following orders:\n"
#             for order in results:
#                 response += (
#                     f"- Order ID: {order['order_id']}\n"
#                     f"  Status: {order['status']}\n"
#                     f"  Customer ID: {order['customer_id']}\n"
#                     "  --------------------------\n"
#                 )
#             return response
#         else:
#             return "No orders found. Please check the details and try again."

#     except sqlite3.OperationalError as e:
#         return f"Database error occurred: {e}"

#     except Exception as e:
#         return f"An unexpected error occurred: {e}"

def check_order_status(
    order_id: Optional[str] = None,
    # customer_id: Optional[str] = None,
) -> str:
    """Fetch the status of an order based on various search parameters using OR condition."""
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        
        query = "SELECT * FROM orders WHERE 1=0"
        params = []
        
        if order_id:
            query += " OR order_id = ?"
            params.append(order_id)
        # if customer_id:
        #     query += " OR customer_id = ?"
        #     params.append(customer_id)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        column_names = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()
        
        if rows:
            results = [dict(zip(column_names, row)) for row in rows]
            response = "Found the following orders:\n"
            for order in results:
                response += (
                    f"- Order ID: {order['order_id']}\n"
                    f"  Status: {order['status']}\n"
                    f"  Customer ID: {order['customer_id']}\n"
                    "  --------------------------\n"
                )
            return response
        else:
            return "No orders found. Please check the details and try again."

    except sqlite3.OperationalError as e:
        return f"Database error occurred: {e}"

    except Exception as e:
        return f"An unexpected error occurred: {e}"



def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            customer_id = configuration.get("customer_id", None)
            state = {**state, "user_info": customer_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}
    
llm = ChatOpenAI(model="gpt-4-turbo-preview",
                 temperature=0,
                 api_key=OPENAI_API_KEY)


# llm = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     temperature=0,
#     groq_api_key=GROQ_API_KEY
# )

ecommerce_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional and empathetic e-commerce support assistant. Your goal is to provide accurate, prompt, "
            "and helpful responses to customer inquiries. Act as a real customer support agent, not an AI. "
            "Use the tools provided to assist customers effectively with product searches, order status inquiries, cancellations, and FAQs. "
            "Always prioritize resolving their queries persistently and professionally, even if an error occurs with a tool or incomplete data is provided. "
            "If a tool fails, do not give up—try alternative approaches such as broadening the search, using other available information, or asking clarifying questions to gather more details. "
            "If the customer writes in English, respond in English. If they write in Roman Urdu, respond in Roman Urdu. "
            "Maintain a friendly yet professional tone in all interactions."
            "properly structure the output"
            "\n\nCurrent customer details:\n<Customer>\n{user_info}\n</Customer>"
            "\nCurrent time: {time}."
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)


ecommerce_tools = [
    search_products,
    fetch_faq,
    check_order_status
]

ecommerce_assistant_runnable = ecommerce_assistant_prompt | llm.bind_tools(ecommerce_tools)

builder = StateGraph(State)

builder.add_node("assistant", Assistant(ecommerce_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(ecommerce_tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
ecommerce_graph = builder.compile(checkpointer=memory)
