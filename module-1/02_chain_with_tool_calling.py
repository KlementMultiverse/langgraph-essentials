"""
Chain with Tool Calling - Module 1

A LangGraph chain that combines:
1. Messages as state (conversation history)
2. Chat models (LLM integration)
3. Tool binding (function calling)
4. Message reducers (append instead of overwrite)

Flow: START → tool_calling_llm → END
"""

from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")


# ============================================================================
# CONCEPT 1: MESSAGES AS STATE
# ============================================================================

# Instead of simple strings, we use structured messages with roles:
# - HumanMessage: From the user
# - AIMessage: From the chat model
# - SystemMessage: Instructions for the model
# - ToolMessage: Results from tool execution

# Example:
# messages = [
#     HumanMessage(content="Hello!"),
#     AIMessage(content="Hi! How can I help?"),
# ]


# ============================================================================
# CONCEPT 2: REDUCERS (add_messages)
# ============================================================================

# Problem: By default, LangGraph OVERWRITES state values
# When a node returns {"messages": [new_message]}, it would replace all old messages!
#
# Solution: Reducers define HOW to merge old and new values
# add_messages is a reducer that APPENDS new messages to existing list
#
# Without reducer (default):
# old_state = {"messages": [msg1, msg2]}
# node_return = {"messages": [msg3]}
# new_state = {"messages": [msg3]}  ← Lost msg1, msg2! ❌
#
# With add_messages reducer:
# old_state = {"messages": [msg1, msg2]}
# node_return = {"messages": [msg3]}
# new_state = {"messages": [msg1, msg2, msg3]}  ← All preserved! ✅


# ============================================================================
# STEP 1: DEFINE STATE WITH REDUCER
# ============================================================================

class MessagesState(TypedDict):
    # Annotated[type, reducer_function]
    # This tells LangGraph: "When updating 'messages', use add_messages to merge"
    messages: Annotated[list[AnyMessage], add_messages]

# Q: Can I use MessagesState from langgraph.graph?
# A: YES! LangGraph provides a pre-built MessagesState:
#    from langgraph.graph import MessagesState
#    It's exactly the same as above, just saves typing.

# Q: Why wrap in list []?
# A: Because add_messages expects a list. Even if your node returns 1 message,
#    you return it as: {"messages": [single_message]}
#    The reducer will append that list to the existing messages list.


# ============================================================================
# CONCEPT 3: TOOLS (FUNCTION CALLING)
# ============================================================================

# Tools let LLMs interact with external systems via Python functions.
# The LLM sees the function signature and decides when to call it.

def multiply(a: int, b: int) -> int:
    """Multiply two numbers together.

    Args:
        a: First integer
        b: Second integer

    Returns:
        The product of a and b
    """
    return a * b

# Q: Why the detailed docstring?
# A: The LLM reads this to understand WHEN and HOW to use the tool.
#    Better docstrings = Better tool usage decisions.

# Q: What does bind_tools() do?
# A: It tells the LLM: "You have access to these functions.
#    When user input suggests you need them, return a tool call instead of text."


# ============================================================================
# STEP 2: CREATE CHAT MODEL WITH TOOLS
# ============================================================================

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Bind tools to LLM
llm_with_tools = llm.bind_tools([multiply])

# Q: What happens when we invoke llm_with_tools?
# A: The LLM looks at the input and decides:
#    - If it needs a tool → Returns AIMessage with tool_calls
#    - If it doesn't → Returns AIMessage with regular content

# Example outputs:
# Input: "Hello!"
# → AIMessage(content="Hi! How can I help?", tool_calls=[])
#
# Input: "What is 2 * 3?"
# → AIMessage(content="", tool_calls=[{"name": "multiply", "args": {"a": 2, "b": 3}}])


# ============================================================================
# STEP 3: DEFINE NODE
# ============================================================================

def tool_calling_llm(state: MessagesState):
    """
    Node that calls the LLM with the current message history.

    The LLM will:
    1. Read all messages in state["messages"]
    2. Decide if it needs to call a tool or respond with text
    3. Return an AIMessage (with or without tool_calls)
    """
    # Get current messages from state
    current_messages = state["messages"]

    # Invoke LLM with full conversation history
    ai_response = llm_with_tools.invoke(current_messages)

    # Return as a dict with "messages" key
    # The add_messages reducer will append this to existing messages
    return {"messages": [ai_response]}

# Q: Why return {"messages": [ai_response]} instead of just ai_response?
# A: LangGraph nodes MUST return a dict matching the state schema.
#    State has key "messages", so we return {"messages": [...]}.
#    The [] wraps the single AIMessage in a list (required by add_messages).

# Q: What if we returned {"messages": ai_response} (no list)?
# A: It would still work! add_messages is smart enough to convert
#    a single message to a list. But wrapping in [] is more explicit.


# ============================================================================
# STEP 4: BUILD GRAPH
# ============================================================================

builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)

graph = builder.compile()


# ============================================================================
# STEP 5: RUN EXAMPLES
# ============================================================================

def run_example(user_input: str):
    """Helper to run the graph and display results"""
    print("\n" + "=" * 60)
    print(f"USER INPUT: {user_input}")
    print("=" * 60)

    # Invoke graph with initial message
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]})

    # Display all messages
    for msg in result["messages"]:
        if isinstance(msg, HumanMessage):
            print(f"\n🧑 Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                print(f"\n🤖 AI (tool call): {msg.tool_calls}")
            else:
                print(f"\n🤖 AI: {msg.content}")

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("CHAIN WITH TOOL CALLING DEMO")
    print("=" * 60)
    print("\nThis demonstrates:")
    print("1. Messages as state")
    print("2. add_messages reducer (preserves conversation history)")
    print("3. Tool binding (function calling)")
    print("4. LLM decides when to use tools")

    # Example 1: Regular conversation (no tool needed)
    run_example("Hello! How are you?")

    # Example 2: Tool calling required
    run_example("What is 15 multiplied by 23?")

    # Example 3: Another regular conversation
    run_example("Thank you!")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
    1. Messages preserve conversation context
    2. add_messages reducer appends instead of overwriting
    3. LLM automatically decides when to call tools
    4. Tool calls return structured arguments (not executed yet)

    Next step: Building an agent that EXECUTES tool calls (not just requests them)
    """)


# ============================================================================
# DEEP DIVE Q&A
# ============================================================================

# Q: Why don't we execute the multiply() function?
# A: This is a CHAIN, not an AGENT. It only requests tool calls.
#    In the next lesson (Agent), we'll add a node that:
#    1. Checks if AI returned tool_calls
#    2. Executes the actual Python function
#    3. Returns results as ToolMessage
#    4. Passes back to LLM to formulate final answer

# Q: What's the difference between Chain and Agent?
# A: CHAIN: Fixed flow (START → LLM → END)
#    AGENT: Dynamic flow (START → LLM → Tool → LLM → ... → END)
#    Agents have loops, chains don't.

# Q: When would I use a chain vs agent?
# A: CHAIN: When you just need to prepare tool calls (pass to external system)
#    AGENT: When you want autonomous execution with multiple steps

# Q: Can add_messages merge messages from multiple nodes?
# A: YES! If you have 3 nodes that all return messages, add_messages
#    will append all of them in order to the messages list.

# Q: What if I want to return multiple messages from one node?
# A: Return them in a list:
#    return {"messages": [AIMessage(...), AIMessage(...), ToolMessage(...)]}

# Q: Does the order of messages matter?
# A: YES! LLMs read messages in order. The sequence represents conversation flow.
#    Mixing up order will confuse the model.

# Q: Can I mix HumanMessage, AIMessage, SystemMessage?
# A: YES! Common pattern:
#    messages = [
#        SystemMessage(content="You are a helpful assistant"),
#        HumanMessage(content="Hello"),
#        AIMessage(content="Hi!"),
#        HumanMessage(content="What's 2+2?"),
#    ]
