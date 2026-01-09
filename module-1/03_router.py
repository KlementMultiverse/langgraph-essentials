"""
Router - Module 1

A LangGraph router that demonstrates:
1. Single LLM decision point
2. Autonomous tool execution
3. Conditional routing (execute tool OR respond directly)
4. Using pre-built ToolNode and tools_condition

Flow: START → tool_calling_llm → [tools OR END] → END
"""

from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")


# ============================================================================
# CONCEPT: ROUTER PATTERN
# ============================================================================

# A Router is a simple agent with ONE decision point:
# - Should I call a tool? → Execute it
# - Should I respond directly? → Return text
#
# After executing (or not), it ENDS. No loop back to LLM.
#
# Router flow:
# START → LLM → [Execute Tool OR Respond] → END
#
# Agent flow (next lesson):
# START → LLM → [Execute Tool] → LOOP BACK TO LLM → ... → END


# ============================================================================
# STEP 1: DEFINE TOOLS
# ============================================================================

def multiply(a: int, b: int) -> int:
    """Multiply two numbers together.

    Args:
        a: First integer
        b: Second integer

    Returns:
        The product of a and b
    """
    return a * b

def add(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First integer
        b: Second integer

    Returns:
        The sum of a and b
    """
    return a + b

def divide(a: float, b: float) -> float:
    """Divide a by b.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        The result of a divided by b
    """
    if b == 0:
        return "Error: Division by zero"
    return a / b


# ============================================================================
# STEP 2: CREATE LLM WITH TOOLS
# ============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools([multiply, add, divide])


# ============================================================================
# STEP 3: DEFINE NODES
# ============================================================================

def tool_calling_llm(state: MessagesState):
    """
    Node that calls LLM with conversation history.

    LLM will decide:
    - Call a tool (returns AIMessage with tool_calls)
    - Respond directly (returns AIMessage with content)
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Q: Why don't we define a tools node?
# A: We use the pre-built ToolNode! It automatically:
#    - Receives AIMessage with tool_calls
#    - Finds the matching tool function
#    - Executes it with the provided arguments
#    - Returns ToolMessage with results


# ============================================================================
# CONCEPT: PRE-BUILT COMPONENTS
# ============================================================================

# 1. ToolNode - Pre-built node for tool execution
# ────────────────────────────────────────────────
# Usage: ToolNode([list_of_tools])
#
# What it does:
# - Receives state with AIMessage containing tool_calls
# - Executes each tool call
# - Returns ToolMessages with results
#
# Example:
# Input: AIMessage(tool_calls=[{"name": "multiply", "args": {"a": 5, "b": 3}}])
# Output: ToolMessage(content="15")

# 2. tools_condition - Pre-built conditional edge
# ────────────────────────────────────────────────
# What it does:
# - Checks if last AIMessage has tool_calls
# - If yes → Routes to "tools" node
# - If no → Routes to END
#
# Pseudo-code:
# def tools_condition(state):
#     if state["messages"][-1].tool_calls:
#         return "tools"
#     return END


# ============================================================================
# STEP 4: BUILD GRAPH
# ============================================================================

builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply, add, divide]))

# Add edges
builder.add_edge(START, "tool_calling_llm")

# Conditional edge: Route based on tool_calls
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,  # Pre-built function that routes to "tools" or END
)

# After tools execute, go to END
builder.add_edge("tools", END)

# Compile
graph = builder.compile()


# ============================================================================
# KEY INSIGHT: THE CONDITIONAL EDGES
# ============================================================================

# Q: Why don't I see an explicit edge from tool_calling_llm to END?
# A: The tools_condition creates TWO edges automatically:
#
#    builder.add_conditional_edges("tool_calling_llm", tools_condition)
#
#    Creates:
#    1. tool_calling_llm → tools (if tool_calls exist)
#    2. tool_calling_llm → END (if no tool_calls)
#
# Visual:
#           tool_calling_llm
#              /        \
#        tool_calls?   No tool_calls?
#            /              \
#         tools             END
#          |
#         END


# ============================================================================
# STEP 5: TEST THE ROUTER
# ============================================================================

def run_example(user_input: str):
    """Helper to run the graph and display results"""
    print("\n" + "=" * 60)
    print(f"USER: {user_input}")
    print("=" * 60)

    result = graph.invoke({"messages": [HumanMessage(content=user_input)]})

    print("\nConversation:")
    for msg in result["messages"]:
        if isinstance(msg, HumanMessage):
            print(f"  🧑 Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                print(f"  🤖 AI: [Calling tool: {msg.tool_calls[0]['name']}]")
            else:
                print(f"  🤖 AI: {msg.content}")
        else:
            print(f"  🔧 Tool: {msg.content}")

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("ROUTER PATTERN DEMO")
    print("=" * 60)
    print("\nA router makes ONE decision:")
    print("- Execute tool → Return result → END")
    print("- Respond directly → END")
    print("\nNo loop back to LLM!")

    # Test 1: No tool needed
    run_example("Hello! How are you?")

    # Test 2: Single tool needed
    run_example("What is 15 multiplied by 23?")

    # Test 3: Different tool
    run_example("Calculate 100 divided by 4")

    # Test 4: Another direct response
    run_example("Thank you!")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
    ✅ Router executes tools (unlike Chain which only requests)
    ✅ One decision point (tool OR respond)
    ✅ No loop back to LLM (unlike Agent)
    ✅ ToolNode handles execution automatically
    ✅ tools_condition routes automatically

    Limitation: Cannot chain tools based on results
    Example that FAILS with Router:
    "Multiply 5 by 3, then add 10 to the result"

    Router would call multiply(5, 3), return "15", then END.
    It never sees the result to decide to call add(15, 10).

    For that, you need an AGENT (next lesson)!
    """)


# ============================================================================
# ROUTER VS CHAIN VS AGENT
# ============================================================================

# CHAIN (Lesson 2):
# ─────────────────
# Flow: START → LLM → END
# Capability: Requests tool calls (doesn't execute)
# Use case: Prepare tool calls for external systems
# Example: LLM returns "call multiply(5, 3)" as structured output

# ROUTER (This Lesson):
# ─────────────────────
# Flow: START → LLM → [tools OR END] → END
# Capability: Executes tools OR responds (one-shot)
# Use case: Single tool decisions (80% of production agents)
# Example: "What is 5 * 3?" → multiply(5, 3) → "15" → END

# AGENT (Next Lesson):
# ────────────────────
# Flow: START → LLM ↔ tools (loops) → END
# Capability: Multi-step reasoning, tool chaining
# Use case: Complex tasks requiring multiple tools
# Example: "Multiply 5 * 3, then add 10"
#   → multiply(5, 3) → "15"
#   → LLM sees "15"
#   → add(15, 10) → "25"
#   → END


# ============================================================================
# PRODUCTION CONSIDERATIONS
# ============================================================================

# When to use Router:
# ✅ Single tool decision needed
# ✅ Independent calculations/lookups
# ✅ Predictable, testable behavior
# ✅ Fast, no multiple LLM calls

# When to use Agent instead:
# ✅ Tool outputs feed into next decision
# ✅ Multi-step reasoning required
# ✅ Dynamic number of steps
# ✅ Complex research/analysis tasks

# Production tip:
# 80% of "agent" use cases are actually routers.
# Start with router, upgrade to agent only when you need loops.


# ============================================================================
# DEEP DIVE Q&A
# ============================================================================

# Q: What if LLM calls multiple tools at once?
# A: ToolNode executes ALL of them in sequence, returns all ToolMessages.
#    Example:
#    LLM returns: [multiply(5,3), add(10,2)]
#    ToolNode executes both → Returns [ToolMessage("15"), ToolMessage("12")]
#    Then END.

# Q: Can the LLM see tool results?
# A: NO! In a router, after tools execute, it goes to END.
#    The LLM never sees the ToolMessage.
#    This is the key difference from an Agent.

# Q: How does tools_condition know to route to "tools"?
# A: It's hardcoded to return the string "tools" when tool_calls exist.
#    LangGraph then maps that string to the node named "tools".

# Q: Can I use custom routing logic instead of tools_condition?
# A: YES! Define your own function:
#    def my_router(state):
#        if some_custom_logic(state):
#            return "tools"
#        return END
#
#    builder.add_conditional_edges("llm", my_router)

# Q: What happens if a tool raises an error?
# A: ToolNode catches it and returns ToolMessage with error info.
#    Example: divide(5, 0) → ToolMessage("Error: Division by zero")

# Q: Can I have multiple ToolNodes?
# A: YES! You can route to different ToolNodes:
#    builder.add_node("math_tools", ToolNode([multiply, add]))
#    builder.add_node("search_tools", ToolNode([web_search]))
#    Then use custom routing to decide which to call.
