"""
Module 3 - Lesson 1: Breakpoints for Human-in-the-Loop

This module demonstrates how to use breakpoints to interrupt graph execution
for human approval workflows.

Key Concepts:
- interrupt_before: Pause execution before a node runs
- interrupt_after: Pause execution after a node runs
- Human approval patterns
- Resuming execution with graph.stream(None, thread)

Learning Outcomes:
✓ Implement approval workflows
✓ Control graph execution flow
✓ Surface state to users for review
✓ Continue execution after user decision

Author: Klement G
Date: 2026-01-13
Course: LangChain Academy - Introduction to LangGraph
"""

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage


# =============================================================================
# TOOLS DEFINITION
# =============================================================================

def multiply(a: int, b: int) -> int:
    """
    Multiply a and b.

    Args:
        a: first int
        b: second int

    Returns:
        Product of a and b
    """
    return a * b


def add(a: int, b: int) -> int:
    """
    Adds a and b.

    Args:
        a: first int
        b: second int

    Returns:
        Sum of a and b
    """
    return a + b


def divide(a: int, b: int) -> float:
    """
    Divide a by b.

    Args:
        a: first int
        b: second int

    Returns:
        Division result
    """
    return a / b


# =============================================================================
# GRAPH SETUP
# =============================================================================

# Initialize tools and LLM
tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

# System message for the assistant
sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)


# Define the assistant node
def assistant(state: MessagesState):
    """
    Assistant node that invokes the LLM with tools.

    Args:
        state: Current graph state containing messages

    Returns:
        Updated state with assistant's response
    """
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# Build the graph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Add edges
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If tool call -> go to tools, else -> END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile with checkpointer and BREAKPOINT
memory = MemorySaver()
graph = builder.compile(
    interrupt_before=["tools"],  # ⚠️ KEY: Interrupt before tools execute
    checkpointer=memory
)


# =============================================================================
# EXECUTION PATTERNS
# =============================================================================

def run_basic_breakpoint_example():
    """
    Example 1: Basic breakpoint - run to interruption, then continue.

    Pattern:
    1. Run until breakpoint
    2. Graph pauses before tools node
    3. User can inspect state
    4. Resume with graph.stream(None, thread)
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Breakpoint")
    print("="*70)

    # Input
    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}
    thread = {"configurable": {"thread_id": "1"}}

    # Run until interruption
    print("\n📍 Running until breakpoint...")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()

    # Check state
    state = graph.get_state(thread)
    print(f"\n🛑 Graph paused at: {state.next}")
    print(f"   Waiting to execute: {state.next[0]} node")

    # Continue execution
    print("\n▶️ Resuming execution...")
    for event in graph.stream(None, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()

    print("\n✅ Execution complete!")


def run_approval_workflow_example():
    """
    Example 2: Human approval workflow.

    Pattern:
    1. Run to breakpoint
    2. Ask user for approval
    3. If approved -> continue
    4. If rejected -> cancel
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Human Approval Workflow")
    print("="*70)

    # Input
    initial_input = {"messages": HumanMessage(content="Multiply 5 and 7")}
    thread = {"configurable": {"thread_id": "2"}}

    # Run to breakpoint
    print("\n📍 Running until breakpoint...")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()

    # Get state to see what tool will be called
    state = graph.get_state(thread)
    last_message = state.values['messages'][-1]

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        tool_call = last_message.tool_calls[0]
        print(f"\n⚠️ Agent wants to call: {tool_call['name']}({tool_call['args']})")

    # Simulate user approval (in real app, use input())
    # user_approval = input("\n🤔 Do you want to call the tool? (yes/no): ")
    user_approval = "yes"  # For demo purposes
    print(f"\n🤔 User decision: {user_approval}")

    if user_approval.lower() == "yes":
        print("\n✅ Approved! Continuing execution...")
        for event in graph.stream(None, thread, stream_mode="values"):
            event['messages'][-1].pretty_print()
        print("\n✅ Task completed!")
    else:
        print("\n❌ Operation cancelled by user.")


def run_multi_approval_example():
    """
    Example 3: Multiple approval points.

    Since we have interrupt_before=["tools"] and the graph loops
    tools -> assistant -> tools, each tool call requires approval.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Multiple Approval Points")
    print("="*70)

    # Complex query that might need multiple tool calls
    initial_input = {
        "messages": HumanMessage(content="Multiply 3 and 4, then add 10 to the result")
    }
    thread = {"configurable": {"thread_id": "3"}}

    print("\n📍 Starting complex calculation with multiple approvals...")

    # First run
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()

    state = graph.get_state(thread)
    if state.next:
        print(f"\n🛑 Paused at: {state.next}")
        print("✅ Approving first tool call...")

        # Continue
        for event in graph.stream(None, thread, stream_mode="values"):
            event['messages'][-1].pretty_print()

        # Check if there's another breakpoint
        state = graph.get_state(thread)
        if state.next:
            print(f"\n🛑 Paused again at: {state.next}")
            print("✅ Approving second tool call...")

            # Continue again
            for event in graph.stream(None, thread, stream_mode="values"):
                event['messages'][-1].pretty_print()

    print("\n✅ All approvals completed!")


# =============================================================================
# KEY LEARNINGS
# =============================================================================

def print_key_concepts():
    """Print the key concepts for breakpoints."""
    print("\n" + "="*70)
    print("📚 KEY CONCEPTS: Breakpoints")
    print("="*70)

    concepts = """
    1. BREAKPOINT TYPES:
       - interrupt_before=["node"]: Stop BEFORE node executes
       - interrupt_after=["node"]: Stop AFTER node executes

    2. RESUMING EXECUTION:
       - graph.stream(None, thread): Continue from last checkpoint
       - Uses the SAME thread ID to access saved state

    3. CHECKING STATE:
       - state = graph.get_state(thread)
       - state.next: Shows which node will execute next
       - state.values: Shows current state data

    4. CHECKPOINTER REQUIRED:
       - MemorySaver() saves state between interruptions
       - Without checkpointer, breakpoints won't work
       - Each thread ID has its own state

    5. APPROVAL PATTERNS:
       - Pause before dangerous operations
       - Surface state to user for review
       - User decides: approve or reject
       - Conditional continuation based on user input

    6. PRODUCTION USE CASES:
       - Tool execution approval (prevent unwanted actions)
       - Cost control (review before expensive operations)
       - Compliance (legal/regulatory review)
       - Debugging (inspect state at specific points)
    """
    print(concepts)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "🚀"*35)
    print("MODULE 3 - LESSON 1: BREAKPOINTS FOR HUMAN-IN-THE-LOOP")
    print("🚀"*35)

    # Run examples
    run_basic_breakpoint_example()
    run_approval_workflow_example()
    run_multi_approval_example()

    # Show key concepts
    print_key_concepts()

    print("\n" + "="*70)
    print("✅ Module 3, Lesson 1 Complete!")
    print("="*70)
    print("\n📖 Next: edit_state_human_feedback.py - Learn to modify state at breakpoints")
    print()
