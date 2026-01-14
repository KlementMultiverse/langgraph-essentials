"""
Module 3 - Lesson 2: Editing State and Human Feedback

This module demonstrates how to modify graph state at breakpoints and
inject human feedback into the agent workflow.

Key Concepts:
- Editing state with graph.update_state()
- add_messages reducer behavior
- Human feedback nodes
- as_node parameter for state updates
- Interactive correction loops

Learning Outcomes:
✓ Modify state after breakpoints
✓ Inject human corrections
✓ Build feedback loops
✓ Create production-grade interactive agents

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
    """Multiply a and b."""
    return a * b


def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b


def divide(a: int, b: int) -> float:
    """Divide a by b."""
    return a / b


# =============================================================================
# GRAPH 1: EDITING STATE BEFORE ASSISTANT
# =============================================================================

# Initialize tools and LLM
tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)


def assistant(state: MessagesState):
    """Assistant node that invokes LLM with tools."""
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# Build graph with interrupt BEFORE assistant
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
graph_edit_before = builder.compile(
    interrupt_before=["assistant"],  # ⚠️ Interrupt BEFORE assistant processes
    checkpointer=memory
)


# =============================================================================
# GRAPH 2: HUMAN FEEDBACK NODE
# =============================================================================

def human_feedback(state: MessagesState):
    """
    No-op node that serves as a placeholder for human feedback.

    This node does nothing - it's just a marker in the graph where
    we interrupt for human input.
    """
    pass


# Build graph with human feedback node
builder2 = StateGraph(MessagesState)
builder2.add_node("assistant", assistant)
builder2.add_node("tools", ToolNode(tools))
builder2.add_node("human_feedback", human_feedback)

# Flow: START -> human_feedback -> assistant -> tools -> human_feedback (loop)
builder2.add_edge(START, "human_feedback")
builder2.add_edge("human_feedback", "assistant")
builder2.add_conditional_edges("assistant", tools_condition)
builder2.add_edge("tools", "human_feedback")

memory2 = MemorySaver()
graph_with_feedback = builder2.compile(
    interrupt_before=["human_feedback"],  # ⚠️ Interrupt at feedback points
    checkpointer=memory2
)


# =============================================================================
# EXECUTION PATTERNS
# =============================================================================

def run_edit_state_example():
    """
    Example 1: Edit state before assistant processes it.

    Pattern:
    1. Run to breakpoint (before assistant)
    2. User input arrives
    3. Update state with correction
    4. Resume - assistant sees corrected input
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Edit State Before Processing")
    print("="*70)

    initial_input = {"messages": "Multiply 2 and 3"}
    thread = {"configurable": {"thread_id": "1"}}

    # Run to breakpoint
    print("\n📍 Running to breakpoint (before assistant)...")
    for event in graph_edit_before.stream(initial_input, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()

    # Check state
    state = graph_edit_before.get_state(thread)
    print(f"\n🛑 Paused at: {state.next}")
    print(f"📊 Current messages: {len(state.values['messages'])}")

    # User realizes they want to change the input
    print("\n✏️ User wants to change the request...")
    print("   Original: 'Multiply 2 and 3'")
    print("   Corrected: 'No, actually multiply 3 and 3!'")

    # Update state - adds a new message
    graph_edit_before.update_state(
        thread,
        {"messages": [HumanMessage(content="No, actually multiply 3 and 3!")]}
    )

    # Check updated state
    new_state = graph_edit_before.get_state(thread).values
    print(f"\n📊 Updated messages: {len(new_state['messages'])}")
    print("\n📝 Message history:")
    for i, m in enumerate(new_state['messages'], 1):
        print(f"   {i}. {m.content[:50]}")

    # Resume execution
    print("\n▶️ Resuming - assistant will see corrected input...")
    for event in graph_edit_before.stream(None, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()

    print("\n✅ Agent used the corrected values (3 and 3)!")


def run_overwrite_message_example():
    """
    Example 2: Overwrite a message using its ID.

    Pattern:
    - If message has ID: add_messages reducer REPLACES it
    - If no ID: add_messages reducer APPENDS it
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Overwrite vs Append Messages")
    print("="*70)

    initial_input = {"messages": "Multiply 5 and 6"}
    thread = {"configurable": {"thread_id": "2"}}

    # Run to breakpoint
    print("\n📍 Running to breakpoint...")
    for event in graph_edit_before.stream(initial_input, thread, stream_mode="values"):
        pass  # Silent

    # Get the last message with its ID
    state = graph_edit_before.get_state(thread)
    last_message = state.values['messages'][-1]

    print(f"\n📨 Original message:")
    print(f"   ID: {last_message.id}")
    print(f"   Content: {last_message.content}")

    # Create a new message with SAME ID (will overwrite)
    print("\n✏️ Overwriting message (using same ID)...")
    last_message.content = "Actually, multiply 7 and 8!"

    graph_edit_before.update_state(
        thread,
        {"messages": [last_message]}  # Has ID -> will replace
    )

    # Check result
    new_state = graph_edit_before.get_state(thread).values
    print(f"\n📊 Total messages: {len(new_state['messages'])}")
    print("   (Same count - message was replaced, not appended)")

    print("\n📝 Message content:")
    for m in new_state['messages']:
        print(f"   - {m.content}")

    print("\n▶️ Resuming execution...")
    for event in graph_edit_before.stream(None, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()


def run_human_feedback_node_example():
    """
    Example 3: Human feedback node pattern.

    Pattern:
    1. Graph starts at human_feedback node
    2. User can inject instructions
    3. Assistant processes
    4. Tools execute
    5. Back to human_feedback (loop)

    This creates a continuous feedback loop!
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Human Feedback Node Pattern")
    print("="*70)

    initial_input = {"messages": "Multiply 4 and 5"}
    thread = {"configurable": {"thread_id": "3"}}

    # First run
    print("\n📍 Starting with human feedback node...")
    for event in graph_with_feedback.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    state = graph_with_feedback.get_state(thread)
    print(f"\n🛑 Paused at: {state.next}")

    # Simulate user feedback
    print("\n💬 User provides feedback: 'Actually, use 6 and 7'")

    # Update state AS IF it came from human_feedback node
    graph_with_feedback.update_state(
        thread,
        {"messages": "no, multiply 6 and 7"},
        as_node="human_feedback"  # ⚠️ KEY: Pretend this came from feedback node
    )

    # Continue execution
    print("\n▶️ Continuing with user's feedback...")
    for event in graph_with_feedback.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # At next feedback point
    state = graph_with_feedback.get_state(thread)
    if state.next and "human_feedback" in state.next:
        print(f"\n🛑 Back at feedback node: {state.next}")
        print("   User can provide more feedback or continue")

        # User approves - no changes
        print("\n✅ User approves - continuing without changes...")
        graph_with_feedback.update_state(
            thread,
            {"messages": []},  # No changes
            as_node="human_feedback"
        )

        for event in graph_with_feedback.stream(None, thread, stream_mode="values"):
            event["messages"][-1].pretty_print()

    print("\n✅ Interactive feedback loop complete!")


def run_interactive_agent_pattern():
    """
    Example 4: Production-ready interactive agent pattern.

    This shows a complete interactive loop with multiple options.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Production Interactive Agent")
    print("="*70)

    initial_input = {"messages": "Multiply 8 and 9"}
    thread = {"configurable": {"thread_id": "4"}}

    print("\n🤖 Starting interactive agent...")
    print("   (In production, this would have a UI)")

    # Initial run
    for event in graph_with_feedback.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # Simulate multiple user interactions
    interactions = [
        ("continue", None),  # Just continue
        ("modify", "Actually add 10 to the result"),  # Add instruction
    ]

    for action, instruction in interactions:
        state = graph_with_feedback.get_state(thread)

        if not state.next:
            print("\n✅ Task complete!")
            break

        print(f"\n🛑 Paused at: {state.next}")
        print(f"   User action: {action}")

        if action == "continue":
            # No changes, just continue
            graph_with_feedback.update_state(
                thread,
                {"messages": []},
                as_node="human_feedback"
            )
        elif action == "modify":
            # Add new instruction
            print(f"   New instruction: {instruction}")
            graph_with_feedback.update_state(
                thread,
                {"messages": [HumanMessage(content=instruction)]},
                as_node="human_feedback"
            )

        # Continue execution
        print("\n▶️ Continuing...")
        for event in graph_with_feedback.stream(None, thread, stream_mode="values"):
            event["messages"][-1].pretty_print()

    print("\n✅ Interactive session complete!")


# =============================================================================
# KEY LEARNINGS
# =============================================================================

def print_key_concepts():
    """Print key concepts for state editing and human feedback."""
    print("\n" + "="*70)
    print("📚 KEY CONCEPTS: State Editing & Human Feedback")
    print("="*70)

    concepts = """
    1. UPDATE_STATE BASICS:
       - graph.update_state(thread, {"messages": [msg]})
       - Uses reducer to merge with existing state
       - For messages: add_messages reducer

    2. ADD_MESSAGES REDUCER:
       - Without message ID: APPENDS to message list
       - With message ID: REPLACES message with that ID
       - Allows both correction and addition

    3. AS_NODE PARAMETER:
       - update_state(..., as_node="node_name")
       - Tells graph which node "sent" the update
       - Determines next node in execution flow
       - Critical for human feedback nodes!

    4. HUMAN FEEDBACK NODE:
       - No-op placeholder: def human_feedback(state): pass
       - Marker for human interaction points
       - Set interrupt_before=["human_feedback"]
       - Creates continuous feedback loops

    5. INTERRUPT POSITIONS:
       - Before assistant: Edit user input before LLM sees it
       - Before tools: Review tool calls before execution
       - At feedback node: Structured interaction points

    6. PRODUCTION PATTERNS:
       - Interactive correction loops
       - Multi-step approval workflows
       - Context injection during execution
       - User-guided agent behavior

    7. STATE INSPECTION:
       - state.values: Current state data
       - state.next: Next nodes to execute
       - state.values['messages']: Message history
       - Use to inform user decisions
    """
    print(concepts)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "🚀"*35)
    print("MODULE 3 - LESSON 2: EDITING STATE & HUMAN FEEDBACK")
    print("🚀"*35)

    # Run examples
    run_edit_state_example()
    run_overwrite_message_example()
    run_human_feedback_node_example()
    run_interactive_agent_pattern()

    # Show key concepts
    print_key_concepts()

    print("\n" + "="*70)
    print("✅ Module 3, Lesson 2 Complete!")
    print("="*70)
    print("\n📖 Next: dynamic_breakpoints.py - Learn conditional interrupts")
    print()
