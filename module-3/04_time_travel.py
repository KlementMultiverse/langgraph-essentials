"""
Module 3 - Lesson 4: Time Travel

This module demonstrates LangGraph's time travel capabilities - browsing history,
replaying execution, and forking from past states.

Key Concepts:
- Checkpoint history navigation
- Replaying from any checkpoint
- Forking: modifying past state and continuing
- Complete execution timeline access
- Non-destructive error recovery

Learning Outcomes:
✓ Browse complete execution history
✓ Replay from any point in time
✓ Fork and create alternate timelines
✓ Debug by reproducing exact states
✓ Implement error recovery patterns

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
# GRAPH SETUP
# =============================================================================

tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)


def assistant(state: MessagesState):
    """Assistant node that invokes LLM with tools."""
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# Compile with checkpointer (required for time travel!)
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# =============================================================================
# PART 1: BROWSING HISTORY
# =============================================================================

def run_browsing_history_example():
    """
    Example 1: Browse complete execution history.

    Demonstrates:
    - get_state() for current state
    - get_state_history() for all checkpoints
    - Checkpoint structure and metadata
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Browsing Execution History")
    print("="*70)

    # Run a complete execution
    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}
    thread = {"configurable": {"thread_id": "1"}}

    print("\n📍 Running agent...")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()

    # Get current state
    print("\n" + "="*70)
    print("📊 CURRENT STATE")
    print("="*70)

    current_state = graph.get_state(thread)
    print(f"\n✓ Next nodes: {current_state.next}")
    print(f"✓ Number of messages: {len(current_state.values['messages'])}")
    print(f"✓ Checkpoint ID: {current_state.config['configurable']['checkpoint_id'][:20]}...")
    print(f"✓ Step: {current_state.metadata.get('step', 'N/A')}")

    # Get ALL historical states
    print("\n" + "="*70)
    print("📜 EXECUTION HISTORY")
    print("="*70)

    all_states = list(graph.get_state_history(thread))
    print(f"\n✓ Total checkpoints: {len(all_states)}")

    # Display each checkpoint
    print("\n📝 Checkpoint Timeline (newest to oldest):")
    for i, state in enumerate(all_states):
        step = state.metadata.get('step', 0)
        msg_count = len(state.values.get('messages', []))
        next_node = state.next[0] if state.next else "END"
        checkpoint_id = state.config['configurable']['checkpoint_id'][:12]

        print(f"\n  [{i}] Step {step} | Checkpoint: {checkpoint_id}...")
        print(f"      Messages: {msg_count} | Next: {next_node}")

        # Show last message content
        if state.values.get('messages'):
            last_msg = state.values['messages'][-1]
            content_preview = str(last_msg.content)[:50]
            print(f"      Last message: {last_msg.__class__.__name__}: {content_preview}...")

    # Inspect a specific checkpoint in detail
    print("\n" + "="*70)
    print("🔍 DETAILED CHECKPOINT INSPECTION")
    print("="*70)

    user_input_checkpoint = all_states[-2]  # Second from end = user input
    print(f"\n📌 Inspecting: User Input Checkpoint")
    print(f"\n✓ Values: {user_input_checkpoint.values}")
    print(f"✓ Next: {user_input_checkpoint.next}")
    print(f"✓ Config: {user_input_checkpoint.config}")
    print(f"✓ Parent: {user_input_checkpoint.parent_config}")

    return all_states


# =============================================================================
# PART 2: REPLAYING EXECUTION
# =============================================================================

def run_replay_example(all_states):
    """
    Example 2: Replay execution from a past checkpoint.

    Demonstrates:
    - Using checkpoint config to replay
    - Graph knows checkpoint was already executed
    - Exact reproduction of past execution
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Replaying from Checkpoint")
    print("="*70)

    # Select a checkpoint to replay from
    to_replay = all_states[-2]  # User input checkpoint

    print(f"\n📍 Replaying from checkpoint:")
    print(f"   State: {to_replay.values}")
    print(f"   Next: {to_replay.next}")
    print(f"   Checkpoint ID: {to_replay.config['configurable']['checkpoint_id'][:20]}...")

    # Replay execution
    print("\n▶️ Starting replay...")
    print("-" * 70)

    for event in graph.stream(None, to_replay.config, stream_mode="values"):
        event['messages'][-1].pretty_print()

    print("-" * 70)
    print("\n✅ Replay complete!")
    print("\n💡 Note: Graph re-executed from the checkpoint with same inputs")


# =============================================================================
# PART 3: FORKING - CREATING ALTERNATE TIMELINES
# =============================================================================

def run_fork_example(all_states):
    """
    Example 3: Fork execution - modify past state and continue.

    Demonstrates:
    - Modifying state at a checkpoint
    - Creating new execution branch
    - Message ID for overwriting vs appending
    - Multiple timelines coexisting
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Forking - Creating Alternate Timeline")
    print("="*70)

    # Select checkpoint to fork from
    to_fork = all_states[-2]
    original_message = to_fork.values["messages"][0]

    print(f"\n📍 Forking from checkpoint:")
    print(f"   Original input: '{original_message.content}'")
    print(f"   Message ID: {original_message.id}")

    # Modify state - use SAME message ID to overwrite
    print(f"\n✏️ Modifying state:")
    print(f"   New input: 'Multiply 5 and 3'")
    print(f"   Using same message ID → will OVERWRITE not APPEND")

    fork_config = graph.update_state(
        to_fork.config,
        {
            "messages": [
                HumanMessage(
                    content='Multiply 5 and 3',
                    id=original_message.id  # SAME ID = overwrite
                )
            ]
        },
    )

    print(f"\n✅ Fork created!")
    print(f"   New checkpoint ID: {fork_config['configurable']['checkpoint_id'][:20]}...")

    # Verify the fork
    thread = {"configurable": {"thread_id": to_fork.config['configurable']['thread_id']}}
    current_state = graph.get_state(thread)

    print(f"\n🔍 Current state after fork:")
    print(f"   Input changed to: '{current_state.values['messages'][0].content}'")

    # Execute the fork
    print("\n▶️ Executing forked timeline...")
    print("-" * 70)

    for event in graph.stream(None, fork_config, stream_mode="values"):
        event['messages'][-1].pretty_print()

    print("-" * 70)
    print("\n✅ Fork execution complete!")
    print("\n💡 Note: Created NEW timeline - original still exists in history!")


# =============================================================================
# PART 4: ADVANCED - MULTIPLE FORKS
# =============================================================================

def run_multiple_forks_example():
    """
    Example 4: Create multiple forks from same checkpoint.

    Demonstrates:
    - Multiple alternate timelines
    - A/B testing scenarios
    - Comparing different approaches
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Multiple Forks (A/B Testing)")
    print("="*70)

    # Fresh execution
    initial_input = {"messages": HumanMessage(content="Multiply 2 and 2")}
    thread = {"configurable": {"thread_id": "2"}}

    print("\n📍 Running original execution...")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        pass  # Silent execution

    # Get checkpoint to fork from
    all_states = list(graph.get_state_history(thread))
    base_checkpoint = all_states[-2]

    print(f"\n✓ Base execution complete")
    print(f"   Input: 'Multiply 2 and 2'")
    print(f"   Result: 4")

    # Fork A: Different operation
    print(f"\n🔀 Creating Fork A: 'Add 10 and 5'")
    fork_a_config = graph.update_state(
        base_checkpoint.config,
        {
            "messages": [
                HumanMessage(
                    content='Add 10 and 5',
                    id=base_checkpoint.values["messages"][0].id
                )
            ]
        }
    )

    print("▶️ Executing Fork A...")
    for event in graph.stream(None, fork_a_config, stream_mode="values"):
        pass

    fork_a_state = graph.get_state(thread)
    fork_a_result = fork_a_state.values['messages'][-1].content
    print(f"✅ Fork A Result: {fork_a_result}")

    # Fork B: Another different operation
    print(f"\n🔀 Creating Fork B: 'Divide 100 by 5'")
    fork_b_config = graph.update_state(
        base_checkpoint.config,
        {
            "messages": [
                HumanMessage(
                    content='Divide 100 by 5',
                    id=base_checkpoint.values["messages"][0].id
                )
            ]
        }
    )

    print("▶️ Executing Fork B...")
    for event in graph.stream(None, fork_b_config, stream_mode="values"):
        pass

    fork_b_state = graph.get_state(thread)
    fork_b_result = fork_b_state.values['messages'][-1].content
    print(f"✅ Fork B Result: {fork_b_result}")

    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY: Three Timelines from One Checkpoint")
    print("="*70)
    print(f"\n  Original: 'Multiply 2 and 2' → 4")
    print(f"  Fork A:   'Add 10 and 5' → 15")
    print(f"  Fork B:   'Divide 100 by 5' → 20")
    print("\n💡 All three timelines exist independently!")


# =============================================================================
# PART 5: ERROR RECOVERY PATTERN
# =============================================================================

def run_error_recovery_example():
    """
    Example 5: Error recovery using time travel.

    Demonstrates:
    - Detecting problematic execution
    - Forking from before the error
    - Correcting and continuing
    - Production-ready recovery pattern
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Error Recovery Pattern")
    print("="*70)

    # Simulate problematic execution
    initial_input = {"messages": HumanMessage(content="Divide 10 by 0")}
    thread = {"configurable": {"thread_id": "3"}}

    print("\n📍 Simulating problematic request...")
    print("   Request: 'Divide 10 by 0'")
    print("   (This would cause division by zero error)")

    # In real scenario, this might fail or produce unexpected result
    try:
        for event in graph.stream(initial_input, thread, stream_mode="values"):
            pass

        print("\n⚠️ Execution completed but result may be problematic")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")

    # Recovery: Go back and fix
    print("\n🔧 Recovery Strategy:")
    print("   1. Get checkpoint before error")
    print("   2. Fork with corrected input")
    print("   3. Continue execution")

    all_states = list(graph.get_state_history(thread))
    if all_states:
        before_error = all_states[-2]

        print(f"\n✏️ Correcting input to: 'Divide 10 by 2'")

        recovery_config = graph.update_state(
            before_error.config,
            {
                "messages": [
                    HumanMessage(
                        content='Divide 10 by 2',
                        id=before_error.values["messages"][0].id
                    )
                ]
            }
        )

        print("\n▶️ Executing recovery...")
        for event in graph.stream(None, recovery_config, stream_mode="values"):
            event['messages'][-1].pretty_print()

        print("\n✅ Recovery successful! Problem avoided.")


# =============================================================================
# KEY LEARNINGS
# =============================================================================

def print_key_concepts():
    """Print key concepts for time travel."""
    print("\n" + "="*70)
    print("📚 KEY CONCEPTS: Time Travel")
    print("="*70)

    concepts = """
    1. CHECKPOINTS ARE SAVE POINTS:
       - Every step creates a checkpoint
       - Complete state snapshot saved
       - Includes: state, config, metadata, next nodes
       - Requires: checkpointer=MemorySaver()

    2. BROWSING HISTORY:
       - get_state(config): Current state
       - get_state_history(config): All checkpoints
       - Navigate timeline like Git log
       - Inspect any past execution point

    3. REPLAYING EXECUTION:
       - graph.stream(None, checkpoint.config)
       - Re-runs from that exact point
       - Same inputs → same results
       - Use for: debugging, reproduction, analysis

    4. FORKING - CREATING ALTERNATE TIMELINES:
       - update_state() at a checkpoint
       - Modify state, then continue
       - Creates NEW checkpoint branch
       - Original timeline preserved

    5. MESSAGE ID BEHAVIOR:
       - WITH same ID: OVERWRITES message
       - WITHOUT ID: APPENDS message
       - Critical for fork control!

    6. CHECKPOINT STRUCTURE:
       - values: State data
       - next: Next nodes to execute
       - config: thread_id + checkpoint_id
       - metadata: Step number, writes, source
       - parent_config: Previous checkpoint

    7. PRODUCTION PATTERNS:
       - Error recovery: Fork from before error
       - A/B testing: Multiple forks, compare results
       - User corrections: Fork with corrected input
       - Debugging: Replay exact failing execution
       - Testing: Verify fixes with historical data

    8. TIMELINE MANAGEMENT:
       - Each fork creates new branch
       - All branches coexist
       - Current state = latest branch
       - Can switch between timelines

    9. MENTAL MODEL: GIT FOR EXECUTION
       - Checkpoints = Git commits
       - Replay = git checkout <commit>
       - Fork = git checkout -b new-branch <commit>
       - History = git log
       - Non-destructive exploration

    10. REQUIREMENTS:
        ✓ Must have checkpointer
        ✓ Must use thread_id
        ✓ Checkpoint IDs are unique
        ✓ Parent-child relationships tracked
    """
    print(concepts)


def print_comparison_table():
    """Print comparison of time travel operations."""
    print("\n" + "="*70)
    print("📊 COMPARISON: Time Travel Operations")
    print("="*70)

    comparison = """
    ┌─────────────────┬──────────────────┬──────────────────────┐
    │   Operation     │   State Change   │   Use Case           │
    ├─────────────────┼──────────────────┼──────────────────────┤
    │ Browse History  │   None           │   Inspect past       │
    │ Replay          │   None           │   Reproduce bugs     │
    │ Fork            │   Modified       │   Error recovery     │
    │                 │                  │   A/B testing        │
    └─────────────────┴──────────────────┴──────────────────────┘

    REPLAY vs FORK:

    Replay:
    - Uses original state unchanged
    - Same inputs → same outputs
    - Doesn't create new checkpoints (re-runs existing)
    - Good for: reproduction, analysis

    Fork:
    - Modifies state at checkpoint
    - Different inputs → different outputs
    - Creates NEW checkpoint branch
    - Good for: corrections, alternatives, recovery
    """
    print(comparison)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "🚀"*35)
    print("MODULE 3 - LESSON 4: TIME TRAVEL")
    print("🚀"*35)
    print("\n💡 Concept: Like Git for your agent execution!")
    print("   - Browse history")
    print("   - Replay any checkpoint")
    print("   - Fork and create alternate timelines")

    # Part 1: Browse history
    all_states = run_browsing_history_example()

    # Part 2: Replay
    run_replay_example(all_states)

    # Part 3: Fork
    run_fork_example(all_states)

    # Part 4: Multiple forks
    run_multiple_forks_example()

    # Part 5: Error recovery
    run_error_recovery_example()

    # Show key concepts
    print_key_concepts()
    print_comparison_table()

    print("\n" + "="*70)
    print("✅ Module 3, Lesson 4 Complete!")
    print("="*70)
    print("\n🎓 Achievement Unlocked: Time Travel Master!")
    print("\n📖 Module 3 Complete! Next: Module 4 - Parallelization")
    print()
