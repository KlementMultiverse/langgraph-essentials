"""
Module 3 - Lesson 3: Dynamic Breakpoints

This module demonstrates how to create conditional interrupts that trigger
based on runtime conditions using NodeInterrupt.

Key Concepts:
- NodeInterrupt for dynamic interruptions
- Conditional logic within nodes
- Custom interrupt messages
- Fixing conditions to resume execution
- Smart approval workflows

Learning Outcomes:
✓ Create conditional interrupts
✓ Pass context in interrupt messages
✓ Build intelligent approval systems
✓ Implement production-grade conditional workflows

Author: Klement G
Date: 2026-01-13
Course: LangChain Academy - Introduction to LangGraph
"""

from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
from langgraph.graph import START, END, StateGraph


# =============================================================================
# STATE DEFINITION
# =============================================================================

class State(TypedDict):
    """Simple state with input string."""
    input: str


# =============================================================================
# NODES WITH CONDITIONAL INTERRUPTS
# =============================================================================

def step_1(state: State) -> State:
    """First processing step - no interrupts."""
    print("---Step 1---")
    print(f"   Processing: {state['input']}")
    return state


def step_2(state: State) -> State:
    """
    Second processing step with CONDITIONAL interrupt.

    This demonstrates NodeInterrupt - the node itself decides
    whether to interrupt based on runtime conditions.

    Business rule: Only process inputs <= 5 characters
    """
    input_text = state['input']

    # CONDITIONAL INTERRUPT ⚠️
    if len(input_text) > 5:
        raise NodeInterrupt(
            f"Received input that is longer than 5 characters: {input_text}"
        )

    print("---Step 2---")
    print(f"   Input passed validation: {input_text}")
    return state


def step_3(state: State) -> State:
    """Final processing step."""
    print("---Step 3---")
    print(f"   Completed: {state['input']}")
    return state


# =============================================================================
# GRAPH SETUP
# =============================================================================

# Build the graph
builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)

builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

# Compile with memory (NO interrupt_before needed!)
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# =============================================================================
# ADVANCED EXAMPLES: PRODUCTION PATTERNS
# =============================================================================

class ApprovalState(TypedDict):
    """State for approval workflow."""
    amount: float
    user_id: str
    approved: bool


def validate_transaction(state: ApprovalState) -> ApprovalState:
    """
    Validate transaction with conditional approval.

    Business rules:
    - Amount > $1000: Requires approval
    - Amount <= $1000: Auto-approved
    """
    amount = state['amount']
    user = state['user_id']

    # Dynamic approval threshold
    if amount > 1000:
        raise NodeInterrupt(
            f"Transaction of ${amount:.2f} by user {user} requires manager approval. "
            f"Exceeds threshold of $1000."
        )

    print(f"✅ Transaction ${amount:.2f} auto-approved (under threshold)")
    return {"approved": True}


def execute_transaction(state: ApprovalState) -> ApprovalState:
    """Execute the approved transaction."""
    print(f"💰 Executing transaction: ${state['amount']:.2f}")
    return state


# Build approval workflow graph
approval_builder = StateGraph(ApprovalState)
approval_builder.add_node("validate", validate_transaction)
approval_builder.add_node("execute", execute_transaction)

approval_builder.add_edge(START, "validate")
approval_builder.add_edge("validate", "execute")
approval_builder.add_edge("execute", END)

approval_graph = approval_builder.compile(checkpointer=MemorySaver())


# =============================================================================
# EXECUTION PATTERNS
# =============================================================================

def run_basic_dynamic_interrupt():
    """
    Example 1: Basic dynamic interrupt with long input.

    Pattern:
    1. Input exceeds threshold
    2. Node conditionally interrupts
    3. State must be fixed to continue
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Dynamic Interrupt - Long Input")
    print("="*70)

    initial_input = {"input": "hello world"}  # 11 characters > 5
    thread_config = {"configurable": {"thread_id": "1"}}

    print(f"\n📍 Processing input: '{initial_input['input']}' ({len(initial_input['input'])} chars)")

    # Run until interruption
    for event in graph.stream(initial_input, thread_config, stream_mode="values"):
        print(f"   Event: {event}")

    # Check state
    state = graph.get_state(thread_config)
    print(f"\n🛑 Graph interrupted at: {state.next}")

    # Check interrupt details
    if state.tasks:
        task = state.tasks[0]
        if task.interrupts:
            interrupt = task.interrupts[0]
            print(f"\n⚠️ Interrupt message:")
            print(f"   {interrupt.value}")

    # Try to resume (won't work - condition still true)
    print("\n❌ Attempting to resume without fixing condition...")
    for event in graph.stream(None, thread_config, stream_mode="values"):
        pass

    state = graph.get_state(thread_config)
    print(f"   Still stuck at: {state.next}")
    print("   (Condition not fixed - will interrupt again)")

    # Fix the condition
    print("\n✏️ Fixing condition: Updating to short input...")
    graph.update_state(thread_config, {"input": "hi"})

    # Resume successfully
    print("\n✅ Resuming with fixed input...")
    for event in graph.stream(None, thread_config, stream_mode="values"):
        print(f"   Event: {event}")

    print("\n✅ Execution complete!")


def run_short_input_no_interrupt():
    """
    Example 2: Short input that doesn't trigger interrupt.

    Demonstrates that interrupt is conditional - only happens when needed.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Short Input - No Interrupt")
    print("="*70)

    initial_input = {"input": "hi"}  # 2 characters <= 5
    thread_config = {"configurable": {"thread_id": "2"}}

    print(f"\n📍 Processing input: '{initial_input['input']}' ({len(initial_input['input'])} chars)")

    # Run to completion (no interrupt)
    for event in graph.stream(initial_input, thread_config, stream_mode="values"):
        print(f"   Event: {event}")

    state = graph.get_state(thread_config)
    print(f"\n✅ Completed without interruption!")
    print(f"   Final state: {state.values}")
    print(f"   Next: {state.next if state.next else 'DONE'}")


def run_approval_workflow_under_threshold():
    """
    Example 3: Transaction under threshold - auto-approved.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Transaction Under Threshold")
    print("="*70)

    # Small transaction
    initial_input = {
        "amount": 500.00,
        "user_id": "user_123",
        "approved": False
    }
    thread_config = {"configurable": {"thread_id": "3"}}

    print(f"\n💳 Transaction request:")
    print(f"   Amount: ${initial_input['amount']:.2f}")
    print(f"   User: {initial_input['user_id']}")

    # Run to completion
    for event in approval_graph.stream(initial_input, thread_config, stream_mode="values"):
        pass

    state = approval_graph.get_state(thread_config)
    print(f"\n✅ Transaction completed!")
    print(f"   Status: {'Approved' if state.values.get('approved') else 'Pending'}")


def run_approval_workflow_over_threshold():
    """
    Example 4: Transaction over threshold - requires approval.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Transaction Over Threshold")
    print("="*70)

    # Large transaction
    initial_input = {
        "amount": 5000.00,
        "user_id": "user_456",
        "approved": False
    }
    thread_config = {"configurable": {"thread_id": "4"}}

    print(f"\n💳 Transaction request:")
    print(f"   Amount: ${initial_input['amount']:.2f}")
    print(f"   User: {initial_input['user_id']}")

    # Run until interrupt
    for event in approval_graph.stream(initial_input, thread_config, stream_mode="values"):
        pass

    state = approval_graph.get_state(thread_config)
    print(f"\n🛑 Transaction requires approval!")
    print(f"   Paused at: {state.next}")

    # Check interrupt message
    if state.tasks and state.tasks[0].interrupts:
        interrupt_msg = state.tasks[0].interrupts[0].value
        print(f"\n⚠️ Approval required:")
        print(f"   {interrupt_msg}")

    # Simulate manager approval
    print("\n👔 Manager reviewing transaction...")
    print("   Decision: Approved ✅")

    # Update state to approve
    approval_graph.update_state(
        thread_config,
        {"approved": True, "amount": initial_input["amount"]}
    )

    # Continue execution
    print("\n▶️ Continuing with approval...")
    for event in approval_graph.stream(None, thread_config, stream_mode="values"):
        pass

    print("\n✅ Transaction completed after approval!")


def run_comparison_static_vs_dynamic():
    """
    Example 5: Compare static vs dynamic breakpoints.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Static vs Dynamic Breakpoints")
    print("="*70)

    comparison = """
    STATIC BREAKPOINT (interrupt_before):
    ✓ Simple to implement
    ✓ Predictable - always stops at same place
    ✗ No conditional logic
    ✗ Interrupts EVERY execution
    ✗ Can't explain why

    Example:
        graph.compile(interrupt_before=["validate"])

    Use when:
    - Fixed approval points
    - Always need human review
    - Simple workflows


    DYNAMIC BREAKPOINT (NodeInterrupt):
    ✓ Conditional - only when needed
    ✓ Custom explanatory messages
    ✓ Can include runtime data
    ✓ Smart, context-aware
    ✗ More complex to implement

    Example:
        if amount > threshold:
            raise NodeInterrupt(f"Needs approval: ${amount}")

    Use when:
    - Conditional approval needed
    - Business rules determine interrupt
    - Want to explain WHY
    - Most executions should flow through


    REAL-WORLD EXAMPLES:

    1. Content Moderation:
       - Static: Review every post
       - Dynamic: Only review if toxicity > 0.7

    2. Financial Transactions:
       - Static: Approve every transfer
       - Dynamic: Only if amount > $1000

    3. Code Deployment:
       - Static: Review every deployment
       - Dynamic: Only if changes > 100 lines or critical files

    4. API Calls:
       - Static: Approve every external call
       - Dynamic: Only if rate limit exceeded or high cost
    """

    print(comparison)


# =============================================================================
# KEY LEARNINGS
# =============================================================================

def print_key_concepts():
    """Print key concepts for dynamic breakpoints."""
    print("\n" + "="*70)
    print("📚 KEY CONCEPTS: Dynamic Breakpoints")
    print("="*70)

    concepts = """
    1. NODE_INTERRUPT BASICS:
       - raise NodeInterrupt("message")
       - Raised FROM WITHIN a node
       - Based on runtime conditions
       - No interrupt_before needed!

    2. CONDITIONAL LOGIC:
       - Any Python condition: if, match, try/except
       - Check state values, external APIs, databases
       - Business rules determine interrupts
       - Most executions flow through normally

    3. INTERRUPT MESSAGES:
       - Pass any string to explain WHY
       - Include runtime data: amounts, thresholds, etc.
       - Retrieved via: state.tasks[0].interrupts[0].value
       - Helps user understand what needs review

    4. RESUMING EXECUTION:
       - graph.stream(None, thread) retries the node
       - If condition still true → interrupts again
       - Must update state to fix condition
       - Then resume successfully

    5. STATE INSPECTION:
       - state.tasks: Check for active tasks
       - state.tasks[0].interrupts: Get interrupt details
       - state.tasks[0].interrupts[0].value: Get message
       - Use to build UI or logging

    6. PRODUCTION PATTERNS:
       - Approval thresholds (financial, data size, etc.)
       - Content moderation (toxicity scores)
       - Rate limiting (API calls, costs)
       - Data quality checks
       - Security validations

    7. COMBINING APPROACHES:
       - Can use BOTH static and dynamic interrupts
       - Static for fixed checkpoints
       - Dynamic for conditional logic
       - Best of both worlds!

    8. WHEN TO USE:
       ✓ When interrupts should be rare
       ✓ When decision depends on data
       ✓ When you need to explain why
       ✓ When business rules are complex
       ✗ When every execution needs review (use static)
    """
    print(concepts)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "🚀"*35)
    print("MODULE 3 - LESSON 3: DYNAMIC BREAKPOINTS")
    print("🚀"*35)

    # Run examples
    run_basic_dynamic_interrupt()
    run_short_input_no_interrupt()
    run_approval_workflow_under_threshold()
    run_approval_workflow_over_threshold()
    run_comparison_static_vs_dynamic()

    # Show key concepts
    print_key_concepts()

    print("\n" + "="*70)
    print("✅ Module 3, Lesson 3 Complete!")
    print("="*70)
    print("\n📖 Next: time_travel.py - Learn to rewind and replay execution")
    print("   (Advanced debugging and error recovery)")
    print()
