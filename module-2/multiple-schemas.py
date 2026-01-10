"""
LangGraph Academy - Module 2: Multiple Schemas
==============================================

Demonstrates how to use different schemas for:
1. Private State - Internal fields hidden from input/output
2. Input/Output Schemas - Control what user sends and receives

Key Concepts:
- OverallState: Main schema with all fields
- PrivateState: Secret fields passed between nodes only
- InputState: Controls what user can INPUT
- OutputState: Controls what user receives as OUTPUT

Author: Learning from LangChain Academy
Date: 2026-01-10
"""

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# =============================================================================
# PART 1: Private State - Hiding Internal Data Between Nodes
# =============================================================================

print("=" * 70)
print("PART 1: Private State - Internal Data Between Nodes")
print("=" * 70)

class OverallState(TypedDict):
    """Public state - what user sees"""
    foo: int


class PrivateState(TypedDict):
    """Private state - only used internally between nodes"""
    baz: int


def node_1(state: OverallState) -> PrivateState:
    """
    Takes OverallState (foo) and returns PrivateState (baz)
    Converts public data to private internal data
    """
    print("---Node 1---")
    print(f"  Input (OverallState): foo={state['foo']}")
    result = state['foo'] + 1
    print(f"  Output (PrivateState): baz={result}")
    return {"baz": result}


def node_2(state: PrivateState) -> OverallState:
    """
    Takes PrivateState (baz) and returns OverallState (foo)
    Converts private data back to public data
    """
    print("---Node 2---")
    print(f"  Input (PrivateState): baz={state['baz']}")
    result = state['baz'] + 1
    print(f"  Output (OverallState): foo={result}")
    return {"foo": result}


# Build graph
builder = StateGraph(OverallState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)
graph_private = builder.compile()

# Run example
print("\n🔍 Running graph with private state:")
print("Input: {'foo': 1}")
result = graph_private.invoke({"foo": 1})
print(f"\n✅ Output: {result}")
print("\n🔑 Key Point: 'baz' was used internally but NOT in output!")
print("   - node_1 created baz=2")
print("   - node_2 used baz=2 internally")
print("   - Output only contains foo=3 (no baz!)")


# =============================================================================
# PART 2: Problem - All Fields Exposed (Without Input/Output Schemas)
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 2: Problem - All Fields Exposed to User")
print("=" * 70)

class OverallStateV1(TypedDict):
    """State with internal 'notes' field that shouldn't be exposed"""
    question: str
    answer: str
    notes: str  # Internal field - but user will see it!


def thinking_node_v1(state: OverallStateV1):
    """Node that does internal thinking"""
    print("---Thinking Node---")
    print(f"  Question: {state.get('question', 'N/A')}")
    return {
        "answer": "bye",
        "notes": "... his name is Lance"  # Internal scratch work
    }


def answer_node_v1(state: OverallStateV1):
    """Node that produces final answer"""
    print("---Answer Node---")
    print(f"  Using notes: {state.get('notes', 'N/A')}")
    return {"answer": "bye Lance"}


# Build graph WITHOUT input/output schemas
graph_v1 = StateGraph(OverallStateV1)
graph_v1.add_node("thinking_node", thinking_node_v1)
graph_v1.add_node("answer_node", answer_node_v1)
graph_v1.add_edge(START, "thinking_node")
graph_v1.add_edge("thinking_node", "answer_node")
graph_v1.add_edge("answer_node", END)
graph_v1 = graph_v1.compile()

# Run example
print("\n⚠️  Without Input/Output Schemas:")
print("Input: {'question': 'hi'}")
result = graph_v1.invoke({"question": "hi"})
print(f"\n❌ Output: {result}")
print("\n🔓 Problem: User sees 'notes' field - it's exposed!")


# =============================================================================
# PART 3: Solution - Input/Output Schemas Filter Data
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 3: Solution - Input/Output Schemas")
print("=" * 70)

class InputState(TypedDict):
    """What user can INPUT - only question"""
    question: str


class OutputState(TypedDict):
    """What user receives as OUTPUT - only answer"""
    answer: str


class OverallStateV2(TypedDict):
    """Internal state - has ALL fields including notes"""
    question: str
    answer: str
    notes: str  # Internal only - will be filtered out!


def thinking_node_v2(state: InputState):
    """
    Takes InputState (only question)
    Returns answer + notes (notes will be internal only)
    """
    print("---Thinking Node---")
    print(f"  Question: {state.get('question', 'N/A')}")
    return {
        "answer": "bye",
        "notes": "... his name is Lance"  # This stays internal!
    }


def answer_node_v2(state: OverallStateV2) -> OutputState:
    """
    Takes OverallState (can access notes internally)
    Returns OutputState (only answer)
    """
    print("---Answer Node---")
    print(f"  Using notes internally: {state.get('notes', 'N/A')}")
    return {"answer": "bye Lance"}


# Build graph WITH input/output schemas ← THE KEY!
graph_v2 = StateGraph(
    OverallStateV2,
    input_schema=InputState,   # ← Filter INPUT
    output_schema=OutputState  # ← Filter OUTPUT
)
graph_v2.add_node("thinking_node", thinking_node_v2)
graph_v2.add_node("answer_node", answer_node_v2)
graph_v2.add_edge(START, "thinking_node")
graph_v2.add_edge("thinking_node", "answer_node")
graph_v2.add_edge("answer_node", END)
graph_v2 = graph_v2.compile()

# Run example
print("\n✅ With Input/Output Schemas:")
print("Input: {'question': 'hi'}")
result = graph_v2.invoke({"question": "hi"})
print(f"\n✅ Output: {result}")
print("\n🔒 Success: 'notes' field is HIDDEN from user!")
print("   - InputState filtered input to only 'question'")
print("   - Graph used 'notes' internally")
print("   - OutputState filtered output to only 'answer'")


# =============================================================================
# PART 4: Real-World Example - Chatbot with Debug Info
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 4: Real-World Example - Chatbot with Hidden Debug Info")
print("=" * 70)

class ChatInputState(TypedDict):
    """User input - just the message"""
    user_message: str


class ChatOutputState(TypedDict):
    """User output - just the response"""
    response: str


class ChatInternalState(TypedDict):
    """Internal state with debug info"""
    user_message: str
    response: str
    debug_info: str      # Hidden from user
    api_calls: int       # Hidden from user
    timestamp: str       # Hidden from user


def process_message(state: ChatInputState):
    """Process user message with internal tracking"""
    print(f"📨 Received: {state['user_message']}")
    return {
        "response": f"Echo: {state['user_message']}",
        "debug_info": "Model: GPT-4, Tokens: 150",
        "api_calls": 1,
        "timestamp": "2026-01-10 14:30:00"
    }


def finalize_response(state: ChatInternalState) -> ChatOutputState:
    """Finalize response, using debug info internally"""
    print(f"🔧 Debug: {state.get('debug_info', 'N/A')}")
    print(f"📊 API Calls: {state.get('api_calls', 0)}")
    print(f"⏰ Timestamp: {state.get('timestamp', 'N/A')}")
    return {"response": state['response']}


# Build chatbot graph
chatbot = StateGraph(
    ChatInternalState,
    input_schema=ChatInputState,
    output_schema=ChatOutputState
)
chatbot.add_node("process", process_message)
chatbot.add_node("finalize", finalize_response)
chatbot.add_edge(START, "process")
chatbot.add_edge("process", "finalize")
chatbot.add_edge("finalize", END)
chatbot = chatbot.compile()

# Run chatbot
print("\n🤖 Chatbot Example:")
print("User Input: {'user_message': 'Hello!'}")
result = chatbot.invoke({"user_message": "Hello!"})
print(f"\n👤 User Sees: {result}")
print("\n✅ Debug info (api_calls, timestamp, debug_info) is HIDDEN!")


# =============================================================================
# Summary
# =============================================================================

print("\n\n" + "=" * 70)
print("📚 KEY TAKEAWAYS")
print("=" * 70)
print("""
1. PRIVATE STATE:
   - Define PrivateState for internal-only fields
   - Nodes can input/output different state types
   - Private fields never appear in final output

2. INPUT/OUTPUT SCHEMAS:
   - InputState: Controls what user can send IN
   - OutputState: Controls what user receives OUT
   - OverallState: Internal schema with ALL fields

3. HOW TO USE:
   StateGraph(
       OverallState,              # Internal schema
       input_schema=InputState,   # Filter input
       output_schema=OutputState  # Filter output
   )

4. REAL-WORLD USE CASES:
   - Hide debug info from users
   - Hide API call counts, timestamps
   - Hide internal reasoning/notes
   - Simplify user-facing API
   - Keep sensitive data internal

5. TYPE HINTS:
   - def node(state: InputState) -> OutputState:
   - Helps document what each node expects/returns
""")
print("=" * 70)
