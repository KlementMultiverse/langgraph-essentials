"""
LangGraph Academy - Module 2: State Reducers
============================================

Reducers control HOW state updates are combined:
- Without reducer: NEW value OVERWRITES old value
- With reducer: NEW value is COMBINED with old value using a function

Key Concepts:
1. operator.add - Built-in reducer for appending lists/concatenating strings
2. add_messages - Special reducer for chat messages
3. Custom reducers - Write your own combining logic
4. MessagesState - Pre-built state with message reducer

Author: Learning from LangChain Academy
Date: 2026-01-10
"""

from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.graph.message import add_messages


# =============================================================================
# PART 1: The Problem - Overwriting vs Appending
# =============================================================================

print("=" * 70)
print("PART 1: Understanding the Problem")
print("=" * 70)

# WITHOUT REDUCER - Values get OVERWRITTEN
class StateWithoutReducer(TypedDict):
    messages: list  # No reducer - will OVERWRITE!


def node_without_reducer_1(state):
    return {"messages": ["Message 1"]}


def node_without_reducer_2(state):
    return {"messages": ["Message 2"]}  # This OVERWRITES Message 1!


# Build graph
builder = StateGraph(StateWithoutReducer)
builder.add_node("node_1", node_without_reducer_1)
builder.add_node("node_2", node_without_reducer_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)
graph_without = builder.compile()

result = graph_without.invoke({"messages": []})
print("\n⚠️  WITHOUT Reducer (Overwriting):")
print(f"Result: {result}")
print(f"Expected: ['Message 1', 'Message 2']")
print(f"Got: {result['messages']} ← Message 1 is LOST!")


# =============================================================================
# PART 2: Solution - Using operator.add Reducer
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 2: Using operator.add Reducer")
print("=" * 70)

class StateWithReducer(TypedDict):
    messages: Annotated[list, add]  # ← Reducer APPENDS instead of overwriting!


def node_with_reducer_1(state):
    return {"messages": ["Message 1"]}


def node_with_reducer_2(state):
    return {"messages": ["Message 2"]}  # This APPENDS to Message 1!


# Build graph
builder = StateGraph(StateWithReducer)
builder.add_node("node_1", node_with_reducer_1)
builder.add_node("node_2", node_with_reducer_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)
graph_with = builder.compile()

result = graph_with.invoke({"messages": []})
print("\n✅ WITH Reducer (Appending):")
print(f"Result: {result}")
print(f"Got: {result['messages']} ← Both messages kept!")


# =============================================================================
# PART 3: add_messages Reducer for Chat Messages
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 3: add_messages Reducer for Chat Messages")
print("=" * 70)

class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # Special message reducer


def user_node(state):
    """Simulates user sending a message"""
    return {"messages": [HumanMessage(content="Hello! How are you?")]}


def ai_node(state):
    """Simulates AI responding"""
    return {"messages": [AIMessage(content="I'm doing great! How can I help?")]}


def user_followup(state):
    """User asks another question"""
    return {"messages": [HumanMessage(content="What's the weather?")]}


# Build chat graph
builder = StateGraph(ChatState)
builder.add_node("user_1", user_node)
builder.add_node("ai_1", ai_node)
builder.add_node("user_2", user_followup)
builder.add_edge(START, "user_1")
builder.add_edge("user_1", "ai_1")
builder.add_edge("ai_1", "user_2")
builder.add_edge("user_2", END)
chat_graph = builder.compile()

result = chat_graph.invoke({"messages": []})
print("\n💬 Chat Conversation:")
for i, msg in enumerate(result['messages'], 1):
    role = "User" if isinstance(msg, HumanMessage) else "AI"
    print(f"{i}. {role}: {msg.content}")


# =============================================================================
# PART 4: MessagesState - Pre-built State Class
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 4: MessagesState - Pre-built Convenience Class")
print("=" * 70)

from langgraph.graph import MessagesState

# MessagesState already has: messages: Annotated[list[AnyMessage], add_messages]
# You don't need to define it yourself!

class MyAppState(MessagesState):
    """
    Inherit from MessagesState to get messages with reducer built-in.
    Add any extra fields you need.
    """
    user_name: str  # Extra field for user name


def greet_user(state):
    """Use both messages and custom fields"""
    return {
        "messages": [AIMessage(content=f"Hello {state['user_name']}! Welcome!")]
    }


def ask_question(state):
    return {
        "messages": [HumanMessage(content="Tell me about LangGraph")]
    }


# Build graph
builder = StateGraph(MyAppState)
builder.add_node("greet", greet_user)
builder.add_node("question", ask_question)
builder.add_edge(START, "greet")
builder.add_edge("greet", "question")
builder.add_edge("question", END)
app_graph = builder.compile()

result = app_graph.invoke({"messages": [], "user_name": "Alice"})
print("\n👤 Using MessagesState with custom fields:")
print(f"User: {result['user_name']}")
print("Messages:")
for i, msg in enumerate(result['messages'], 1):
    role = "User" if isinstance(msg, HumanMessage) else "AI"
    print(f"  {i}. {role}: {msg.content}")


# =============================================================================
# PART 5: Custom Reducers
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 5: Custom Reducers - Write Your Own Logic")
print("=" * 70)

def keep_last_3_messages(existing: list, new: list) -> list:
    """
    Custom reducer: Only keep the last 3 messages total.
    Useful for limiting context window size.
    """
    combined = existing + new
    return combined[-3:]  # Keep only last 3


class LimitedChatState(TypedDict):
    messages: Annotated[list[AnyMessage], keep_last_3_messages]  # Custom reducer!


def add_msg(content: str, is_human: bool):
    """Helper to create message nodes"""
    def node(state):
        msg = HumanMessage(content=content) if is_human else AIMessage(content=content)
        return {"messages": [msg]}
    return node


# Build graph with 5 messages (but only last 3 will be kept)
builder = StateGraph(LimitedChatState)
builder.add_node("msg1", add_msg("Message 1", True))
builder.add_node("msg2", add_msg("Message 2", False))
builder.add_node("msg3", add_msg("Message 3", True))
builder.add_node("msg4", add_msg("Message 4", False))
builder.add_node("msg5", add_msg("Message 5", True))

builder.add_edge(START, "msg1")
builder.add_edge("msg1", "msg2")
builder.add_edge("msg2", "msg3")
builder.add_edge("msg3", "msg4")
builder.add_edge("msg4", "msg5")
builder.add_edge("msg5", END)

limited_graph = builder.compile()

result = limited_graph.invoke({"messages": []})
print("\n🔢 Custom Reducer (Keep Last 3):")
print(f"Added 5 messages, but only kept last 3:")
for i, msg in enumerate(result['messages'], 1):
    print(f"  {i}. {msg.content}")


# =============================================================================
# PART 6: Another Custom Reducer - Sum Numbers
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 6: Custom Reducer for Numbers")
print("=" * 70)

def sum_reducer(existing: int, new: int) -> int:
    """Custom reducer: Sum numbers instead of replacing"""
    return existing + new


class CounterState(TypedDict):
    count: Annotated[int, sum_reducer]  # Will SUM instead of replace
    name: str  # Normal field (no reducer)


def increment_by_5(state):
    return {"count": 5}


def increment_by_3(state):
    return {"count": 3}


# Build counter graph
builder = StateGraph(CounterState)
builder.add_node("add5", increment_by_5)
builder.add_node("add3", increment_by_3)
builder.add_edge(START, "add5")
builder.add_edge("add5", "add3")
builder.add_edge("add3", END)
counter_graph = builder.compile()

result = counter_graph.invoke({"count": 10, "name": "MyCounter"})
print("\n🔢 Number Sum Reducer:")
print(f"Started with: 10")
print(f"Added: 5 + 3")
print(f"Final count: {result['count']} (10 + 5 + 3 = 18)")


# =============================================================================
# Summary
# =============================================================================

print("\n\n" + "=" * 70)
print("📚 KEY TAKEAWAYS")
print("=" * 70)
print("""
1. WITHOUT reducer: messages: list
   → Each update OVERWRITES the previous value

2. WITH operator.add: messages: Annotated[list, add]
   → Each update APPENDS to the list

3. WITH add_messages: messages: Annotated[list[AnyMessage], add_messages]
   → Special reducer for chat messages (handles updates, deletions)

4. MessagesState: Pre-built class with message reducer
   → Just inherit from it: class MyState(MessagesState)

5. Custom reducers: Write your own function
   → def my_reducer(existing, new) -> combined

WHEN TO USE WHAT:
- Chat applications → use MessagesState or add_messages
- Lists that grow → use operator.add
- Custom logic → write your own reducer
- Simple values that replace → no reducer needed
""")
print("=" * 70)
