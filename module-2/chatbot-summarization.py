"""
LangGraph Academy - Module 2: Chatbot with Summarization
=========================================================

Instead of DELETING old messages (trimming/filtering), we SUMMARIZE them!

Key Concepts:
1. Summarization preserves information (unlike trimming)
2. Old messages compressed into summary
3. Summary included in context for future messages
4. Automatic summarization when conversation gets too long

Benefits:
- Keeps important context (names, facts, preferences)
- Reduces token count (summary shorter than all messages)
- AI remembers the full conversation history

Author: Learning from LangChain Academy
Date: 2026-01-10
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import Literal


# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)


# =============================================================================
# PART 1: The Problem with Trimming/Filtering
# =============================================================================

print("=" * 70)
print("PART 1: The Problem - Trimming LOSES Information")
print("=" * 70)

print("""
❌ TRIMMING/FILTERING APPROACH:

Conversation:
1. User: "Hi, my name is John"
2. AI: "Hello John!"
3. User: "I love whales"
4. AI: "Whales are amazing!"
5. User: "Tell me about dolphins"

After trimming (keep last 2):
→ [msg4, msg5]  ← msg1, msg2, msg3 DELETED!

Problem: AI forgets user's name is John! 😞

✅ SUMMARIZATION APPROACH:

Same conversation, but summarize instead:
→ Summary: "User's name is John. Interested in ocean mammals, especially whales."
→ Keep: [Summary, msg4, msg5]

Result: AI remembers John's name! 🎉
""")


# =============================================================================
# PART 2: Extended State with Summary Field
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: State Schema with Summary Field")
print("=" * 70)

class State(MessagesState):
    """
    Extended state that includes a summary field.

    Fields:
    - messages: list of messages (from MessagesState)
    - summary: str - compressed history of old messages
    """
    summary: str


print("""
State Schema:
- messages: Annotated[list[AnyMessage], add_messages]  ← From MessagesState
- summary: str  ← NEW! Stores compressed conversation history

The summary field stores old conversation info in compressed form.
""")


# =============================================================================
# PART 3: call_model Node - Include Summary in Context
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: call_model - Smart Node with Summary Support")
print("=" * 70)

def call_model(state: State):
    """
    Call LLM with messages + summary (if exists).

    Logic:
    1. Get summary from state (if exists)
    2. If summary exists, prepend it as SystemMessage
    3. Send enhanced messages to LLM
    4. Return AI response
    """

    # Get summary if it exists
    summary = state.get("summary", "")

    print(f"  📊 Current messages: {len(state['messages'])}")
    print(f"  📝 Summary exists: {bool(summary)}")

    # If there is summary, add it to context
    if summary:
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Prepend summary to messages
        messages = [SystemMessage(content=system_message)] + state["messages"]

        print(f"  ✅ Added summary to context")
        print(f"  📤 Sending {len(messages)} messages to LLM (including summary)")
    else:
        messages = state["messages"]
        print(f"  📤 Sending {len(messages)} messages to LLM (no summary)")

    # Invoke LLM
    response = llm.invoke(messages)

    return {"messages": response}


print("""
call_model Function Flow:

1. Check if summary exists
   - state.get("summary", "") → safe access

2. IF summary exists:
   - Create SystemMessage with summary
   - Prepend to current messages
   - [SystemMessage(summary), msg1, msg2, ...]

3. IF no summary:
   - Just use current messages
   - [msg1, msg2, ...]

4. Send to LLM and return response

Key Benefit: AI sees compressed history + recent messages!
""")


# =============================================================================
# PART 4: summarize_conversation Node - Create/Extend Summary
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: summarize_conversation - Create/Extend Summary")
print("=" * 70)

def summarize_conversation(state: State):
    """
    Summarize the conversation and delete old messages.

    Logic:
    1. Get existing summary (if any)
    2. Create prompt to create/extend summary
    3. Ask LLM to summarize
    4. Delete all but last 2 messages
    5. Return new summary + RemoveMessage list
    """

    # Get any existing summary
    summary = state.get("summary", "")

    print(f"\n  📊 Summarizing {len(state['messages'])} messages")
    print(f"  📝 Existing summary: {bool(summary)}")

    # Create summarization prompt
    if summary:
        # Summary exists - ask LLM to EXTEND it
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        print(f"  🔄 Extending existing summary")
    else:
        # No summary - ask LLM to CREATE one
        summary_message = "Create a summary of the conversation above:"
        print(f"  ✨ Creating new summary")

    # Add prompt to messages
    messages = state["messages"] + [HumanMessage(content=summary_message)]

    # Ask LLM to summarize
    response = llm.invoke(messages)

    print(f"  ✅ Summary created: {response.content[:50]}...")

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    print(f"  🗑️  Deleting {len(delete_messages)} old messages")
    print(f"  💾 Keeping last 2 messages")

    return {
        "summary": response.content,  # Update summary
        "messages": delete_messages   # Delete old messages
    }


print("""
summarize_conversation Function Flow:

1. Get existing summary
   - state.get("summary", "")

2. Create prompt based on whether summary exists:
   - IF exists: "Extend the summary..."
   - IF new: "Create a summary..."

3. Add prompt to messages and send to LLM
   - messages + [HumanMessage(prompt)]

4. LLM generates summary

5. Delete old messages (keep last 2)
   - [RemoveMessage(id=m.id) for m in messages[:-2]]

6. Return:
   - summary: new/extended summary
   - messages: list of RemoveMessage objects

Result: Old messages compressed into summary!
""")


# =============================================================================
# PART 5: should_continue - Conditional Edge
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: should_continue - Decide When to Summarize")
print("=" * 70)

def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """
    Determine whether to summarize or end.

    Logic:
    - If more than 6 messages → summarize
    - Otherwise → end
    """

    messages = state["messages"]
    message_count = len(messages)

    print(f"\n  📊 Message count: {message_count}")

    # If there are more than 6 messages, summarize
    if message_count > 6:
        print(f"  🔄 Too many messages (> 6) → Going to summarize")
        return "summarize_conversation"

    # Otherwise just end
    print(f"  ✅ Few messages (≤ 6) → Ending without summary")
    return END


print("""
should_continue Function:

Conditional edge that decides next step based on message count.

IF len(messages) > 6:
    → return "summarize_conversation"
    → Graph goes to summarize node

ELSE:
    → return END
    → Graph stops

Threshold (6) is configurable - adjust based on your needs!
""")


# =============================================================================
# PART 6: Build Complete Graph
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: Building the Complete Graph")
print("=" * 70)

# Build graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)  # Shorthand - uses function name

# Add edges
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# Compile with memory
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

print("""
Graph Structure:

    START
      ↓
  conversation (call_model)
      ↓
  should_continue (conditional)
      ↓
  If > 6 messages:
      ↓
  summarize_conversation
      ↓
  END

  If ≤ 6 messages:
      ↓
  END (skip summarization)

Memory: MemorySaver (in-memory checkpointer for persistence)
""")


# =============================================================================
# PART 7: Run Example - Short Conversation (No Summary)
# =============================================================================

print("\n" + "=" * 70)
print("PART 7: Example 1 - Short Conversation (No Summary Needed)")
print("=" * 70)

# Configuration for thread
config = {"configurable": {"thread_id": "1"}}

# Turn 1
print("\n🔵 Turn 1: User introduces themselves")
input_message = HumanMessage(content="Hi! My name is Alice.")
output = graph.invoke({"messages": [input_message]}, config)
print(f"   AI: {output['messages'][-1].content[:100]}...")

# Turn 2
print("\n🔵 Turn 2: User asks about whales")
input_message = HumanMessage(content="Tell me about whales")
output = graph.invoke({"messages": [input_message]}, config)
print(f"   AI: {output['messages'][-1].content[:100]}...")

print(f"\n📊 Total messages: {len(output['messages'])}")
print(f"📝 Summary: {output.get('summary', 'None')}")
print("\n✅ Only 4 messages (≤ 6), no summarization triggered!")


# =============================================================================
# PART 8: Run Example - Long Conversation (With Summary)
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 8: Example 2 - Long Conversation (Summarization Triggered)")
print("=" * 70)

# New thread
config2 = {"configurable": {"thread_id": "2"}}

turns = [
    "Hi! My name is Bob.",
    "I'm interested in ocean mammals.",
    "Tell me about whales",
    "What about dolphins?",
    "How about seals?",
]

for i, msg in enumerate(turns, 1):
    print(f"\n🔵 Turn {i}: {msg}")
    output = graph.invoke({"messages": [HumanMessage(content=msg)]}, config2)
    print(f"   AI: {output['messages'][-1].content[:80]}...")
    print(f"   📊 Messages: {len(output['messages'])}, Summary: {bool(output.get('summary'))}")

print(f"\n📝 Final Summary: {output.get('summary', 'None')[:150]}...")
print(f"📊 Final message count: {len(output['messages'])}")
print("\n✅ After turn 4 (> 6 messages), summarization was triggered!")
print("   Old messages compressed into summary, only recent messages kept!")


# =============================================================================
# PART 9: Demonstrate Summary in Context
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 9: Summary Keeps Context - AI Remembers!")
print("=" * 70)

# Continue conversation - AI should remember name from summary
print("\n🔵 Turn 6: Ask personal question")
input_message = HumanMessage(content="What's my name?")
output = graph.invoke({"messages": [input_message]}, config2)

print(f"   AI Response: {output['messages'][-1].content}")
print(f"\n✅ AI remembers name (Bob) from summary, even though original message was deleted!")
print(f"📝 Summary in context: {output.get('summary', '')[:100]}...")


# =============================================================================
# Summary
# =============================================================================

print("\n\n" + "=" * 70)
print("📚 KEY TAKEAWAYS")
print("=" * 70)
print("""
1. SUMMARIZATION vs TRIMMING:
   ❌ Trimming: Deletes old messages → Loses information
   ✅ Summarization: Compresses old messages → Keeps context

2. STATE SCHEMA:
   class State(MessagesState):
       summary: str  ← Stores compressed conversation history

3. call_model NODE:
   - Checks if summary exists
   - Prepends summary as SystemMessage
   - Sends [Summary, current messages] to LLM
   - AI sees full context (compressed + recent)

4. summarize_conversation NODE:
   - Creates/extends summary using LLM
   - Deletes old messages (keeps last 2)
   - Returns new summary + RemoveMessage list

5. should_continue CONDITIONAL:
   - IF > 6 messages → go to summarize_conversation
   - ELSE → END
   - Threshold is configurable!

6. COMPLETE FLOW:
   START → conversation → check count
     ↓
   If > 6: summarize → delete old → END
   If ≤ 6: END

7. BENEFITS:
   ✅ Reduces token count (summary < all messages)
   ✅ Keeps important context (names, facts, preferences)
   ✅ AI remembers full conversation
   ✅ Controls costs while preserving information

8. PRODUCTION TIPS:
   - Adjust threshold based on your needs (6, 10, 20...)
   - Use MemorySaver or other checkpointer for persistence
   - Monitor summary quality
   - Consider periodic re-summarization for very long conversations

9. WHEN TO USE:
   - Long-running conversations
   - Customer support chatbots
   - Personal assistants
   - Any app where context matters over time
""")
print("=" * 70)
