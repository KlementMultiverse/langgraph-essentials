"""
LangGraph Academy - Module 2: Chatbot with External Memory (SQLite)
====================================================================

Persistent memory using external database instead of in-memory storage.

Key Concepts:
1. MemorySaver = In-memory (data lost on restart)
2. SqliteSaver = Database file (data persists forever)
3. Same chatbot code, just different checkpointer
4. Production-ready persistence

Benefits of External Memory:
- Data survives app restarts
- Multiple users with separate conversations
- Scalable to millions of conversations
- Can query/analyze conversation history

Author: Learning from LangChain Academy
Date: 2026-01-10
"""

import sqlite3
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import Literal


# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)


# =============================================================================
# PART 1: The Problem with MemorySaver
# =============================================================================

print("=" * 70)
print("PART 1: The Problem - MemorySaver is Temporary")
print("=" * 70)

print("""
❌ MemorySaver (In-Memory Storage):

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

Problems:
1. Data stored in RAM (computer memory)
2. When you restart program → ALL DATA LOST!
3. When you close app → Conversation history GONE!
4. Cannot share across processes
5. Not production-ready

Example:
User: "My name is Alice"
Bot: "Hi Alice!"
[Restart program]
User: "What's my name?"
Bot: "I don't know" ← FORGOT EVERYTHING!

✅ SqliteSaver (Database Storage):

conn = sqlite3.connect("database.db")
memory = SqliteSaver(conn)
graph = workflow.compile(checkpointer=memory)

Benefits:
1. Data stored in database file
2. Survives restarts ✓
3. Persists forever ✓
4. Can share across processes ✓
5. Production-ready ✓

Example:
User: "My name is Alice"
Bot: "Hi Alice!"
[Restart program]
User: "What's my name?"
Bot: "Your name is Alice!" ← REMEMBERS!
""")


# =============================================================================
# PART 2: State Schema (Same as Before)
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: State Schema (Same as Summarization)")
print("=" * 70)

class State(MessagesState):
    """Extended state with summary field"""
    summary: str


# =============================================================================
# PART 3: Nodes (Same as Before)
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: Node Functions (Unchanged)")
print("=" * 70)

def call_model(state: State):
    """Call LLM with messages + summary (if exists)"""
    summary = state.get("summary", "")

    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]

    response = llm.invoke(messages)
    return {"messages": response}


def summarize_conversation(state: State):
    """Summarize conversation and delete old messages"""
    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """Determine whether to summarize or end"""
    messages = state["messages"]

    if len(messages) > 6:
        return "summarize_conversation"

    return END


print("""
All node functions are IDENTICAL to chatbot-summarization!

The ONLY difference is the checkpointer:
- Before: MemorySaver (in-memory)
- Now: SqliteSaver (database)

Same code, different persistence layer!
""")


# =============================================================================
# PART 4: SQLite Checkpointer - In-Memory vs File-Based
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: SQLite Checkpointer Options")
print("=" * 70)

print("""
Option 1: In-Memory SQLite (Like MemorySaver)
----------------------------------------------
conn = sqlite3.connect(":memory:", check_same_thread=False)
memory = SqliteSaver(conn)

- Uses ":memory:" special string
- Creates database in RAM
- Data lost on restart (like MemorySaver)
- Good for: Testing, temporary use


Option 2: File-Based SQLite (Persistent!) ✅
--------------------------------------------
conn = sqlite3.connect("chatbot_memory.db", check_same_thread=False)
memory = SqliteSaver(conn)

- Creates "chatbot_memory.db" file
- Data persists forever
- Survives restarts
- Good for: Production, real apps


Parameter Explained:
--------------------
check_same_thread=False
- Allows SQLite to be used in multi-threaded apps
- LangGraph may use multiple threads
- Always set this to False for LangGraph!
""")


# =============================================================================
# PART 5: Build Graph with File-Based SQLite
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: Building Graph with Persistent SQLite")
print("=" * 70)

# Create file-based SQLite database
db_path = "chatbot_memory.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)

print(f"✅ Created SQLite database: {db_path}")

# Build graph (same as before)
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# Compile with SqliteSaver checkpointer
graph = workflow.compile(checkpointer=memory)

print("✅ Compiled graph with SqliteSaver checkpointer")
print("""
Graph Structure:
    START → conversation → should_continue
              ↓
    If > 6: summarize_conversation → END
    If ≤ 6: END

Checkpointer: SqliteSaver (file-based, persistent!)
""")


# =============================================================================
# PART 6: Run Conversation - Session 1
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: Session 1 - Creating Initial Conversation")
print("=" * 70)

# Configuration with thread_id
config = {"configurable": {"thread_id": "user_alice_001"}}

print(f"🔑 Thread ID: {config['configurable']['thread_id']}")
print("   (This identifies the conversation in the database)\n")

# Turn 1
print("🔵 Turn 1: User introduces themselves")
output = graph.invoke({"messages": [HumanMessage(content="Hi! My name is Alice.")]}, config)
print(f"   AI: {output['messages'][-1].content[:100]}...")

# Turn 2
print("\n🔵 Turn 2: User shares interest")
output = graph.invoke({"messages": [HumanMessage(content="I love ocean mammals!")]}, config)
print(f"   AI: {output['messages'][-1].content[:100]}...")

# Turn 3
print("\n🔵 Turn 3: Ask about whales")
output = graph.invoke({"messages": [HumanMessage(content="Tell me about whales")]}, config)
print(f"   AI: {output['messages'][-1].content[:100]}...")

print(f"\n📊 Messages in state: {len(output['messages'])}")
print(f"📝 Summary: {output.get('summary', 'None')}")
print("\n✅ Conversation saved to database!")
print(f"   File: {db_path}")


# =============================================================================
# PART 7: Simulate Restart - Load Conversation from Database
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 7: SIMULATE RESTART - Reload from Database")
print("=" * 70)

print("🔄 Simulating app restart...")
print("   (In real scenario, you'd close and reopen the app)\n")

# Reconnect to database (simulating restart)
conn_new = sqlite3.connect(db_path, check_same_thread=False)
memory_new = SqliteSaver(conn_new)

# Rebuild graph with new connection
workflow_new = StateGraph(State)
workflow_new.add_node("conversation", call_model)
workflow_new.add_node(summarize_conversation)
workflow_new.add_edge(START, "conversation")
workflow_new.add_conditional_edges("conversation", should_continue)
workflow_new.add_edge("summarize_conversation", END)
graph_new = workflow_new.compile(checkpointer=memory_new)

print("✅ Reconnected to database")
print("✅ Graph rebuilt\n")

# Continue conversation with SAME thread_id
print("🔵 Turn 4: Ask personal question after 'restart'")
output = graph_new.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    config  # Same thread_id!
)

print(f"   AI: {output['messages'][-1].content}")
print("\n🎉 SUCCESS! AI remembers name from before 'restart'!")
print(f"   Previous conversation loaded from: {db_path}")


# =============================================================================
# PART 8: Multiple Users - Different Conversations
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 8: Multiple Users - Separate Conversations")
print("=" * 70)

# User 1 (Alice) - existing thread
config_alice = {"configurable": {"thread_id": "user_alice_001"}}

# User 2 (Bob) - new thread
config_bob = {"configurable": {"thread_id": "user_bob_002"}}

print("👤 User: Bob (NEW conversation)")
print(f"🔑 Thread ID: {config_bob['configurable']['thread_id']}\n")

# Bob's conversation
print("🔵 Bob Turn 1: Introduce")
output_bob = graph_new.invoke(
    {"messages": [HumanMessage(content="Hi, I'm Bob")]},
    config_bob
)
print(f"   AI: {output_bob['messages'][-1].content[:100]}...")

print("\n🔵 Bob Turn 2: Ask about his name")
output_bob = graph_new.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    config_bob
)
print(f"   AI: {output_bob['messages'][-1].content}")

# Alice's conversation (still separate!)
print("\n\n👤 User: Alice (EXISTING conversation)")
print(f"🔑 Thread ID: {config_alice['configurable']['thread_id']}\n")

print("🔵 Alice Turn 5: Ask about her name")
output_alice = graph_new.invoke(
    {"messages": [HumanMessage(content="What's my name again?")]},
    config_alice
)
print(f"   AI: {output_alice['messages'][-1].content}")

print("\n✅ Two separate conversations stored in same database!")
print("   - Alice's thread: user_alice_001")
print("   - Bob's thread: user_bob_002")
print("   - No cross-contamination!")


# =============================================================================
# PART 9: Comparison - MemorySaver vs SqliteSaver
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 9: MemorySaver vs SqliteSaver Comparison")
print("=" * 70)

comparison = """
┌──────────────────────┬─────────────────┬─────────────────────┐
│ Feature              │ MemorySaver     │ SqliteSaver         │
├──────────────────────┼─────────────────┼─────────────────────┤
│ Storage Location     │ RAM (memory)    │ Database file       │
├──────────────────────┼─────────────────┼─────────────────────┤
│ Persists on Restart  │ ❌ No           │ ✅ Yes              │
├──────────────────────┼─────────────────┼─────────────────────┤
│ Survives App Close   │ ❌ No           │ ✅ Yes              │
├──────────────────────┼─────────────────┼─────────────────────┤
│ Multi-Process        │ ❌ No           │ ✅ Yes (with care)  │
├──────────────────────┼─────────────────┼─────────────────────┤
│ Scalability          │ Limited by RAM  │ Disk space          │
├──────────────────────┼─────────────────┼─────────────────────┤
│ Production Ready     │ ❌ No           │ ✅ Yes              │
├──────────────────────┼─────────────────┼─────────────────────┤
│ Setup Complexity     │ Easy            │ Easy                │
├──────────────────────┼─────────────────┼─────────────────────┤
│ Speed                │ Very fast       │ Fast                │
├──────────────────────┼─────────────────┼─────────────────────┤
│ Use Case             │ Testing, demos  │ Production apps     │
└──────────────────────┴─────────────────┴─────────────────────┘

CODE COMPARISON:

MemorySaver:
------------
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

SqliteSaver:
------------
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("database.db", check_same_thread=False)
memory = SqliteSaver(conn)
graph = workflow.compile(checkpointer=memory)

THAT'S THE ONLY DIFFERENCE! 🎉
"""

print(comparison)


# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("📚 KEY TAKEAWAYS")
print("=" * 70)
print("""
1. PERSISTENCE OPTIONS:
   - MemorySaver: In-memory (temporary)
   - SqliteSaver: Database file (persistent)
   - PostgreSaver: PostgreSQL (production scale)
   - RedisSaver: Redis (distributed systems)

2. SQLITE CHECKPOINTER:
   # In-memory (temporary):
   conn = sqlite3.connect(":memory:", check_same_thread=False)

   # File-based (persistent):
   conn = sqlite3.connect("database.db", check_same_thread=False)

   memory = SqliteSaver(conn)

3. THREAD_ID:
   - Identifies each conversation
   - Different users = different thread_ids
   - Same user, same conversation = same thread_id
   - Format: {"configurable": {"thread_id": "user_123"}}

4. CODE CHANGES:
   ❌ Node functions: NO CHANGE
   ❌ Graph structure: NO CHANGE
   ❌ State schema: NO CHANGE
   ✅ Checkpointer: ONLY CHANGE!

5. WHEN TO USE SQLITE:
   ✅ Production chatbots
   ✅ Multi-user applications
   ✅ Need conversation history
   ✅ Want data persistence
   ✅ Medium scale (< 100k users)

6. WHEN TO USE POSTGRES:
   ✅ Large scale (> 100k users)
   ✅ High concurrency
   ✅ Enterprise applications
   ✅ Need advanced queries
   ✅ Distributed systems

7. BEST PRACTICES:
   - Always use check_same_thread=False
   - Use meaningful thread_ids (e.g., user IDs)
   - Close connections properly (use context managers)
   - Backup database files regularly
   - Consider database size limits

8. PRODUCTION TIPS:
   - Use PostgreSQL for production (more robust)
   - Implement connection pooling
   - Add error handling for DB failures
   - Monitor database size and performance
   - Implement data retention policies
   - Encrypt sensitive conversation data
""")
print("=" * 70)


# Clean up
conn.close()
conn_new.close()
print("\n🧹 Closed database connections")
