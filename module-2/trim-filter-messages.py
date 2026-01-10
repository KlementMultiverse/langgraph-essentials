"""
LangGraph Academy - Module 2: Trim and Filter Messages
======================================================

Managing message history in long-running conversations to:
- Control token usage and costs
- Stay within model context limits
- Improve performance

Three Methods:
1. RemoveMessage - Delete old messages from state
2. Simple Slicing - Filter what you send to LLM
3. trim_messages - Smart trimming based on token count (BEST!)

Author: Learning from LangChain Academy
Date: 2026-01-10
"""

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, trim_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END


# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)


# =============================================================================
# PART 1: The Problem - Growing Message History
# =============================================================================

print("=" * 70)
print("PART 1: The Problem - Unbounded Message Growth")
print("=" * 70)

# Create a sample conversation
messages = [
    AIMessage("Hi! I'm here to help.", name="Bot", id="1"),
    HumanMessage("Hi!", name="User", id="2"),
    AIMessage("What would you like to know about?", name="Bot", id="3"),
    HumanMessage("Tell me about ocean mammals", name="User", id="4"),
    AIMessage("Ocean mammals include whales, dolphins, seals...", name="Bot", id="5"),
    HumanMessage("What about whales specifically?", name="User", id="6"),
]

print(f"\n📊 Current conversation has {len(messages)} messages")

# Count tokens (approximate)
total_content = " ".join([m.content for m in messages])
approx_tokens = len(total_content.split()) * 1.3  # Rough estimate
print(f"📏 Approximate tokens: {int(approx_tokens)}")

print("\n⚠️  Problem:")
print("   - Messages keep growing with each turn")
print("   - More messages = more tokens = more cost")
print("   - Eventually hit context limits")
print("   - Need to manage message history!")


# =============================================================================
# PART 2: Method 1 - RemoveMessage (Delete from State)
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 2: Method 1 - RemoveMessage (Delete Old Messages)")
print("=" * 70)

def filter_messages_node(state: MessagesState):
    """
    Keep only the last 2 messages, delete the rest from state.
    Uses RemoveMessage with add_messages reducer.
    """
    print(f"  📥 Input: {len(state['messages'])} messages")

    # Create RemoveMessage for all but last 2
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    print(f"  🗑️  Deleting: {len(delete_messages)} messages")
    print(f"  ✅ Keeping: last 2 messages")

    return {"messages": delete_messages}


def chat_model_node_v1(state: MessagesState):
    """Process messages with LLM"""
    print(f"  🤖 Sending {len(state['messages'])} messages to LLM")
    return {"messages": [llm.invoke(state["messages"])]}


# Build graph with RemoveMessage
builder = StateGraph(MessagesState)
builder.add_node("filter", filter_messages_node)
builder.add_node("chat_model", chat_model_node_v1)
builder.add_edge(START, "filter")
builder.add_edge("filter", "chat_model")
builder.add_edge("chat_model", END)
graph_v1 = builder.compile()

print("\n🔧 Running graph with RemoveMessage:")
output = graph_v1.invoke({"messages": messages})

print(f"\n📊 Output has {len(output['messages'])} messages:")
for i, m in enumerate(output['messages'], 1):
    role = "Bot" if isinstance(m, AIMessage) else "User"
    print(f"  {i}. {role}: {m.content[:50]}...")

print("\n🔑 Key Point: Old messages were DELETED from state permanently!")


# =============================================================================
# PART 3: Method 2 - Simple Slicing (Filter Input to LLM)
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 3: Method 2 - Simple Slicing (Keep in State, Filter to LLM)")
print("=" * 70)

def chat_model_node_v2(state: MessagesState):
    """
    Keep all messages in state, but only send last 2 to LLM.
    State still has full history!
    """
    all_messages = state["messages"]
    recent_messages = all_messages[-2:]  # Only last 2

    print(f"  📊 State has: {len(all_messages)} messages")
    print(f"  🤖 Sending to LLM: {len(recent_messages)} messages")

    return {"messages": [llm.invoke(recent_messages)]}


# Build graph with slicing
builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node_v2)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph_v2 = builder.compile()

print("\n🔧 Running graph with slicing:")
output = graph_v2.invoke({"messages": messages})

print(f"\n📊 Output has {len(output['messages'])} messages (all kept in state):")
print(f"   - Original 6 messages still in state")
print(f"   - Only sent last 2 to LLM")
print(f"   - Plus 1 new AI response = 7 total")

print("\n🔑 Key Point: Full history preserved in state, but LLM only sees recent!")


# =============================================================================
# PART 4: Method 3 - trim_messages (Smart Token-Based Trimming)
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 4: Method 3 - trim_messages (Token-Based Trimming) ⭐ BEST!")
print("=" * 70)

def chat_model_node_v3(state: MessagesState):
    """
    Trim messages based on TOKEN COUNT, not message count.
    Much smarter for cost control!
    """
    print(f"  📊 Input: {len(state['messages'])} messages")

    # Trim to max 100 tokens
    trimmed = trim_messages(
        state["messages"],
        max_tokens=100,                          # Token limit
        strategy="last",                         # Keep newest
        token_counter=ChatOpenAI(model="gpt-4o"),  # Use GPT-4o tokenizer
        allow_partial=False                      # Don't split messages
    )

    print(f"  ✂️  Trimmed to: {len(trimmed)} messages (max 100 tokens)")
    print(f"  🤖 Sending trimmed messages to LLM")

    return {"messages": [llm.invoke(trimmed)]}


# Build graph with trim_messages
builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node_v3)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph_v3 = builder.compile()

print("\n🔧 Running graph with trim_messages:")

# Create longer messages to demonstrate token trimming
long_messages = [
    HumanMessage("Hi there!", name="User", id="1"),
    AIMessage("Hello! How can I help you today?", name="Bot", id="2"),
    HumanMessage("I want to learn about ocean mammals, specifically whales and their migration patterns across different oceans.", name="User", id="3"),
    AIMessage("Great question! Whales migrate thousands of miles between feeding and breeding grounds. Different species have different patterns.", name="Bot", id="4"),
    HumanMessage("Tell me more about humpback whales specifically.", name="User", id="5"),
]

output = graph_v3.invoke({"messages": long_messages})

print(f"\n📊 Output: {len(output['messages'])} total messages")
print("\n🔑 Key Point: Trimmed based on TOKENS, not message count!")


# =============================================================================
# PART 5: Comparison of All Three Methods
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 5: Comparison - Which Method to Use?")
print("=" * 70)

comparison = """
┌─────────────────────┬──────────────┬─────────────┬─────────────────┐
│ Method              │ State Size   │ LLM Input   │ Best For        │
├─────────────────────┼──────────────┼─────────────┼─────────────────┤
│ 1. RemoveMessage    │ Reduced      │ Reduced     │ Memory savings  │
│                     │ (deletes old)│             │                 │
├─────────────────────┼──────────────┼─────────────┼─────────────────┤
│ 2. Simple Slicing   │ Full history │ Reduced     │ Keep full log   │
│                     │ (keeps all)  │ (filtered)  │ Simple logic    │
├─────────────────────┼──────────────┼─────────────┼─────────────────┤
│ 3. trim_messages ⭐  │ Full history │ Smart trim  │ Production use  │
│                     │              │ (by tokens) │ Cost control    │
└─────────────────────┴──────────────┴─────────────┴─────────────────┘

RECOMMENDATION: Use trim_messages for production!
- Most accurate token control
- Prevents context limit errors
- Optimizes cost
- Built-in token counting
"""

print(comparison)


# =============================================================================
# PART 6: Advanced trim_messages Parameters
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 6: Understanding trim_messages Parameters")
print("=" * 70)

print("""
trim_messages(
    messages,                   # List of messages to trim
    max_tokens=100,            # Maximum tokens to keep
    strategy="last",           # "last" or "first" (which messages to keep)
    token_counter=ChatOpenAI(), # Model to use for counting tokens
    allow_partial=False        # Allow splitting messages mid-content?
)

📚 Parameter Explanations:

1. max_tokens=100
   - Keep only messages that fit in 100 tokens
   - Prevents exceeding model context limits
   - Controls API costs

2. strategy="last"
   - "last": Keep newest messages (most common)
   - "first": Keep oldest messages (rare)
   - Most chatbots use "last" for recent context

3. token_counter=ChatOpenAI(model="gpt-4o")
   - Uses GPT-4o's tokenizer to count
   - Different models count differently!
   - Always match your actual model

4. allow_partial=False
   - False: Keep whole messages only (recommended)
   - True: Can split message content mid-sentence
   - False keeps messages coherent

🎯 Common Configurations:

# Conservative (low cost):
trim_messages(msgs, max_tokens=500, strategy="last")

# Moderate (balanced):
trim_messages(msgs, max_tokens=2000, strategy="last")

# Large context (expensive but comprehensive):
trim_messages(msgs, max_tokens=8000, strategy="last")
""")


# =============================================================================
# PART 7: Real-World Example - Production Chatbot
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 7: Production Chatbot with Smart Trimming")
print("=" * 70)

class ProductionChatbotState(MessagesState):
    """Extended state with metadata"""
    user_id: str
    session_id: str


def production_chat_node(state: ProductionChatbotState):
    """
    Production-ready chat node with smart message trimming.
    """
    print(f"\n👤 User: {state.get('user_id', 'N/A')}")
    print(f"🔑 Session: {state.get('session_id', 'N/A')}")
    print(f"📊 Total messages: {len(state['messages'])}")

    # Trim to 2000 tokens for production balance
    trimmed = trim_messages(
        state["messages"],
        max_tokens=2000,
        strategy="last",
        token_counter=ChatOpenAI(model="gpt-4o"),
        allow_partial=False
    )

    print(f"✂️  Trimmed to: {len(trimmed)} messages (~2000 tokens)")

    # Invoke LLM with trimmed messages
    response = llm.invoke(trimmed)

    return {"messages": [response]}


# Build production chatbot
prod_builder = StateGraph(ProductionChatbotState)
prod_builder.add_node("chat", production_chat_node)
prod_builder.add_edge(START, "chat")
prod_builder.add_edge("chat", END)
prod_chatbot = prod_builder.compile()

# Test production chatbot
print("\n🚀 Running production chatbot:")
result = prod_chatbot.invoke({
    "messages": long_messages,
    "user_id": "user_123",
    "session_id": "session_abc"
})

print(f"\n✅ Response generated successfully!")
print(f"📊 Final state has {len(result['messages'])} messages")


# =============================================================================
# Summary
# =============================================================================

print("\n\n" + "=" * 70)
print("📚 KEY TAKEAWAYS")
print("=" * 70)
print("""
1. MESSAGE MANAGEMENT IS CRITICAL:
   - Long conversations = high costs
   - Must manage message history
   - Three main approaches available

2. METHOD 1 - RemoveMessage:
   - Deletes old messages from state
   - Syntax: [RemoveMessage(id=m.id) for m in messages[:-2]]
   - Use when: Memory is constrained

3. METHOD 2 - Simple Slicing:
   - Keep full state, filter LLM input
   - Syntax: llm.invoke(messages[-2:])
   - Use when: Need full history log

4. METHOD 3 - trim_messages ⭐ RECOMMENDED:
   - Smart token-based trimming
   - Accurate cost control
   - Production-ready
   - Use when: Building real applications

5. PRODUCTION BEST PRACTICES:
   ✓ Use trim_messages with max_tokens
   ✓ Match token_counter to your model
   ✓ Set allow_partial=False
   ✓ Use strategy="last" for recent context
   ✓ Monitor token usage in production

6. TYPICAL TOKEN LIMITS:
   - Conservative: 500-1000 tokens
   - Balanced: 2000-4000 tokens
   - Large context: 8000+ tokens
   (Adjust based on your model and budget)
""")
print("=" * 70)
