"""
Module 1: Agent with Memory - Persistent Conversations

This implementation extends the basic ReAct agent with memory/persistence.
The agent can now remember previous conversations across multiple interactions!

Key Concepts:
- Checkpointer: Saves graph state after each step
- Thread ID: Identifier for conversation threads
- Persistence: Agent remembers context from earlier in the conversation
- Multi-turn conversations: Natural follow-up questions work

Memory Flow:
    User: "Add 3 and 4"     → Agent: "Result is 7"
    User: "Multiply by 2"    → Agent knows "that" = 7 → "Result is 14"

Without memory, the second question would fail - agent wouldn't know what to multiply!

Author: Klement G
Date: 2026-01-10
Course: LangChain Academy - Module 1
"""

import os
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# PART 1: DEFINE TOOLS (Same as basic agent)
# ============================================================================

def add(a: int, b: int) -> int:
    """Adds two numbers together.

    Args:
        a: First integer
        b: Second integer

    Returns:
        The sum of a and b
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiplies two numbers together.

    Args:
        a: First integer
        b: Second integer

    Returns:
        The product of a and b
    """
    return a * b


def divide(a: int, b: int) -> float:
    """Divides the first number by the second.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        The quotient of a divided by b
    """
    if b == 0:
        return "Error: Cannot divide by zero"
    return a / b


def subtract(a: int, b: int) -> int:
    """Subtracts the second number from the first.

    Args:
        a: Number to subtract from
        b: Number to subtract

    Returns:
        The difference
    """
    return a - b


# ============================================================================
# PART 2: CONFIGURE LLM
# ============================================================================

tools = [add, multiply, divide, subtract]
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)


# ============================================================================
# PART 3: DEFINE THE ASSISTANT NODE
# ============================================================================

sys_msg = SystemMessage(
    content="""You are a helpful mathematical assistant with memory.

You can remember previous calculations in this conversation and refer back to them.

Your capabilities:
1. Perform arithmetic operations (add, subtract, multiply, divide)
2. Remember previous results in the conversation
3. Understand references like "that", "the result", "the previous answer"
4. Chain multiple operations together

Always use tools for calculations - never estimate."""
)


def assistant(state: MessagesState):
    """
    The assistant node with memory context.

    Key difference from basic agent: The entire conversation history
    is preserved, so the LLM can reference previous results.

    Args:
        state: Current conversation state with ALL previous messages

    Returns:
        Updated state with new AI message
    """
    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}


# ============================================================================
# PART 4: BUILD THE AGENT GRAPH WITH MEMORY
# ============================================================================

def create_agent_with_memory():
    """
    Creates an agent with persistent memory.

    The key difference: compile(checkpointer=memory)

    How Memory Works:
    1. MemorySaver checkpointer saves state after each step
    2. States are organized by thread_id
    3. When you invoke with the same thread_id, it loads previous state
    4. This gives the agent conversation memory!

    Returns:
        Compiled agent graph with memory enabled
    """
    # Initialize graph builder
    builder = StateGraph(MessagesState)

    # Add nodes (same as before)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Define edges (same as before)
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    # ⭐ THE KEY DIFFERENCE: Add a checkpointer for memory
    memory = MemorySaver()

    # Compile with checkpointer - this enables persistence!
    return builder.compile(checkpointer=memory)


# ============================================================================
# PART 5: USING MEMORY - THREAD MANAGEMENT
# ============================================================================

class ConversationManager:
    """
    Manages multi-turn conversations with memory.

    Each conversation has a thread_id. The agent remembers everything
    within the same thread.
    """

    def __init__(self):
        self.agent = create_agent_with_memory()
        self.thread_counter = 1

    def new_conversation(self):
        """
        Start a new conversation thread.

        Returns:
            New thread_id
        """
        thread_id = f"thread_{self.thread_counter}"
        self.thread_counter += 1
        print(f"\n🆕 Started new conversation: {thread_id}")
        return thread_id

    def send_message(self, query: str, thread_id: str):
        """
        Send a message in a conversation thread.

        Args:
            query: User's message
            thread_id: Conversation thread to use

        Returns:
            Agent's response
        """
        # Configuration specifies which thread to use
        config = {"configurable": {"thread_id": thread_id}}

        # Create message
        messages = [HumanMessage(content=query)]

        # Invoke agent with thread config
        # The agent will load previous state from this thread!
        result = self.agent.invoke({"messages": messages}, config)

        # Return final message
        return result["messages"][-1].content

    def show_conversation_history(self, thread_id: str):
        """
        Display all messages in a conversation thread.

        Args:
            thread_id: Thread to display
        """
        config = {"configurable": {"thread_id": thread_id}}

        # Get current state for this thread
        state = self.agent.get_state(config)

        print(f"\n📜 Conversation History (Thread: {thread_id})")
        print("="*80)

        for msg in state.values['messages']:
            if hasattr(msg, 'pretty_print'):
                msg.pretty_print()
            print()


# ============================================================================
# PART 6: DEMO - MEMORY IN ACTION
# ============================================================================

def demo_memory():
    """
    Demonstrates how memory enables multi-turn conversations.
    """
    print("\n" + "="*80)
    print("🧠 AGENT WITH MEMORY - DEMONSTRATION")
    print("="*80)

    manager = ConversationManager()

    # Start a conversation thread
    thread = manager.new_conversation()

    print("\n" + "="*80)
    print("TURN 1: Initial Calculation")
    print("="*80)

    query1 = "Add 15 and 27"
    print(f"\n👤 User: {query1}")
    response1 = manager.send_message(query1, thread)
    print(f"🤖 Agent: {response1}")

    print("\n" + "="*80)
    print("TURN 2: Follow-up (references previous result)")
    print("="*80)

    query2 = "Multiply that by 3"
    print(f"\n👤 User: {query2}")
    print("   ⚠️  Notice: 'that' refers to the previous result (42)")
    response2 = manager.send_message(query2, thread)
    print(f"🤖 Agent: {response2}")

    print("\n" + "="*80)
    print("TURN 3: Another follow-up")
    print("="*80)

    query3 = "Now divide the result by 7"
    print(f"\n👤 User: {query3}")
    print("   ⚠️  Agent remembers: 42 * 3 = 126")
    response3 = manager.send_message(query3, thread)
    print(f"🤖 Agent: {response3}")

    # Show full conversation
    print("\n" + "="*80)
    print("FULL CONVERSATION REPLAY")
    print("="*80)
    manager.show_conversation_history(thread)

    print("\n" + "="*80)
    print("💡 KEY INSIGHT: Without memory, this wouldn't work!")
    print("="*80)
    print("\nWithout memory:")
    print("  User: 'Multiply that by 3'")
    print("  Agent: '❌ What should I multiply? I don't remember.'")
    print("\nWith memory:")
    print("  User: 'Multiply that by 3'")
    print("  Agent: '✅ I remember 42 from before. 42 * 3 = 126'")
    print("="*80 + "\n")


# ============================================================================
# PART 7: DEMO - MULTIPLE CONVERSATIONS
# ============================================================================

def demo_multiple_threads():
    """
    Demonstrates separate conversation threads.
    """
    print("\n" + "="*80)
    print("🧵 MULTIPLE CONVERSATION THREADS")
    print("="*80)

    manager = ConversationManager()

    # Conversation 1
    thread1 = manager.new_conversation()
    print("\n👥 Conversation with Alice:")
    print(f"Alice: {manager.send_message('Add 10 and 20', thread1)}")
    print(f"Alice: {manager.send_message('Double it', thread1)}")

    # Conversation 2
    thread2 = manager.new_conversation()
    print("\n👥 Conversation with Bob:")
    print(f"Bob: {manager.send_message('Multiply 5 and 8', thread2)}")
    print(f"Bob: {manager.send_message('Subtract 15', thread2)}")

    # Back to conversation 1
    print("\n👥 Back to Alice:")
    print(f"Alice: {manager.send_message('Add 5 to that', thread1)}")
    print("   ⚠️  'that' refers to 60 from Alice's conversation, not Bob's!")

    print("\n" + "="*80)
    print("💡 Each thread maintains its own separate memory!")
    print("="*80 + "\n")


# ============================================================================
# PART 8: INTERACTIVE MODE WITH MEMORY
# ============================================================================

def interactive_mode():
    """
    Interactive chat with memory.
    """
    print("\n" + "="*80)
    print("💬 INTERACTIVE AGENT WITH MEMORY")
    print("="*80)
    print("\nCommands:")
    print("  - Type your math question")
    print("  - 'new' to start a new conversation")
    print("  - 'history' to see conversation history")
    print("  - 'quit' to exit")
    print("="*80 + "\n")

    manager = ConversationManager()
    current_thread = manager.new_conversation()

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break

            if user_input.lower() == 'new':
                current_thread = manager.new_conversation()
                continue

            if user_input.lower() == 'history':
                manager.show_conversation_history(current_thread)
                continue

            if not user_input:
                continue

            # Send message and get response
            response = manager.send_message(user_input, current_thread)
            print(f"\n🤖 Agent: {response}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


# ============================================================================
# RUN THE PROGRAM
# ============================================================================

def main():
    """
    Main demonstration function.
    """
    print("\n" + "="*80)
    print("🚀 LANGGRAPH AGENT WITH MEMORY")
    print("="*80)

    # Demo 1: Memory in action
    demo_memory()

    # Demo 2: Multiple threads
    demo_multiple_threads()

    print("\n" + "="*80)
    print("📚 SUMMARY: How Memory Works")
    print("="*80)
    print("""
1. **Checkpointer**: Saves graph state after each step
2. **Thread ID**: Identifies which conversation to use
3. **State Persistence**: Entire conversation history is preserved
4. **Context Awareness**: Agent can reference previous results

Memory enables:
✅ Natural follow-up questions
✅ References to previous results ("that", "the result", etc.)
✅ Multi-turn conversations
✅ Multiple separate conversation threads
    """)
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()
        print("\nTip: Run with --interactive for chat mode:")
        print("    python 05_agent_with_memory.py --interactive\n")
