"""
Module 1: Agent - ReAct Pattern Implementation

This is a complete implementation of the ReAct (Reason + Act) agent pattern.
The agent can autonomously use multiple tools in sequence to solve complex tasks.

Key Concepts:
- ReAct Pattern: Reason → Act → Observe → Repeat
- Agent Loop: Tool results feed back to the LLM for next decision
- Autonomous Execution: Agent decides when to stop

Flow:
    START → assistant → tools_condition → tools → assistant (loop) → END

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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# PART 1: DEFINE TOOLS
# ============================================================================
# Tools are functions the agent can use to accomplish tasks.
# The LLM reads the docstrings to understand what each tool does.

def add(a: int, b: int) -> int:
    """Adds two numbers together.

    Use this tool when you need to perform addition.

    Args:
        a: First integer to add
        b: Second integer to add

    Returns:
        The sum of a and b
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiplies two numbers together.

    Use this tool when you need to perform multiplication.

    Args:
        a: First integer to multiply
        b: Second integer to multiply

    Returns:
        The product of a and b
    """
    return a * b


def divide(a: int, b: int) -> float:
    """Divides the first number by the second number.

    Use this tool when you need to perform division.

    Args:
        a: Numerator (number to be divided)
        b: Denominator (number to divide by)

    Returns:
        The quotient of a divided by b
    """
    if b == 0:
        return "Error: Division by zero is not allowed"
    return a / b


def subtract(a: int, b: int) -> int:
    """Subtracts the second number from the first number.

    Use this tool when you need to perform subtraction.

    Args:
        a: Number to subtract from
        b: Number to subtract

    Returns:
        The difference of a minus b
    """
    return a - b


# ============================================================================
# PART 2: CONFIGURE THE LLM WITH TOOLS
# ============================================================================

# List of all available tools
tools = [add, multiply, divide, subtract]

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Bind tools to the LLM
# parallel_tool_calls=False ensures sequential execution (important for math)
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)


# ============================================================================
# PART 3: DEFINE THE ASSISTANT NODE
# ============================================================================

# System message defines the agent's behavior
sys_msg = SystemMessage(
    content="""You are a helpful mathematical assistant that can perform arithmetic operations.

Your job is to:
1. Analyze the user's request
2. Break it down into steps
3. Use the available tools (add, subtract, multiply, divide) to solve it
4. Show your work clearly
5. Provide the final answer

Always use tools for calculations - never guess or estimate numbers."""
)


def assistant(state: MessagesState):
    """
    The assistant node - this is where the LLM makes decisions.

    Flow:
    1. Receives current state (all messages so far)
    2. Adds system message + conversation history
    3. Invokes LLM to decide next action
    4. Returns LLM response (could be tool call or final answer)

    Args:
        state: Current conversation state with all messages

    Returns:
        Updated state with new AI message
    """
    # Invoke LLM with system message + full conversation history
    response = llm_with_tools.invoke([sys_msg] + state["messages"])

    # Return updated state
    return {"messages": [response]}


# ============================================================================
# PART 4: BUILD THE AGENT GRAPH
# ============================================================================

def create_agent():
    """
    Creates and compiles the agent graph.

    Graph Structure:
        START → assistant → tools_condition (decision point)
                              ↓ (if tool call)  ↓ (if no tool call)
                           tools              END
                              ↓
                           assistant (loop back)

    The loop (tools → assistant) is what makes this an agent!
    The agent keeps calling tools until the task is complete.

    Returns:
        Compiled agent graph ready for execution
    """
    # Initialize graph builder with MessagesState
    builder = StateGraph(MessagesState)

    # Add nodes
    builder.add_node("assistant", assistant)  # LLM decision maker
    builder.add_node("tools", ToolNode(tools))  # Tool executor

    # Define edges
    builder.add_edge(START, "assistant")  # Always start at assistant

    # Conditional edge: routes based on LLM's decision
    builder.add_conditional_edges(
        "assistant",
        tools_condition,  # Checks if LLM called a tool
        # If tool call → goes to "tools"
        # If no tool call → goes to END
    )

    # The critical loop: after tools execute, go back to assistant
    # This allows the agent to call multiple tools in sequence
    builder.add_edge("tools", "assistant")

    # Compile the graph
    return builder.compile()


# ============================================================================
# PART 5: HELPER FUNCTIONS
# ============================================================================

def print_messages(messages):
    """
    Pretty print all messages in the conversation.

    Args:
        messages: List of messages to print
    """
    print("\n" + "="*80)
    print("AGENT EXECUTION TRACE")
    print("="*80 + "\n")

    for i, msg in enumerate(messages, 1):
        if hasattr(msg, 'type'):
            msg_type = msg.type
        else:
            msg_type = msg.__class__.__name__

        print(f"Step {i}: {msg_type.upper()}")
        print("-" * 80)

        if hasattr(msg, 'pretty_print'):
            msg.pretty_print()
        else:
            print(msg.content if hasattr(msg, 'content') else str(msg))
        print()


def run_agent(query: str, verbose: bool = True):
    """
    Run the agent with a query and display results.

    Args:
        query: User's question or task
        verbose: Whether to print execution trace

    Returns:
        Final state with all messages
    """
    # Create the agent
    agent = create_agent()

    # Create initial state with user message
    initial_state = {"messages": [HumanMessage(content=query)]}

    # Run the agent
    print(f"\n🤖 Running agent with query: '{query}'")
    print("="*80)

    final_state = agent.invoke(initial_state)

    # Print results
    if verbose:
        print_messages(final_state["messages"])

    # Extract final answer
    last_message = final_state["messages"][-1]
    print("\n" + "="*80)
    print("FINAL ANSWER")
    print("="*80)
    print(last_message.content)
    print("="*80 + "\n")

    return final_state


# ============================================================================
# PART 6: EXAMPLE USAGE & TESTING
# ============================================================================

def main():
    """
    Main function demonstrating the agent in action.
    """
    print("\n" + "="*80)
    print("🚀 LANGGRAPH AGENT - REACT PATTERN DEMO")
    print("="*80)
    print("\nThis agent can solve complex math problems using multiple tools.")
    print("Watch how it breaks down the problem and uses tools autonomously!\n")

    # Example 1: Simple sequential operations
    print("\n" + "="*80)
    print("EXAMPLE 1: Sequential Operations")
    print("="*80)
    run_agent("Add 15 and 27. Then multiply the result by 3. Finally divide by 7.")

    # Example 2: Complex calculation
    print("\n" + "="*80)
    print("EXAMPLE 2: Complex Calculation")
    print("="*80)
    run_agent("Calculate: (100 + 50) * 2 - 75")

    # Example 3: Multiple operations
    print("\n" + "="*80)
    print("EXAMPLE 3: Multiple Steps")
    print("="*80)
    run_agent("What is 8 times 7? Then subtract 12. Then divide by 4.")

    print("\n" + "="*80)
    print("✅ Demo Complete!")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. The agent autonomously decides which tools to use")
    print("2. It calls tools in sequence until the task is complete")
    print("3. Tool results feed back to the LLM for next decision")
    print("4. This is the ReAct pattern: Reason → Act → Observe → Repeat")
    print("="*80 + "\n")


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode():
    """
    Interactive mode - chat with the agent.
    """
    print("\n" + "="*80)
    print("🤖 INTERACTIVE AGENT MODE")
    print("="*80)
    print("\nAsk the agent to perform mathematical calculations.")
    print("Type 'quit' to exit.\n")

    agent = create_agent()

    while True:
        try:
            query = input("You: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break

            if not query:
                continue

            # Run agent
            state = {"messages": [HumanMessage(content=query)]}
            result = agent.invoke(state)

            # Print final answer
            final_message = result["messages"][-1]
            print(f"\nAgent: {final_message.content}\n")
            print("-" * 80 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


# ============================================================================
# RUN THE PROGRAM
# ============================================================================

if __name__ == "__main__":
    import sys

    # Check if interactive mode requested
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()

        print("\nTip: Run with --interactive flag for chat mode:")
        print("    python 04_agent.py --interactive\n")
