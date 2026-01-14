"""
Module 4 - Lesson 1: Parallelization (Fan-Out & Fan-In)

This module demonstrates how to run multiple nodes in parallel using
fan-out and fan-in patterns for efficient execution.

Key Concepts:
- Fan-out: One node branches to multiple nodes
- Fan-in: Multiple nodes converge to one node
- Reducers for parallel state updates
- Custom reducers for ordering
- Parallel data gathering (Wikipedia + Web Search)

Learning Outcomes:
✓ Run multiple nodes simultaneously
✓ Use reducers to combine parallel writes
✓ Build multi-source research workflows
✓ Control state update ordering
✓ Optimize execution time with parallelization

Author: Klement G
Date: 2026-01-14
Course: LangChain Academy - Introduction to LangGraph
"""

import operator
from typing import Any, List, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_tavily import TavilySearch


# =============================================================================
# PART 1: BASIC PARALLELIZATION CONCEPTS
# =============================================================================

class SimpleState(TypedDict):
    """State without reducer - will fail with parallel writes"""
    state: List[str]


class ReturnNodeValue:
    """Helper class that returns a node's identifier to state"""
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: SimpleState) -> Any:
        print(f"Adding {self._value} to {state['state']}")
        return {"state": [self._value]}


def run_linear_graph_example():
    """
    Example 1: Linear graph (no parallelization).

    Flow: START → a → b → c → d → END
    Each node overwrites state sequentially.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Linear Graph (Sequential)")
    print("="*70)

    builder = StateGraph(SimpleState)

    # Add nodes
    builder.add_node("a", ReturnNodeValue("I'm A"))
    builder.add_node("b", ReturnNodeValue("I'm B"))
    builder.add_node("c", ReturnNodeValue("I'm C"))
    builder.add_node("d", ReturnNodeValue("I'm D"))

    # Linear flow
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", "c")
    builder.add_edge("c", "d")
    builder.add_edge("d", END)

    graph = builder.compile()

    print("\n📊 Graph Structure:")
    print("   START → a → b → c → d → END")

    print("\n▶️ Running graph...")
    result = graph.invoke({"state": []})

    print(f"\n✅ Final Result: {result['state']}")
    print("   Each node overwrites the previous state")


def run_parallel_without_reducer_example():
    """
    Example 2: Parallel execution WITHOUT reducer (FAILS).

    Flow: START → a → [b, c] → d → END

    This will fail because both b and c try to write to 'state'
    in the same step without a reducer to combine them.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Parallel Without Reducer ❌ FAILS")
    print("="*70)

    builder = StateGraph(SimpleState)

    builder.add_node("a", ReturnNodeValue("I'm A"))
    builder.add_node("b", ReturnNodeValue("I'm B"))
    builder.add_node("c", ReturnNodeValue("I'm C"))
    builder.add_node("d", ReturnNodeValue("I'm D"))

    # Fan-out from a to b and c (parallel)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("a", "c")

    # Fan-in from b and c to d
    builder.add_edge("b", "d")
    builder.add_edge("c", "d")
    builder.add_edge("d", END)

    graph = builder.compile()

    print("\n📊 Graph Structure:")
    print("          ┌─→ b ─┐")
    print("   START → a     ├→ d → END")
    print("          └─→ c ─┘")

    print("\n▶️ Attempting to run graph...")

    try:
        graph.invoke({"state": []})
        print("\n✅ Success (unexpected!)")
    except Exception as e:
        print(f"\n❌ Error (expected): {str(e)[:100]}...")
        print("\n💡 Problem: Both b and c try to write to 'state' in parallel")
        print("   Solution: Use a reducer to combine parallel writes!")


def run_parallel_with_reducer_example():
    """
    Example 3: Parallel execution WITH reducer (SUCCESS).

    Using operator.add as reducer allows combining parallel writes.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Parallel With Reducer ✅ SUCCESS")
    print("="*70)

    class StateWithReducer(TypedDict):
        # operator.add concatenates lists!
        state: Annotated[list, operator.add]

    builder = StateGraph(StateWithReducer)

    builder.add_node("a", ReturnNodeValue("I'm A"))
    builder.add_node("b", ReturnNodeValue("I'm B"))
    builder.add_node("c", ReturnNodeValue("I'm C"))
    builder.add_node("d", ReturnNodeValue("I'm D"))

    # Same structure as Example 2
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("a", "c")
    builder.add_edge("b", "d")
    builder.add_edge("c", "d")
    builder.add_edge("d", END)

    graph = builder.compile()

    print("\n📊 Graph Structure:")
    print("          ┌─→ b ─┐")
    print("   START → a     ├→ d → END")
    print("          └─→ c ─┘")
    print("\n💡 State has reducer: Annotated[list, operator.add]")

    print("\n▶️ Running graph...")
    result = graph.invoke({"state": []})

    print(f"\n✅ Final Result: {result['state']}")
    print("   All values preserved! Reducer combined parallel writes.")


def run_uneven_parallel_paths_example():
    """
    Example 4: Parallel paths with different lengths.

    One path: a → b → b2
    Other path: a → c
    Both converge at d

    The graph waits for ALL parallel paths to complete.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Uneven Parallel Paths")
    print("="*70)

    class StateWithReducer(TypedDict):
        state: Annotated[list, operator.add]

    builder = StateGraph(StateWithReducer)

    builder.add_node("a", ReturnNodeValue("I'm A"))
    builder.add_node("b", ReturnNodeValue("I'm B"))
    builder.add_node("b2", ReturnNodeValue("I'm B2"))
    builder.add_node("c", ReturnNodeValue("I'm C"))
    builder.add_node("d", ReturnNodeValue("I'm D"))

    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("a", "c")
    builder.add_edge("b", "b2")  # Extra step in b path
    builder.add_edge(["b2", "c"], "d")  # Wait for BOTH!
    builder.add_edge("d", END)

    graph = builder.compile()

    print("\n📊 Graph Structure:")
    print("          ┌─→ b → b2 ─┐")
    print("   START → a          ├→ d → END")
    print("          └─→ c ──────┘")
    print("\n💡 d waits for BOTH b2 AND c to complete")

    print("\n▶️ Running graph...")
    result = graph.invoke({"state": []})

    print(f"\n✅ Final Result: {result['state']}")
    print("   b2 and c both completed before d ran")


def run_custom_reducer_example():
    """
    Example 5: Custom reducer for sorting.

    By default, parallel updates happen in non-deterministic order.
    Custom reducer allows control over ordering.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Custom Reducer (Sorting)")
    print("="*70)

    def sorting_reducer(left, right):
        """Combines and sorts the values in a list"""
        if not isinstance(left, list):
            left = [left]
        if not isinstance(right, list):
            right = [right]

        return sorted(left + right, reverse=False)

    class StateWithSorting(TypedDict):
        state: Annotated[list, sorting_reducer]

    builder = StateGraph(StateWithSorting)

    builder.add_node("a", ReturnNodeValue("I'm A"))
    builder.add_node("b", ReturnNodeValue("I'm B"))
    builder.add_node("b2", ReturnNodeValue("I'm B2"))
    builder.add_node("c", ReturnNodeValue("I'm C"))
    builder.add_node("d", ReturnNodeValue("I'm D"))

    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("a", "c")
    builder.add_edge("b", "b2")
    builder.add_edge(["b2", "c"], "d")
    builder.add_edge("d", END)

    graph = builder.compile()

    print("\n📊 Graph Structure:")
    print("          ┌─→ b → b2 ─┐")
    print("   START → a          ├→ d → END")
    print("          └─→ c ──────┘")
    print("\n💡 Custom sorting_reducer sorts all values alphabetically")

    print("\n▶️ Running graph...")
    result = graph.invoke({"state": []})

    print(f"\n✅ Final Result: {result['state']}")
    print("   Values are sorted! Custom reducer controls ordering.")


# =============================================================================
# PART 2: REAL-WORLD APPLICATION - PARALLEL SEARCH
# =============================================================================

class ResearchState(TypedDict):
    """State for research assistant with parallel search"""
    question: str
    answer: str
    context: Annotated[list, operator.add]


def search_web(state: ResearchState) -> dict:
    """
    Retrieve documents from web search using Tavily.

    This runs in parallel with search_wikipedia.
    """
    print("\n🌐 Searching web...")

    try:
        tavily_search = TavilySearch(max_results=3)
        data = tavily_search.invoke({"query": state['question']})
        search_docs = data.get("results", data)

        formatted = "\n\n---\n\n".join([
            f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ])

        print(f"   ✓ Found {len(search_docs)} web results")
        return {"context": [formatted]}

    except Exception as e:
        print(f"   ⚠️ Web search failed: {e}")
        return {"context": ["Web search unavailable"]}


def search_wikipedia(state: ResearchState) -> dict:
    """
    Retrieve documents from Wikipedia.

    This runs in parallel with search_web.
    """
    print("\n📚 Searching Wikipedia...")

    try:
        search_docs = WikipediaLoader(
            query=state['question'],
            load_max_docs=2
        ).load()

        formatted = "\n\n---\n\n".join([
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}">\n'
            f'{doc.page_content}\n</Document>'
            for doc in search_docs
        ])

        print(f"   ✓ Found {len(search_docs)} Wikipedia results")
        return {"context": [formatted]}

    except Exception as e:
        print(f"   ⚠️ Wikipedia search failed: {e}")
        return {"context": ["Wikipedia search unavailable"]}


def generate_answer(state: ResearchState) -> dict:
    """
    Generate answer using context from parallel searches.

    This node waits for BOTH search_web and search_wikipedia to complete.
    """
    print("\n🤖 Generating answer...")

    context = state["context"]
    question = state["question"]

    # Create prompt
    prompt = f"Answer the question '{question}' using this context: {context}"

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Generate answer
    answer = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Answer the question concisely.")
    ])

    print("   ✓ Answer generated")

    return {"answer": answer}


def run_parallel_research_example():
    """
    Example 6: Real-world parallel research assistant.

    Searches Wikipedia AND web simultaneously, then generates answer.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Parallel Research Assistant")
    print("="*70)

    builder = StateGraph(ResearchState)

    # Add nodes
    builder.add_node("search_web", search_web)
    builder.add_node("search_wikipedia", search_wikipedia)
    builder.add_node("generate_answer", generate_answer)

    # Parallel search (fan-out from START)
    builder.add_edge(START, "search_wikipedia")
    builder.add_edge(START, "search_web")

    # Fan-in to answer generation
    builder.add_edge("search_wikipedia", "generate_answer")
    builder.add_edge("search_web", "generate_answer")
    builder.add_edge("generate_answer", END)

    graph = builder.compile()

    print("\n📊 Graph Structure:")
    print("          ┌─→ search_wikipedia ─┐")
    print("   START ─┤                      ├→ generate_answer → END")
    print("          └─→ search_web ────────┘")
    print("\n💡 Both searches run simultaneously for faster results")

    # Example question
    question = "What is LangChain?"

    print(f"\n❓ Question: {question}")
    print("\n▶️ Running parallel research...")

    result = graph.invoke({"question": question})

    print("\n" + "="*70)
    print("📝 FINAL ANSWER")
    print("="*70)
    print(result['answer'].content)

    print("\n" + "="*70)
    print("✅ Research Complete!")
    print("   Both sources searched in parallel")
    print("   Answer generated from combined context")


# =============================================================================
# KEY LEARNINGS
# =============================================================================

def print_key_concepts():
    """Print key concepts for parallelization."""
    print("\n" + "="*70)
    print("📚 KEY CONCEPTS: Parallelization")
    print("="*70)

    concepts = """
    1. FAN-OUT & FAN-IN PATTERN:
       - Fan-out: One node → multiple nodes
       - Fan-in: Multiple nodes → one node
       - Enables parallel execution
       - Graph waits for ALL parallel paths

    2. REDUCERS ARE REQUIRED:
       - Without reducer: ERROR in parallel execution
       - With reducer: Combines parallel writes
       - operator.add: List concatenation
       - Custom reducers: Custom logic

    3. OPERATOR.ADD FOR LISTS:
       - [1, 2] + [3] = [1, 2, 3]
       - Appends values from parallel nodes
       - Most common reducer for parallel execution

    4. CUSTOM REDUCERS:
       - Control ordering (sorting)
       - Custom merge logic
       - Handle complex state updates
       - Example: sorting_reducer

    5. PARALLEL EXECUTION TIMING:
       - Nodes in same step run simultaneously
       - Graph waits for slowest path
       - Overall execution time = longest path time
       - Not shortest path + shortest path!

    6. REAL-WORLD BENEFITS:
       - Faster execution (concurrent operations)
       - Multi-source data gathering
       - Independent API calls
       - Parallel processing pipelines

    7. WHEN TO USE:
       ✓ Independent operations (can run separately)
       ✓ I/O-bound tasks (API calls, searches)
       ✓ Multi-source data gathering
       ✓ No dependencies between operations
       ✗ Sequential dependencies
       ✗ Shared mutable resources (without reducer)

    8. GRAPH SYNCHRONIZATION:
       - Fan-in node waits for ALL fan-out nodes
       - Use list of nodes: ["node1", "node2"]
       - Ensures data from all sources available
    """
    print(concepts)


def print_comparison_table():
    """Print comparison of sequential vs parallel execution."""
    print("\n" + "="*70)
    print("📊 COMPARISON: Sequential vs Parallel")
    print("="*70)

    comparison = """
    SEQUENTIAL EXECUTION:
    ────────────────────────────────────────────
    Structure:  START → a → b → c → END
    Time:       t_a + t_b + t_c
    State:      No reducer needed
    Use case:   Dependent operations

    PARALLEL EXECUTION:
    ────────────────────────────────────────────
                    ┌─→ b ─┐
    Structure:  a ──┤      ├─→ d
                    └─→ c ─┘

    Time:       t_a + max(t_b, t_c) + t_d
    State:      Reducer REQUIRED
    Use case:   Independent operations

    EXAMPLE: Web + Wikipedia Search
    ────────────────────────────────────────────
    Sequential:  3s + 2s = 5s total
    Parallel:    max(3s, 2s) = 3s total
    Speedup:     40% faster! ⚡
    """
    print(comparison)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "🚀"*35)
    print("MODULE 4 - LESSON 1: PARALLELIZATION")
    print("🚀"*35)
    print("\n💡 Concept: Run multiple nodes simultaneously for faster execution")

    # Basic examples
    run_linear_graph_example()
    run_parallel_without_reducer_example()
    run_parallel_with_reducer_example()
    run_uneven_parallel_paths_example()
    run_custom_reducer_example()

    # Real-world example
    print("\n" + "="*70)
    print("🌟 REAL-WORLD APPLICATION")
    print("="*70)

    try:
        run_parallel_research_example()
    except Exception as e:
        print(f"\n⚠️ Research example requires API keys:")
        print(f"   - OPENAI_API_KEY")
        print(f"   - TAVILY_API_KEY")
        print(f"\n   Error: {e}")

    # Show key concepts
    print_key_concepts()
    print_comparison_table()

    print("\n" + "="*70)
    print("✅ Module 4, Lesson 1 Complete!")
    print("="*70)
    print("\n📖 Next: map_reduce.py - Process multiple items in parallel")
    print()
