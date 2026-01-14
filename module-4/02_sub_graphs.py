"""
Module 4 - Lesson 2: Sub-Graphs

This module demonstrates how to create modular, reusable graph components
using sub-graphs with their own state schemas.

Key Concepts:
- Sub-graphs as modular components
- State communication through overlapping keys
- Input and output state schemas
- Parallel sub-graph execution
- Multi-agent team patterns

Learning Outcomes:
✓ Create sub-graphs with custom states
✓ Compose sub-graphs into parent graphs
✓ Control data flow with state schemas
✓ Build modular multi-agent systems
✓ Design team-based agent architectures

Author: Klement G
Date: 2026-01-14
Course: LangChain Academy - Introduction to LangGraph
"""

from operator import add
from typing import List, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# =============================================================================
# DOMAIN MODELS
# =============================================================================

class Log(TypedDict):
    """Structure for log entries"""
    id: str
    question: str
    docs: Optional[List]
    answer: str
    grade: Optional[int]
    grader: Optional[str]
    feedback: Optional[str]


# =============================================================================
# SUB-GRAPH 1: FAILURE ANALYSIS
# =============================================================================

class FailureAnalysisState(TypedDict):
    """
    Internal state for Failure Analysis sub-graph.

    This is the FULL state available inside the sub-graph.
    """
    cleaned_logs: List[Log]  # Input from parent
    failures: List[Log]      # Internal state
    fa_summary: str          # Output to parent
    processed_logs: List[str]  # Output to parent


class FailureAnalysisOutputState(TypedDict):
    """
    Output state for Failure Analysis sub-graph.

    Only these keys will be returned to the parent graph.
    This prevents unnecessary data from being passed back.
    """
    fa_summary: str
    processed_logs: List[str]


def get_failures(state: FailureAnalysisState) -> dict:
    """
    Node: Extract logs that contain failures.

    Filters logs to find those with grade < passing threshold.
    """
    print("\n  🔍 [FA] Analyzing logs for failures...")

    cleaned_logs = state["cleaned_logs"]

    # Find logs with grade (indicating they were graded/reviewed)
    failures = [log for log in cleaned_logs if "grade" in log and log["grade"] == 0]

    print(f"     Found {len(failures)} failures out of {len(cleaned_logs)} logs")

    return {"failures": failures}


def generate_fa_summary(state: FailureAnalysisState) -> dict:
    """
    Node: Generate summary of failures.

    In production, this would use an LLM to analyze and summarize.
    """
    print("\n  📝 [FA] Generating failure analysis summary...")

    failures = state["failures"]

    if not failures:
        fa_summary = "No failures detected. All logs passed quality checks."
        processed_logs = []
    else:
        # In production: fa_summary = llm.invoke(f"Summarize these failures: {failures}")
        fa_summary = f"Found {len(failures)} failure(s). Common issues: Poor quality retrieval of documentation."
        processed_logs = [f"failure-analysis-on-log-{failure['id']}" for failure in failures]

    print(f"     Summary: {fa_summary}")
    print(f"     Processed: {len(processed_logs)} logs")

    return {
        "fa_summary": fa_summary,
        "processed_logs": processed_logs
    }


def build_failure_analysis_subgraph() -> StateGraph:
    """
    Build the Failure Analysis sub-graph.

    Flow: START → get_failures → generate_summary → END
    """
    print("\n🏗️ Building Failure Analysis Sub-Graph...")

    fa_builder = StateGraph(
        state_schema=FailureAnalysisState,
        output_schema=FailureAnalysisOutputState  # Controls what's returned
    )

    # Add nodes
    fa_builder.add_node("get_failures", get_failures)
    fa_builder.add_node("generate_summary", generate_fa_summary)

    # Define flow
    fa_builder.add_edge(START, "get_failures")
    fa_builder.add_edge("get_failures", "generate_summary")
    fa_builder.add_edge("generate_summary", END)

    print("   ✓ Failure Analysis sub-graph built")

    return fa_builder


# =============================================================================
# SUB-GRAPH 2: QUESTION SUMMARIZATION
# =============================================================================

class QuestionSummarizationState(TypedDict):
    """
    Internal state for Question Summarization sub-graph.

    This is the FULL state available inside the sub-graph.
    """
    cleaned_logs: List[Log]  # Input from parent
    qs_summary: str          # Internal state
    report: str              # Output to parent
    processed_logs: List[str]  # Output to parent


class QuestionSummarizationOutputState(TypedDict):
    """
    Output state for Question Summarization sub-graph.

    Only these keys will be returned to the parent graph.
    """
    report: str
    processed_logs: List[str]


def generate_qs_summary(state: QuestionSummarizationState) -> dict:
    """
    Node: Generate summary of questions.

    Analyzes common themes and topics in questions.
    """
    print("\n  🔍 [QS] Analyzing questions...")

    cleaned_logs = state["cleaned_logs"]

    # Extract questions
    questions = [log["question"] for log in cleaned_logs]

    # In production: qs_summary = llm.invoke(f"Summarize these questions: {questions}")
    qs_summary = f"Analyzed {len(questions)} questions. Common topics: Usage of ChatOllama and Chroma vector store."

    processed_logs = [f"summary-on-log-{log['id']}" for log in cleaned_logs]

    print(f"     Summary: {qs_summary}")
    print(f"     Processed: {len(processed_logs)} logs")

    return {
        "qs_summary": qs_summary,
        "processed_logs": processed_logs
    }


def send_to_slack(state: QuestionSummarizationState) -> dict:
    """
    Node: Generate report and send to Slack.

    In production, this would format and send via Slack API.
    """
    print("\n  📊 [QS] Generating report...")

    qs_summary = state["qs_summary"]

    # In production: report = format_for_slack(qs_summary)
    # In production: slack_client.send_message(report)
    report = f"📊 Question Analysis Report:\n{qs_summary}\n\nReport generated successfully."

    print(f"     Report: {report[:80]}...")
    print("     ✓ Report ready (would be sent to Slack in production)")

    return {"report": report}


def build_question_summarization_subgraph() -> StateGraph:
    """
    Build the Question Summarization sub-graph.

    Flow: START → generate_summary → send_to_slack → END
    """
    print("\n🏗️ Building Question Summarization Sub-Graph...")

    qs_builder = StateGraph(
        state_schema=QuestionSummarizationState,
        output_schema=QuestionSummarizationOutputState  # Controls what's returned
    )

    # Add nodes
    qs_builder.add_node("generate_summary", generate_qs_summary)
    qs_builder.add_node("send_to_slack", send_to_slack)

    # Define flow
    qs_builder.add_edge(START, "generate_summary")
    qs_builder.add_edge("generate_summary", "send_to_slack")
    qs_builder.add_edge("send_to_slack", END)

    print("   ✓ Question Summarization sub-graph built")

    return qs_builder


# =============================================================================
# PARENT GRAPH: LOG ANALYSIS SYSTEM
# =============================================================================

class EntryGraphState(TypedDict):
    """
    Parent graph state.

    Communication with sub-graphs happens through overlapping keys:
    - cleaned_logs: Input TO both sub-graphs
    - fa_summary: Output FROM failure analysis sub-graph
    - report: Output FROM question summarization sub-graph
    - processed_logs: Output FROM both sub-graphs (needs reducer!)
    """
    raw_logs: List[Log]
    cleaned_logs: List[Log]
    fa_summary: str
    report: str
    processed_logs: Annotated[List[str], add]  # Reducer for parallel outputs


def clean_logs(state: EntryGraphState) -> dict:
    """
    Node: Clean and prepare raw logs.

    This runs first and prepares data for sub-graphs.
    """
    print("\n🧹 Cleaning logs...")

    raw_logs = state["raw_logs"]

    # In production: cleaned_logs = data_cleaning_pipeline(raw_logs)
    cleaned_logs = raw_logs  # Simplified for demo

    print(f"   ✓ Cleaned {len(cleaned_logs)} logs")

    return {"cleaned_logs": cleaned_logs}


def build_entry_graph() -> StateGraph:
    """
    Build the parent graph with sub-graphs as nodes.

    Flow:
        START → clean_logs → [failure_analysis, question_summarization] → END

    The two sub-graphs run in PARALLEL!
    """
    print("\n" + "="*70)
    print("🏗️ Building Parent Graph with Sub-Graphs")
    print("="*70)

    # Build sub-graphs
    fa_graph = build_failure_analysis_subgraph().compile()
    qs_graph = build_question_summarization_subgraph().compile()

    # Build parent graph
    print("\n🏗️ Building Entry Graph...")

    entry_builder = StateGraph(EntryGraphState)

    # Add regular node
    entry_builder.add_node("clean_logs", clean_logs)

    # Add sub-graphs as nodes! ⭐
    entry_builder.add_node("failure_analysis", fa_graph)
    entry_builder.add_node("question_summarization", qs_graph)

    # Define flow
    entry_builder.add_edge(START, "clean_logs")

    # Fan-out to both sub-graphs (PARALLEL!)
    entry_builder.add_edge("clean_logs", "failure_analysis")
    entry_builder.add_edge("clean_logs", "question_summarization")

    # Fan-in from both sub-graphs
    entry_builder.add_edge("failure_analysis", END)
    entry_builder.add_edge("question_summarization", END)

    print("   ✓ Entry graph built with 2 sub-graphs")
    print("\n📊 Graph Structure:")
    print("   START → clean_logs → [failure_analysis, question_summarization] → END")
    print("                             ↓                    ↓")
    print("                        (sub-graph)          (sub-graph)")

    return entry_builder


# =============================================================================
# EXECUTION EXAMPLES
# =============================================================================

def run_basic_example():
    """
    Example: Run the complete log analysis system.

    Demonstrates:
    - Parent graph coordinating sub-graphs
    - Parallel execution of sub-graphs
    - State communication through overlapping keys
    """
    print("\n" + "="*70)
    print("EXAMPLE: Log Analysis with Sub-Graphs")
    print("="*70)

    # Build the complete system
    entry_graph = build_entry_graph()
    graph = entry_graph.compile()

    # Create sample logs
    print("\n📝 Creating sample logs...")

    question_answer = Log(
        id="1",
        question="How can I import ChatOllama?",
        answer="To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'",
    )

    question_answer_feedback = Log(
        id="2",
        question="How can I use Chroma vector store?",
        answer="To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).",
        grade=0,
        grader="Document Relevance Recall",
        feedback="The retrieved documents discuss vector stores in general, but not Chroma specifically",
    )

    question_answer_success = Log(
        id="3",
        question="What is LangChain?",
        answer="LangChain is a framework for building LLM applications.",
        grade=1,
        grader="Answer Quality",
        feedback="Accurate and concise answer.",
    )

    raw_logs = [question_answer, question_answer_feedback, question_answer_success]

    print(f"   ✓ Created {len(raw_logs)} sample logs")
    print("     - Log 1: Question without feedback")
    print("     - Log 2: Question with failure (grade=0)")
    print("     - Log 3: Question with success (grade=1)")

    # Run the graph
    print("\n" + "="*70)
    print("▶️ EXECUTING GRAPH")
    print("="*70)

    result = graph.invoke({"raw_logs": raw_logs})

    # Display results
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)

    print(f"\n✅ Raw Logs: {len(result['raw_logs'])} logs")
    print(f"✅ Cleaned Logs: {len(result['cleaned_logs'])} logs")

    print(f"\n📊 Failure Analysis Summary:")
    print(f"   {result['fa_summary']}")

    print(f"\n📊 Question Summarization Report:")
    print(f"   {result['report']}")

    print(f"\n📋 Processed Logs: {len(result['processed_logs'])} entries")
    for log in result['processed_logs']:
        print(f"   - {log}")

    print("\n" + "="*70)
    print("✅ Execution Complete!")
    print("="*70)
    print("\n💡 Notice:")
    print("   - Both sub-graphs ran in parallel")
    print("   - Each had its own internal state")
    print("   - Results combined in parent state")
    print("   - processed_logs merged with operator.add reducer")


# =============================================================================
# KEY LEARNINGS
# =============================================================================

def print_key_concepts():
    """Print key concepts for sub-graphs."""
    print("\n" + "="*70)
    print("📚 KEY CONCEPTS: Sub-Graphs")
    print("="*70)

    concepts = """
    1. SUB-GRAPHS ARE MODULAR COMPONENTS:
       - Self-contained graphs with own state
       - Reusable across different parent graphs
       - Enable team-based agent patterns
       - Promote clean separation of concerns

    2. STATE COMMUNICATION:
       - Through OVERLAPPING KEYS between parent and sub-graph
       - Parent provides input via shared keys
       - Sub-graphs return output via shared keys
       - Like function parameters and return values

    3. STATE SCHEMAS:
       - Input schema: Keys sub-graph needs from parent
       - Internal schema: Full state inside sub-graph
       - Output schema: Keys sub-graph returns to parent
       - Output schema prevents data leakage

    4. ADDING SUB-GRAPHS TO PARENT:
       sub_graph = builder.compile()
       parent.add_node("sub_graph_name", sub_graph)
       - Treat compiled sub-graph as a node!
       - Can have multiple sub-graphs
       - Can run in parallel or sequence

    5. PARALLEL SUB-GRAPH EXECUTION:
       - Same as parallel nodes
       - Need reducers for shared output keys
       - Example: processed_logs with operator.add
       - Each sub-graph adds to same list

    6. OUTPUT STATE SCHEMA:
       StateGraph(
           state_schema=FullState,
           output_schema=OutputState  # Only these keys returned
       )
       - Prevents returning unnecessary data
       - Cleaner parent state
       - Avoids reducer conflicts

    7. REAL-WORLD PATTERNS:
       - Multi-agent teams (each agent = sub-graph)
       - Microservices architecture
       - Pipeline stages with different concerns
       - Parallel specialized processors

    8. BENEFITS:
       ✓ Modularity: Reusable components
       ✓ Separation: Each sub-graph focused on one task
       ✓ Testability: Test sub-graphs independently
       ✓ Scalability: Add/remove sub-graphs easily
       ✓ Maintainability: Clear boundaries

    9. WHEN TO USE SUB-GRAPHS:
       ✓ Multiple specialized agents/teams
       ✓ Different state requirements per task
       ✓ Reusable components
       ✓ Complex multi-stage pipelines
       ✓ Separation of concerns
       ✗ Simple linear workflows (overkill)
       ✗ Tightly coupled operations

    10. COMMUNICATION DIAGRAM:

        Parent State: {cleaned_logs, fa_summary, report, processed_logs}
                                    ↓
                    ┌───────────────────────────────┐
                    │                               │
              [FA Sub-Graph]              [QS Sub-Graph]
              State: {cleaned_logs,       State: {cleaned_logs,
                      failures,                   qs_summary,
                      fa_summary,                 report,
                      processed_logs}             processed_logs}
                    │                               │
                    └───────────────────────────────┘
                                    ↓
        Parent gets: {fa_summary, report, processed_logs}
    """
    print(concepts)


def print_comparison_table():
    """Print comparison of different graph patterns."""
    print("\n" + "="*70)
    print("📊 COMPARISON: Graph Patterns")
    print("="*70)

    comparison = """
    SINGLE GRAPH (Modules 1-3):
    ────────────────────────────────────────────
    Structure:  All nodes in one graph
    State:      Single shared state
    Modularity: Limited
    Reusability: Low
    Use case:   Simple workflows

    PARALLEL NODES (Lesson 1):
    ────────────────────────────────────────────
    Structure:  Nodes run in parallel
    State:      Single shared state
    Modularity: Medium
    Reusability: Medium
    Use case:   Independent operations

    SUB-GRAPHS (This Lesson):
    ────────────────────────────────────────────
    Structure:  Graphs within graphs
    State:      Multiple state schemas
    Modularity: High
    Reusability: High
    Use case:   Multi-agent teams, complex systems

    EXAMPLE: Log Analysis
    ────────────────────────────────────────────
    Without sub-graphs:
    - 20 nodes in single graph
    - Complex state management
    - Hard to test/reuse

    With sub-graphs:
    - 2 sub-graphs (4 nodes each)
    - Clean state boundaries
    - Each sub-graph testable
    - Reusable in other systems
    """
    print(comparison)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "🚀"*35)
    print("MODULE 4 - LESSON 2: SUB-GRAPHS")
    print("🚀"*35)
    print("\n💡 Concept: Build modular, reusable graph components")
    print("   Think: Microservices for agents!")

    # Run main example
    run_basic_example()

    # Show key concepts
    print_key_concepts()
    print_comparison_table()

    print("\n" + "="*70)
    print("✅ Module 4, Lesson 2 Complete!")
    print("="*70)
    print("\n🎓 Achievement Unlocked: Sub-Graph Architect!")
    print("\n📖 Next: Continue with remaining Module 4 lessons")
    print("   - Map-Reduce patterns")
    print("   - Complete Research Assistant")
    print()
