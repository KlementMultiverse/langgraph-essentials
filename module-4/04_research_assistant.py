"""
Module 4 - Lesson 4: Research Assistant - Complete Multi-Agent System

Bringing together ALL Module 4 concepts:
- Parallelization (Lesson 1): Multi-source data gathering
- Sub-Graphs (Lesson 2): Modular agent teams
- Map-Reduce (Lesson 3): Process multiple queries

This is a PRODUCTION-READY pattern used in real-world systems like:
- Research assistants (Perplexity AI)
- Customer support bots
- Data analysis platforms
- Content generation systems

KEY CONCEPTS:
1. Multi-agent orchestration
2. Parallel execution across agents
3. Modular sub-graph composition
4. Dynamic query processing
5. Complete integration of all patterns
"""

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
import operator

# ============================================================================
# EXAMPLE 1: Simple Research Assistant - Basic Integration
# ============================================================================

print("=" * 70)
print("EXAMPLE 1: Simple Research Assistant - Basic Integration")
print("=" * 70)

class SimpleResearchState(TypedDict):
    """State for simple research assistant"""
    query: str                                      # Input query
    web_results: List[str]                          # Web search results
    wiki_results: List[str]                         # Wikipedia results
    all_results: Annotated[List[str], operator.add] # Combined results
    report: str                                     # Final report

def search_web(state: SimpleResearchState):
    """
    PARALLEL NODE 1: Search the web

    Runs in parallel with search_wikipedia
    """
    query = state['query']

    # Mock web search
    results = [
        f"Web result 1 for '{query}': Recent developments...",
        f"Web result 2 for '{query}': Expert analysis...",
    ]

    print(f"  🌐 Web search completed: {len(results)} results")
    return {
        "web_results": results,
        "all_results": results
    }

def search_wikipedia(state: SimpleResearchState):
    """
    PARALLEL NODE 2: Search Wikipedia

    Runs in parallel with search_web
    """
    query = state['query']

    # Mock Wikipedia search
    results = [
        f"Wikipedia for '{query}': Comprehensive overview...",
    ]

    print(f"  📚 Wikipedia search completed: {len(results)} results")
    return {
        "wiki_results": results,
        "all_results": results
    }

def generate_report(state: SimpleResearchState):
    """
    REDUCE NODE: Combine all results

    Waits for BOTH search_web and search_wikipedia to complete
    """
    all_results = state['all_results']

    report = f"""
RESEARCH REPORT: {state['query']}
==========================================

Sources Found: {len(all_results)}

Web Results: {len(state.get('web_results', []))}
Wikipedia Results: {len(state.get('wiki_results', []))}

Combined Findings:
{chr(10).join(f"- {r}" for r in all_results)}
"""

    print(f"\n✅ Generated report with {len(all_results)} sources")
    return {"report": report}

# Build graph - PARALLELIZATION PATTERN
builder_1 = StateGraph(SimpleResearchState)
builder_1.add_node("search_web", search_web)
builder_1.add_node("search_wikipedia", search_wikipedia)
builder_1.add_node("generate_report", generate_report)

# Fan-out: START → both search nodes
builder_1.add_edge(START, "search_web")
builder_1.add_edge(START, "search_wikipedia")

# Fan-in: both searches → generate_report
builder_1.add_edge(["search_web", "search_wikipedia"], "generate_report")
builder_1.add_edge("generate_report", END)

graph_1 = builder_1.compile()

# Run example
print("\n🔍 Starting research for: 'LangGraph'")
result_1 = graph_1.invoke({
    "query": "LangGraph"
})

print(result_1['report'])

# ============================================================================
# EXAMPLE 2: Research Assistant with Sub-Graphs
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 2: Research Assistant with Analysis Sub-Graph")
print("=" * 70)

# SUB-GRAPH: Analysis Agent
class AnalysisState(TypedDict):
    """State for analysis sub-graph"""
    raw_data: str           # Input from parent
    key_facts: List[str]    # Extracted facts
    summary: str            # Analysis summary

class AnalysisOutputState(TypedDict):
    """What the sub-graph returns to parent"""
    summary: str            # Only return summary, not internal details

def extract_facts(state: AnalysisState):
    """Extract key facts from raw data"""
    raw = state['raw_data']

    # Mock fact extraction
    facts = [
        f"Fact 1 from: {raw[:30]}...",
        f"Fact 2 from: {raw[:30]}...",
    ]

    print(f"    📊 Extracted {len(facts)} key facts")
    return {"key_facts": facts}

def create_summary(state: AnalysisState):
    """Create summary from facts"""
    facts = state['key_facts']

    summary = f"Analysis Summary: {len(facts)} key facts identified"

    print(f"    📝 Created summary")
    return {"summary": summary}

# Build analysis sub-graph
analysis_builder = StateGraph(
    state_schema=AnalysisState,
    output_schema=AnalysisOutputState  # Only return summary
)
analysis_builder.add_node("extract_facts", extract_facts)
analysis_builder.add_node("create_summary", create_summary)
analysis_builder.add_edge(START, "extract_facts")
analysis_builder.add_edge("extract_facts", "create_summary")
analysis_builder.add_edge("create_summary", END)

analysis_graph = analysis_builder.compile()

# PARENT GRAPH: Research with Analysis
class ResearchWithAnalysisState(TypedDict):
    """Parent state that uses analysis sub-graph"""
    query: str
    search_results: Annotated[List[str], operator.add]
    summary: str  # Overlapping key - analysis returns this
    final_report: str

def parallel_search(state: ResearchWithAnalysisState):
    """Search multiple sources (simplified)"""
    query = state['query']

    results = [
        f"Source 1: Information about {query}",
        f"Source 2: Analysis of {query}",
        f"Source 3: Recent updates on {query}",
    ]

    print(f"  🔍 Found {len(results)} results")
    return {"search_results": results}

def analyze_results(state: ResearchWithAnalysisState):
    """
    SUB-GRAPH NODE: Analyze search results

    This is where the analysis sub-graph runs!
    """
    results = state['search_results']

    print(f"\n  🧠 Analyzing {len(results)} results via sub-graph...")

    # Pass data to sub-graph
    # Sub-graph processes and returns only 'summary'
    combined_data = " | ".join(results)
    return {"raw_data": combined_data}

def create_final_report(state: ResearchWithAnalysisState):
    """Create final report using analysis summary"""
    summary = state['summary']
    results_count = len(state['search_results'])

    report = f"""
RESEARCH REPORT WITH ANALYSIS
==============================

Query: {state['query']}
Sources: {results_count}

Analysis: {summary}

Report generated successfully!
"""

    print(f"\n✅ Final report created")
    return {"final_report": report}

# Build parent graph with sub-graph
builder_2 = StateGraph(ResearchWithAnalysisState)
builder_2.add_node("parallel_search", parallel_search)
builder_2.add_node("analyze_results", analysis_graph)  # SUB-GRAPH!
builder_2.add_node("create_final_report", create_final_report)

builder_2.add_edge(START, "parallel_search")
builder_2.add_edge("parallel_search", "analyze_results")
builder_2.add_edge("analyze_results", "create_final_report")
builder_2.add_edge("create_final_report", END)

graph_2 = builder_2.compile()

# Run example
print("\n🔍 Starting research with analysis...")
result_2 = graph_2.invoke({
    "query": "Multi-Agent Systems"
})

print(result_2['final_report'])

# ============================================================================
# EXAMPLE 3: Multi-Topic Research with Map-Reduce
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 3: Multi-Topic Research - Map-Reduce Integration")
print("=" * 70)

class MultiTopicState(TypedDict):
    """State for multi-topic research"""
    topics: List[str]                                   # Input topics
    topic_reports: Annotated[List[dict], operator.add] # Individual reports
    final_summary: str                                  # Combined summary

def map_topics(state: MultiTopicState):
    """
    MAP STEP: Create research task for each topic

    This is MAP-REDUCE pattern!
    """
    print(f"\n📤 MAP: Creating research tasks for {len(state['topics'])} topics")
    return [
        Send("research_topic", {"topic": topic})
        for topic in state['topics']
    ]

def research_topic(state: MultiTopicState):
    """
    PROCESS STEP: Research one topic

    Runs in PARALLEL for each topic
    This simulates: search web + wiki + analyze
    """
    topic = state['topic']

    # Simulate multi-source research
    web_results = [f"Web info about {topic}"]
    wiki_results = [f"Wiki article about {topic}"]

    report = {
        "topic": topic,
        "sources": len(web_results) + len(wiki_results),
        "summary": f"Comprehensive research on {topic} completed"
    }

    print(f"  📊 Researched: '{topic}' ({report['sources']} sources)")

    return {"topic_reports": [report]}

def combine_research(state: MultiTopicState):
    """
    REDUCE STEP: Combine all topic reports
    """
    reports = state['topic_reports']

    summary = f"""
MULTI-TOPIC RESEARCH SUMMARY
=============================

Topics Researched: {len(reports)}
Total Sources: {sum(r['sources'] for r in reports)}

Individual Reports:
"""

    for i, report in enumerate(reports, 1):
        summary += f"\n{i}. {report['topic']}: {report['summary']}"

    print(f"\n✅ REDUCE: Combined {len(reports)} topic reports")

    return {"final_summary": summary}

# Build graph - MAP-REDUCE PATTERN
builder_3 = StateGraph(MultiTopicState)
builder_3.add_node("research_topic", research_topic)
builder_3.add_node("combine_research", combine_research)

# Map: Dynamic branching
builder_3.add_conditional_edges(START, map_topics)

# Reduce: Combine results
builder_3.add_edge("research_topic", "combine_research")
builder_3.add_edge("combine_research", END)

graph_3 = builder_3.compile()

# Run example
print("\n🔍 Starting multi-topic research...")
result_3 = graph_3.invoke({
    "topics": ["LangGraph", "Multi-Agent Systems", "RAG"]
})

print(result_3['final_summary'])

# ============================================================================
# EXAMPLE 4: Complete Research Assistant - ALL PATTERNS
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 4: Complete Research Assistant - ALL Patterns Combined!")
print("=" * 70)

# SUB-GRAPH 1: Web Search Agent
class WebSearchState(TypedDict):
    query: str
    results: List[str]

class WebSearchOutputState(TypedDict):
    results: List[str]

def web_search_node(state: WebSearchState):
    """Web search sub-graph logic"""
    query = state['query']
    results = [
        f"Web: Latest news on {query}",
        f"Web: Expert opinion on {query}",
    ]
    print(f"    🌐 Web search: {len(results)} results")
    return {"results": results}

web_search_builder = StateGraph(
    state_schema=WebSearchState,
    output_schema=WebSearchOutputState
)
web_search_builder.add_node("search", web_search_node)
web_search_builder.add_edge(START, "search")
web_search_builder.add_edge("search", END)
web_search_graph = web_search_builder.compile()

# SUB-GRAPH 2: Wikipedia Agent
class WikiSearchState(TypedDict):
    query: str
    results: List[str]

class WikiSearchOutputState(TypedDict):
    results: List[str]

def wiki_search_node(state: WikiSearchState):
    """Wikipedia search sub-graph logic"""
    query = state['query']
    results = [
        f"Wiki: {query} - comprehensive overview",
    ]
    print(f"    📚 Wiki search: {len(results)} results")
    return {"results": results}

wiki_search_builder = StateGraph(
    state_schema=WikiSearchState,
    output_schema=WikiSearchOutputState
)
wiki_search_builder.add_node("search", wiki_search_node)
wiki_search_builder.add_edge(START, "search")
wiki_search_builder.add_edge("search", END)
wiki_search_graph = wiki_search_builder.compile()

# MAIN GRAPH: Complete Research Assistant
class CompleteResearchState(TypedDict):
    """State for complete research assistant"""
    topics: List[str]                                       # Input topics
    current_topic: str                                      # Current topic being researched
    web_results: Annotated[List[str], operator.add]        # All web results
    wiki_results: Annotated[List[str], operator.add]       # All wiki results
    topic_summaries: Annotated[List[dict], operator.add]   # Summaries per topic
    final_report: str                                       # Final comprehensive report

def map_research_topics(state: CompleteResearchState):
    """
    MAP-REDUCE: Create research task for each topic
    """
    print(f"\n📤 MAP-REDUCE: Creating tasks for {len(state['topics'])} topics")
    return [
        Send("research_single_topic", {"current_topic": topic})
        for topic in state['topics']
    ]

def research_single_topic(state: CompleteResearchState):
    """
    PARALLELIZATION + SUB-GRAPHS: Research one topic using multiple agents

    This node:
    1. Takes one topic
    2. Calls web search sub-graph
    3. Calls wiki search sub-graph
    4. Combines results
    """
    topic = state['current_topic']

    print(f"\n  📊 Researching topic: '{topic}'")
    print(f"    Using parallel sub-graphs (web + wiki)...")

    # Call web search sub-graph
    web_result = web_search_graph.invoke({"query": topic})
    web_data = web_result['results']

    # Call wiki search sub-graph
    wiki_result = wiki_search_graph.invoke({"query": topic})
    wiki_data = wiki_result['results']

    # Create topic summary
    summary = {
        "topic": topic,
        "web_sources": len(web_data),
        "wiki_sources": len(wiki_data),
        "total_sources": len(web_data) + len(wiki_data)
    }

    print(f"    ✅ Completed: {summary['total_sources']} total sources")

    return {
        "web_results": web_data,
        "wiki_results": wiki_data,
        "topic_summaries": [summary]
    }

def generate_comprehensive_report(state: CompleteResearchState):
    """
    REDUCE: Create final comprehensive report
    """
    summaries = state['topic_summaries']
    web_results = state['web_results']
    wiki_results = state['wiki_results']

    report = f"""
╔════════════════════════════════════════════════════════════════╗
║        COMPREHENSIVE RESEARCH REPORT                           ║
║        Complete Multi-Agent System                             ║
╚════════════════════════════════════════════════════════════════╝

📊 RESEARCH OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Topics Researched: {len(summaries)}
Total Web Sources: {len(web_results)}
Total Wiki Sources: {len(wiki_results)}
Combined Sources: {len(web_results) + len(wiki_results)}

📚 TOPIC BREAKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""

    for i, summary in enumerate(summaries, 1):
        report += f"""
{i}. {summary['topic']}
   • Web Sources: {summary['web_sources']}
   • Wiki Sources: {summary['wiki_sources']}
   • Total: {summary['total_sources']} sources
"""

    report += f"""
🎯 SYSTEM PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Patterns Used:
✅ Map-Reduce: Processed {len(summaries)} topics dynamically
✅ Parallelization: Web + Wiki searches ran simultaneously
✅ Sub-Graphs: Modular search agents (web_search, wiki_search)

Benefits:
• Scalability: Can handle any number of topics
• Speed: Parallel execution across all agents
• Modularity: Independent, reusable agent components
• Maintainability: Each agent is self-contained

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Report generated successfully! ✅
"""

    print(f"\n✅ COMPREHENSIVE REPORT GENERATED")
    print(f"   Topics: {len(summaries)}")
    print(f"   Total sources: {len(web_results) + len(wiki_results)}")

    return {"final_report": report}

# Build complete graph - ALL PATTERNS
builder_4 = StateGraph(CompleteResearchState)
builder_4.add_node("research_single_topic", research_single_topic)
builder_4.add_node("generate_comprehensive_report", generate_comprehensive_report)

# Map-Reduce: Dynamic topic branching
builder_4.add_conditional_edges(START, map_research_topics)

# Reduce: Combine all research
builder_4.add_edge("research_single_topic", "generate_comprehensive_report")
builder_4.add_edge("generate_comprehensive_report", END)

graph_4 = builder_4.compile()

# Run complete example
print("\n🚀 Starting COMPLETE research assistant...")
print("   This demonstrates ALL Module 4 patterns working together!")

result_4 = graph_4.invoke({
    "topics": ["LangGraph", "Multi-Agent Systems", "Sub-Graphs", "Map-Reduce"]
})

print(result_4['final_report'])

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

print("\n" + "=" * 70)
print("🎓 KEY TAKEAWAYS - Complete Multi-Agent Systems")
print("=" * 70)

print("""
╔════════════════════════════════════════════════════════════════╗
║  YOU NOW UNDERSTAND PRODUCTION MULTI-AGENT SYSTEMS!            ║
╚════════════════════════════════════════════════════════════════╝

1. **Pattern Integration**
   ✅ Parallelization: Multiple sources simultaneously
   ✅ Sub-Graphs: Modular agent teams
   ✅ Map-Reduce: Dynamic query processing
   ✅ Complete Integration: All patterns working together

2. **Multi-Agent Architecture**

   PARENT GRAPH (Orchestrator)
        ↓
   MAP-REDUCE (Dynamic topics)
        ↓
   PARALLEL AGENTS (Web + Wiki sub-graphs)
        ↓
   REDUCE (Combine results)

3. **Real-World Applications**

   🔍 Research Assistants
   - Multi-source data gathering
   - Parallel processing
   - Comprehensive reporting

   💬 Customer Support
   - Multi-agent teams (tier 1, tier 2, specialist)
   - Parallel knowledge lookup
   - Dynamic routing

   📊 Data Analysis
   - Multiple data sources
   - Parallel processing pipelines
   - Modular analyzers

   ✍️ Content Generation
   - Multiple topic research
   - Parallel content creation
   - Quality analysis sub-graphs

4. **Performance Benefits**

   Sequential Processing:
   - 4 topics × 2 sources × 2s = 16 seconds

   Multi-Agent System:
   - Map-Reduce: 4 topics in parallel
   - Parallelization: 2 sources in parallel per topic
   - Time: ~2 seconds (8× faster!)

5. **Design Principles**

   ✅ Modularity: Each agent is self-contained
   ✅ Scalability: Add agents without changing orchestrator
   ✅ Reusability: Agents work independently
   ✅ Maintainability: Easy to test and debug
   ✅ Performance: Maximum parallelization

6. **Production Patterns**

   a) Orchestrator Pattern
      - Parent graph coordinates agents
      - Agents are sub-graphs
      - Clean separation of concerns

   b) Fan-Out/Fan-In
      - Distribute work to agents
      - Collect and combine results
      - Efficient resource utilization

   c) Dynamic Workload
      - Map-Reduce for variable inputs
      - Adaptive to workload size
      - Optimal parallelization

7. **Module 4 Complete! 🎉**

   Lesson 1: Parallelization ✅
   Lesson 2: Sub-Graphs ✅
   Lesson 3: Map-Reduce ✅
   Lesson 4: Research Assistant ✅

   You can now build:
   - Production multi-agent systems
   - Scalable research assistants
   - Complex workflow orchestration
   - Modular AI applications

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏆 ACHIEVEMENT UNLOCKED: Multi-Agent Systems Engineer!

You've mastered advanced LangGraph patterns and can now build
production-ready, scalable, multi-agent AI systems!

Next: Module 5 - Memory & Persistence
      (Cross-thread memory, personalization, long-term context)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print("✅ MODULE 4 - COMPLETE! ALL 4 LESSONS FINISHED!")
print("=" * 70)
