"""
Module 4 - Lesson 3: Map-Reduce Patterns

Learn to process multiple items in parallel using dynamic branching with the Send API.

KEY CONCEPTS:
1. Send API for dynamic node creation
2. Map: Apply operation to each item (parallel)
3. Reduce: Combine all results
4. Batch processing patterns
5. Dynamic branching vs fixed parallelization

REAL-WORLD USE CASES:
- Batch document processing
- Multi-topic research
- Content generation at scale
- Data pipeline transformations
- Competitive analysis
"""

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
import operator

# ============================================================================
# EXAMPLE 1: Basic Map-Reduce - Number Processing
# ============================================================================

print("=" * 70)
print("EXAMPLE 1: Basic Map-Reduce - Processing Numbers")
print("=" * 70)

class BasicState(TypedDict):
    """State for basic map-reduce example"""
    numbers: List[int]                              # Input: List of numbers
    squared: Annotated[List[int], operator.add]     # Map output (REDUCER!)
    sum_of_squares: int                             # Reduce output

def map_numbers(state: BasicState):
    """
    MAP STEP: Create Send() for each number

    This is where dynamic branching happens!
    - For each number, create a Send() command
    - Each Send() will invoke "square_number" with that number
    - All invocations run in PARALLEL
    """
    print(f"\n📤 MAP: Creating tasks for {len(state['numbers'])} numbers")
    return [
        Send("square_number", {"number": num})
        for num in state['numbers']
    ]

def square_number(state: BasicState):
    """
    PROCESS STEP: Square one number

    This function runs ONCE for EACH number in parallel!
    - Receives one number via state
    - Processes it
    - Returns result to 'squared' list (combined by reducer)
    """
    num = state['number']
    result = num ** 2
    print(f"  🔢 Processing: {num}^2 = {result}")
    return {"squared": [result]}

def sum_squares(state: BasicState):
    """
    REDUCE STEP: Combine all squared numbers

    This runs ONCE after all map operations complete
    - Receives all squared numbers
    - Combines them into final result
    """
    total = sum(state['squared'])
    print(f"\n✅ REDUCE: Sum of squares = {total}")
    return {"sum_of_squares": total}

# Build graph
builder_1 = StateGraph(BasicState)
builder_1.add_node("square_number", square_number)
builder_1.add_node("sum_squares", sum_squares)

# Dynamic branching from START
builder_1.add_conditional_edges(START, map_numbers)

# All square_number nodes feed into sum_squares
builder_1.add_edge("square_number", "sum_squares")
builder_1.add_edge("sum_squares", END)

graph_1 = builder_1.compile()

# Run example
result_1 = graph_1.invoke({
    "numbers": [1, 2, 3, 4, 5]
})

print(f"\n🎯 FINAL RESULT:")
print(f"   Input: {[1, 2, 3, 4, 5]}")
print(f"   Squared: {result_1['squared']}")
print(f"   Sum: {result_1['sum_of_squares']}")

# ============================================================================
# EXAMPLE 2: Map-Reduce with Conditional Logic
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 2: Map-Reduce with Filtering")
print("=" * 70)

class FilterState(TypedDict):
    """State with conditional processing"""
    items: List[str]
    processed: Annotated[List[str], operator.add]
    valid_count: int

def map_items(state: FilterState):
    """MAP: Create task for each item"""
    print(f"\n📤 MAP: Creating tasks for {len(state['items'])} items")
    return [
        Send("process_item", {"item": item})
        for item in state['items']
    ]

def process_item(state: FilterState):
    """
    PROCESS: Validate and transform item

    Conditional logic in map phase:
    - Only return valid items
    - Invalid items filtered out
    """
    item = state['item']

    # Conditional logic: Only process items longer than 3 chars
    if len(item) > 3:
        result = item.upper()
        print(f"  ✅ Valid: '{item}' → '{result}'")
        return {"processed": [result]}
    else:
        print(f"  ❌ Invalid: '{item}' (too short)")
        return {"processed": []}  # Empty list = filtered out

def count_valid(state: FilterState):
    """REDUCE: Count valid items"""
    count = len(state['processed'])
    print(f"\n✅ REDUCE: {count} valid items processed")
    return {"valid_count": count}

# Build graph
builder_2 = StateGraph(FilterState)
builder_2.add_node("process_item", process_item)
builder_2.add_node("count_valid", count_valid)
builder_2.add_conditional_edges(START, map_items)
builder_2.add_edge("process_item", "count_valid")
builder_2.add_edge("count_valid", END)

graph_2 = builder_2.compile()

# Run example
result_2 = graph_2.invoke({
    "items": ["cat", "elephant", "ox", "tiger", "ant", "bear"]
})

print(f"\n🎯 FINAL RESULT:")
print(f"   Input: {['cat', 'elephant', 'ox', 'tiger', 'ant', 'bear']}")
print(f"   Valid (>3 chars): {result_2['processed']}")
print(f"   Count: {result_2['valid_count']}")

# ============================================================================
# EXAMPLE 3: Joke Generator - Multiple Topics (LLM Use Case)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 3: Joke Generator - Map-Reduce with Mock LLM")
print("=" * 70)

class JokeState(TypedDict):
    """State for joke generation"""
    topics: List[str]                               # Input topics
    jokes: Annotated[List[dict], operator.add]      # Generated jokes
    best_joke: dict                                 # Final selection

def map_topics(state: JokeState):
    """MAP: Create joke generation task for each topic"""
    print(f"\n📤 MAP: Generating jokes for {len(state['topics'])} topics")
    return [
        Send("generate_joke", {"topic": topic})
        for topic in state['topics']
    ]

def generate_joke(state: JokeState):
    """
    PROCESS: Generate joke for one topic

    In real implementation, this would call an LLM
    """
    topic = state['topic']

    # Mock LLM response (in real app, use OpenAI/Anthropic)
    jokes_db = {
        "cats": "Why did the cat sit on the computer? To keep an eye on the mouse!",
        "programming": "Why do programmers prefer dark mode? Because light attracts bugs!",
        "pizza": "Why did the pizza go to therapy? It had too many toppings to deal with!",
        "coffee": "How does a programmer drink coffee? In Java!",
        "dogs": "Why did the dog go to school? To get a little ruff education!"
    }

    joke_text = jokes_db.get(topic, f"I don't have a joke about {topic}!")
    joke_data = {
        "topic": topic,
        "joke": joke_text,
        "rating": len(joke_text) % 10  # Mock rating
    }

    print(f"  😄 Generated joke about '{topic}' (rating: {joke_data['rating']})")
    return {"jokes": [joke_data]}

def select_best_joke(state: JokeState):
    """
    REDUCE: Select the best joke

    In real implementation, this would use LLM to judge quality
    """
    jokes = state['jokes']

    # Mock selection: pick highest rated
    best = max(jokes, key=lambda j: j['rating'])

    print(f"\n✅ REDUCE: Selected best joke from {len(jokes)} options")
    print(f"   Winner: '{best['topic']}' (rating: {best['rating']})")

    return {"best_joke": best}

# Build graph
builder_3 = StateGraph(JokeState)
builder_3.add_node("generate_joke", generate_joke)
builder_3.add_node("select_best", select_best_joke)
builder_3.add_conditional_edges(START, map_topics)
builder_3.add_edge("generate_joke", "select_best")
builder_3.add_edge("select_best", END)

graph_3 = builder_3.compile()

# Run example
result_3 = graph_3.invoke({
    "topics": ["cats", "programming", "pizza", "coffee"]
})

print(f"\n🎯 FINAL RESULT:")
print(f"   Topics: {result_3['topics']}")
print(f"   Best Joke: {result_3['best_joke']['joke']}")

# ============================================================================
# EXAMPLE 4: Document Research - Multi-Stage Map-Reduce
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 4: Document Research - Nested Map-Reduce")
print("=" * 70)

class ResearchState(TypedDict):
    """State for document research"""
    queries: List[str]                                  # Input queries
    documents: Annotated[List[dict], operator.add]      # Found documents
    summaries: Annotated[List[str], operator.add]       # Document summaries
    final_report: str                                   # Final combined report

def map_queries(state: ResearchState):
    """MAP STAGE 1: Search for each query"""
    print(f"\n📤 MAP STAGE 1: Searching {len(state['queries'])} queries")
    return [
        Send("search_query", {"query": query})
        for query in state['queries']
    ]

def search_query(state: ResearchState):
    """PROCESS STAGE 1: Search one query (mock)"""
    query = state['query']

    # Mock search results
    doc = {
        "query": query,
        "title": f"Research paper about {query}",
        "content": f"This is detailed information about {query}. " * 3
    }

    print(f"  🔍 Searched: '{query}' → Found 1 document")
    return {"documents": [doc]}

def map_documents(state: ResearchState):
    """MAP STAGE 2: Summarize each document"""
    print(f"\n📤 MAP STAGE 2: Summarizing {len(state['documents'])} documents")
    return [
        Send("summarize_doc", {"document": doc})
        for doc in state['documents']
    ]

def summarize_doc(state: ResearchState):
    """PROCESS STAGE 2: Summarize one document (mock)"""
    doc = state['document']

    # Mock summarization
    summary = f"Summary of '{doc['title']}': Key findings about {doc['query']}."

    print(f"  📝 Summarized: '{doc['title']}'")
    return {"summaries": [summary]}

def create_report(state: ResearchState):
    """REDUCE: Combine all summaries into final report"""
    summaries = state['summaries']

    report = f"Research Report ({len(summaries)} sources)\n\n"
    for i, summary in enumerate(summaries, 1):
        report += f"{i}. {summary}\n"

    print(f"\n✅ REDUCE: Created final report from {len(summaries)} summaries")

    return {"final_report": report}

# Build graph with TWO map-reduce stages
builder_4 = StateGraph(ResearchState)
builder_4.add_node("search_query", search_query)
builder_4.add_node("summarize_doc", summarize_doc)
builder_4.add_node("create_report", create_report)

# Stage 1: Map queries → Search
builder_4.add_conditional_edges(START, map_queries)

# Stage 2: Map documents → Summarize
builder_4.add_conditional_edges("search_query", map_documents)

# Final reduce
builder_4.add_edge("summarize_doc", "create_report")
builder_4.add_edge("create_report", END)

graph_4 = builder_4.compile()

# Run example
result_4 = graph_4.invoke({
    "queries": ["LangGraph", "Map-Reduce", "Sub-Graphs"]
})

print(f"\n🎯 FINAL RESULT:")
print(f"   Queries: {result_4['queries']}")
print(f"   Documents found: {len(result_4['documents'])}")
print(f"   Report:\n{result_4['final_report']}")

# ============================================================================
# EXAMPLE 5: Competitive Analysis - Real-World Pattern
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 5: Competitive Analysis - Complete Pattern")
print("=" * 70)

class CompetitorState(TypedDict):
    """State for competitive analysis"""
    companies: List[str]                                # Companies to analyze
    analyses: Annotated[List[dict], operator.add]       # Individual analyses
    strengths: Annotated[List[str], operator.add]       # All strengths found
    weaknesses: Annotated[List[str], operator.add]      # All weaknesses found
    recommendation: str                                 # Final recommendation

def map_companies(state: CompetitorState):
    """MAP: Analyze each company"""
    print(f"\n📤 MAP: Analyzing {len(state['companies'])} companies")
    return [
        Send("analyze_company", {"company": company})
        for company in state['companies']
    ]

def analyze_company(state: CompetitorState):
    """PROCESS: Analyze one company (mock)"""
    company = state['company']

    # Mock analysis
    analysis = {
        "company": company,
        "market_share": len(company) * 2,  # Mock metric
        "growth_rate": len(company) * 1.5,
        "score": len(company) * 3
    }

    strengths = [f"{company} has strong brand recognition"]
    weaknesses = [f"{company} lacks presence in emerging markets"]

    print(f"  🏢 Analyzed: {company} (score: {analysis['score']})")

    return {
        "analyses": [analysis],
        "strengths": strengths,
        "weaknesses": weaknesses
    }

def create_recommendation(state: CompetitorState):
    """REDUCE: Create strategic recommendation"""
    analyses = state['analyses']
    strengths = state['strengths']
    weaknesses = state['weaknesses']

    # Find best competitor
    best = max(analyses, key=lambda a: a['score'])

    recommendation = f"""
COMPETITIVE ANALYSIS REPORT
===========================

Companies Analyzed: {len(analyses)}

Top Competitor: {best['company']}
- Market Share: {best['market_share']}%
- Growth Rate: {best['growth_rate']}%
- Overall Score: {best['score']}

Total Strengths Identified: {len(strengths)}
Total Weaknesses Identified: {len(weaknesses)}

RECOMMENDATION: Focus on differentiating from {best['company']}.
"""

    print(f"\n✅ REDUCE: Generated strategic recommendation")
    print(f"   Top competitor: {best['company']}")

    return {"recommendation": recommendation}

# Build graph
builder_5 = StateGraph(CompetitorState)
builder_5.add_node("analyze_company", analyze_company)
builder_5.add_node("create_recommendation", create_recommendation)
builder_5.add_conditional_edges(START, map_companies)
builder_5.add_edge("analyze_company", "create_recommendation")
builder_5.add_edge("create_recommendation", END)

graph_5 = builder_5.compile()

# Run example
result_5 = graph_5.invoke({
    "companies": ["TechCorp", "InnovateCo", "FutureAI", "DataDynamics"]
})

print(f"\n🎯 FINAL RESULT:")
print(result_5['recommendation'])

# ============================================================================
# EXAMPLE 6: Dynamic Workload Distribution
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 6: Dynamic Workload - Variable Complexity")
print("=" * 70)

class WorkloadState(TypedDict):
    """State for variable workload processing"""
    tasks: List[dict]                               # Tasks with different complexity
    results: Annotated[List[dict], operator.add]    # Processing results
    stats: dict                                     # Statistics

def map_tasks(state: WorkloadState):
    """MAP: Create processor for each task"""
    print(f"\n📤 MAP: Distributing {len(state['tasks'])} tasks")
    return [
        Send("process_task", {"task": task})
        for task in state['tasks']
    ]

def process_task(state: WorkloadState):
    """
    PROCESS: Handle task with variable complexity

    Different tasks take different amounts of work
    """
    task = state['task']
    task_type = task['type']
    complexity = task['complexity']

    # Simulate different processing based on complexity
    processing_time = complexity * 0.1  # Mock time

    result = {
        "task": task['name'],
        "type": task_type,
        "complexity": complexity,
        "time": processing_time,
        "status": "completed"
    }

    print(f"  ⚙️  Processed: {task['name']} (complexity: {complexity}, time: {processing_time:.1f}s)")

    return {"results": [result]}

def generate_stats(state: WorkloadState):
    """REDUCE: Generate processing statistics"""
    results = state['results']

    total_time = sum(r['time'] for r in results)
    avg_complexity = sum(r['complexity'] for r in results) / len(results)

    stats = {
        "total_tasks": len(results),
        "total_time": total_time,
        "avg_complexity": avg_complexity,
        "types": {}
    }

    # Count by type
    for result in results:
        task_type = result['type']
        stats['types'][task_type] = stats['types'].get(task_type, 0) + 1

    print(f"\n✅ REDUCE: Generated statistics")
    print(f"   Total tasks: {stats['total_tasks']}")
    print(f"   Total time: {stats['total_time']:.1f}s")
    print(f"   Avg complexity: {stats['avg_complexity']:.1f}")

    return {"stats": stats}

# Build graph
builder_6 = StateGraph(WorkloadState)
builder_6.add_node("process_task", process_task)
builder_6.add_node("generate_stats", generate_stats)
builder_6.add_conditional_edges(START, map_tasks)
builder_6.add_edge("process_task", "generate_stats")
builder_6.add_edge("generate_stats", END)

graph_6 = builder_6.compile()

# Run example with variable complexity
result_6 = graph_6.invoke({
    "tasks": [
        {"name": "Task A", "type": "analysis", "complexity": 3},
        {"name": "Task B", "type": "processing", "complexity": 7},
        {"name": "Task C", "type": "analysis", "complexity": 2},
        {"name": "Task D", "type": "generation", "complexity": 5},
        {"name": "Task E", "type": "processing", "complexity": 8},
    ]
})

print(f"\n🎯 FINAL RESULT:")
print(f"   Tasks completed: {result_6['stats']['total_tasks']}")
print(f"   Types: {result_6['stats']['types']}")
print(f"   Average complexity: {result_6['stats']['avg_complexity']:.1f}")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

print("\n" + "=" * 70)
print("🎓 KEY TAKEAWAYS - Map-Reduce Patterns")
print("=" * 70)

print("""
1. **Send API = Dynamic Branching**
   - Create nodes at runtime based on data
   - Pattern: Send("node_name", {"data": value})

2. **Map-Reduce Flow**
   - MAP: Create Send() for each item
   - PROCESS: Handle each item in parallel
   - REDUCE: Combine all results

3. **Always Use Reducers**
   - results: Annotated[List, operator.add]
   - Combines parallel writes to same key

4. **When to Use Map-Reduce**
   ✅ Unknown # of items at design time
   ✅ Same operation on multiple items
   ✅ Batch processing
   ✅ Independent item processing

5. **When NOT to Use**
   ❌ Fixed # of operations (use parallelization)
   ❌ Sequential dependencies
   ❌ Single item processing

6. **Real-World Patterns**
   - Batch document processing
   - Multi-source research
   - Content generation at scale
   - Competitive analysis
   - ETL pipelines

7. **Advanced Patterns**
   - Nested map-reduce (Example 4)
   - Conditional processing (Example 2)
   - Multi-stage pipelines
   - Variable complexity workloads

8. **Performance Benefits**
   Sequential: N items × T time = N×T total
   Map-Reduce: max(T for all items) = T total
   Speedup: Up to N× faster!
""")

print("\n" + "=" * 70)
print("✅ MODULE 4 - LESSON 3 COMPLETE!")
print("=" * 70)
print("""
You now understand:
- Send API for dynamic branching
- Map-Reduce pattern implementation
- Batch processing at scale
- Real-world use cases

Next: Lesson 4 - Research Assistant (bringing it all together!)
""")
