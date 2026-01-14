# Module 4: Parallelization & Advanced Patterns

**Master multi-agent workflows, parallel execution, and complex graph patterns**

---

## 📚 Overview

This module builds on human-in-the-loop and memory concepts to create sophisticated multi-agent systems. Learn to run operations in parallel, process multiple items with map-reduce, and compose modular graphs.

---

## 🎯 Learning Objectives

By completing this module, you will be able to:

✅ **Run nodes in parallel** for faster execution
✅ **Use reducers** to combine parallel state updates
✅ **Build map-reduce patterns** for batch processing
✅ **Create sub-graphs** for modular composition
✅ **Design multi-agent systems** with complex workflows

---

## 📂 Module Contents

### **Lesson 1: Parallelization** (`01_parallelization.py`) ✅

Learn to run multiple nodes simultaneously using fan-out and fan-in patterns.

**Key Concepts:**
- Fan-out: One node branches to multiple nodes
- Fan-in: Multiple nodes converge to one node
- Reducers for parallel state updates (`operator.add`)
- Custom reducers for ordering
- Parallel data gathering (Wikipedia + Web Search)

**Patterns:**
```
          ┌─→ node_b ─┐
START → a─┤           ├─→ d → END
          └─→ node_c ─┘
```

**Real-World Use Cases:**
- Multi-source data gathering
- Concurrent API calls
- Independent operation execution
- Research assistants with multiple sources

**Run it:**
```bash
python 01_parallelization.py
```

---

### **Lesson 2: Sub-Graphs** (`02_sub_graphs.py`) ✅

Build modular graphs by composing sub-graphs with separate state schemas.

**Key Concepts:**
- Sub-graphs as modular components
- Input/output state schemas (`output_schema` parameter)
- State communication through overlapping keys
- Parallel sub-graph execution
- Graph composition and reusability

**Patterns:**
```
                    ┌─→ sub_graph_1 ─┐
START → prepare ───┤                 ├─→ combine → END
                    └─→ sub_graph_2 ─┘

Each sub-graph has its own state schema!
```

**Real-World Use Cases:**
- Multi-agent teams (each agent is a sub-graph)
- Log analysis systems (failure + question analysis)
- Modular ETL pipelines
- Reusable workflow components
- Independent service orchestration

**Run it:**
```bash
python 02_sub_graphs.py
```

---

### **Lesson 3: Map-Reduce** (`03_map_reduce.py`) ⏳ Coming Soon

Process multiple items in parallel using map-reduce patterns.

**Key Concepts:**
- Send API for dynamic branching
- Map: Apply operation to each item
- Reduce: Combine results
- Batch processing patterns

---

### **Lesson 4: Research Assistant** (`04_research_assistant.py`) ⏳ Coming Soon

Tie everything together in a production multi-agent system.

**Key Concepts:**
- Multi-agent orchestration
- Memory integration
- Human-in-the-loop workflows
- Complete production patterns

---

## 🔥 Key Patterns

### **Sequential Execution** (Previous Modules)
```
START → node_1 → node_2 → node_3 → END
Time: t1 + t2 + t3
```

### **Parallel Execution** (This Module)
```
          ┌─→ node_b ─┐
START → a─┤           ├─→ d → END
          └─→ node_c ─┘
Time: t_a + max(t_b, t_c) + t_d
```

**Speedup:** When t_b and t_c overlap, total time is reduced!

---

## 💡 Reducers Explained

### **Without Reducer (FAILS)**
```python
class State(TypedDict):
    data: List[str]  # ERROR: Multiple writes in same step
```

### **With Reducer (SUCCESS)**
```python
import operator
from typing import Annotated

class State(TypedDict):
    data: Annotated[list, operator.add]  # Combines parallel writes
```

**What `operator.add` does:**
```python
[1, 2] + [3] = [1, 2, 3]  # List concatenation
```

**Custom Reducer Example:**
```python
def sorting_reducer(left, right):
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return sorted(left + right)

class State(TypedDict):
    data: Annotated[list, sorting_reducer]  # Sorts all values
```

---

## 🏗️ Architecture Patterns

### **Simple Fan-Out/Fan-In**
```python
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "d")
builder.add_edge("c", "d")
```

### **Parallel Research**
```python
# Fan-out to multiple sources
builder.add_edge(START, "search_web")
builder.add_edge(START, "search_wikipedia")

# Fan-in to combine results
builder.add_edge("search_web", "generate_answer")
builder.add_edge("search_wikipedia", "generate_answer")
```

### **Waiting for All Paths**
```python
# d waits for BOTH b2 AND c
builder.add_edge(["b2", "c"], "d")
```

### **Sub-Graph Composition**
```python
# Define sub-graph with separate state
class SubGraphState(TypedDict):
    input_data: str
    result: str

sub_builder = StateGraph(state_schema=SubGraphState)
sub_graph = sub_builder.compile()

# Use sub-graph as node in parent
parent_builder.add_node("process", sub_graph)
```

### **Sub-Graph with Output Schema**
```python
# Control what data sub-graph returns
class SubGraphOutputState(TypedDict):
    result: str  # Only return result, not input_data

sub_builder = StateGraph(
    state_schema=SubGraphState,
    output_schema=SubGraphOutputState  # Filters returned keys
)
```

---

## 🛠️ Technical Details

### **When Parallelization Works**
✅ Independent operations (no dependencies)
✅ I/O-bound tasks (API calls, file reads)
✅ Multi-source data gathering
✅ Read-only operations

### **When to Avoid**
❌ Sequential dependencies (A must complete before B)
❌ Shared mutable resources without synchronization
❌ Order-dependent operations
❌ CPU-bound single-threaded operations

### **Performance Considerations**
```
Sequential:    3 API calls × 2s each = 6s total
Parallel:      3 API calls in parallel = 2s total
Speedup:       3x faster! ⚡
```

---

## 🎓 Key Takeaways

1. **Parallelization = Speed**
   - Run independent operations simultaneously
   - Reduce overall execution time
   - Better resource utilization

2. **Reducers Enable Parallel Writes**
   - Multiple nodes can write to same key
   - Reducer combines the values
   - `operator.add` for simple concatenation
   - Custom reducers for complex logic

3. **Graph Synchronization**
   - Fan-in nodes wait for ALL fan-out nodes
   - No partial results processed
   - Ensures data completeness

4. **Sub-Graphs = Modularity**
   - Each sub-graph has its own state schema
   - State communication through **overlapping keys**
   - Use `output_schema` to control returned data
   - Build reusable, composable components
   - Run sub-graphs in parallel for independent tasks

5. **State Communication Pattern**
   ```
   Parent provides:  cleaned_logs
                          ↓
   Sub-graph uses:   cleaned_logs → processes → generates result
                          ↓
   Sub-graph returns: result (filtered by output_schema)
                          ↓
   Parent receives:  result (merged into parent state)
   ```

6. **Real-World Impact**
   - Multi-source research: 40-60% faster
   - Concurrent API calls: Linear speedup
   - Better user experience (lower latency)
   - Modular architecture: Easier to test and maintain

7. **Design Principles**
   - Identify independent operations
   - Use appropriate reducers
   - Use sub-graphs for logical modularity
   - Define clear input/output contracts
   - Consider failure handling
   - Test with various timing scenarios

---

## 🚀 Next Steps

After mastering Lessons 1-2, you're ready for:

- **Lesson 3**: Map-Reduce - Process multiple items in parallel
- **Lesson 4**: Research Assistant - Complete multi-agent system

---

## 📖 Additional Resources

- [LangGraph Docs: Branching](https://langchain-ai.github.io/langgraph/how-tos/branching/)
- [LangGraph Docs: Reducers](https://langchain-ai.github.io/langgraph/concepts/#reducers)
- [LangGraph Docs: Sub-Graphs](https://langchain-ai.github.io/langgraph/how-tos/subgraph/)
- [LangGraph Docs: State Schemas](https://langchain-ai.github.io/langgraph/concepts/#state)
- [Python operator module](https://docs.python.org/3/library/operator.html)

---

## 🏆 Progress Tracker

| Lesson | Status | Completion |
|--------|--------|------------|
| 1. Parallelization | ✅ Complete | 100% |
| 2. Sub-Graphs | ✅ Complete | 100% |
| 3. Map-Reduce | ⏳ Not Started | 0% |
| 4. Research Assistant | ⏳ Not Started | 0% |

**Module Progress:** 50% Complete (2/4 lessons)

---

**Status**: 🔄 In Progress
**Last Updated**: 2026-01-14
**Next Lesson**: Map-Reduce
