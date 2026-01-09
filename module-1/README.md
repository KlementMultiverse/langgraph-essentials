# Module 1: LangGraph Foundations

## Overview

Module 1 introduces the core concepts of LangGraph by building progressively from simple graphs to chains with tool calling.

**What you'll learn:**
- State management with TypedDict
- Nodes, edges, and conditional routing
- Messages and conversation history
- Reducers (add_messages)
- Tool binding and function calling
- Chat model integration

---

## Files

### 01_simple_graph.py
**Concepts:** State, Nodes, Edges, Conditional Routing

A basic LangGraph with 3 nodes demonstrating:
- State definition with TypedDict
- Node functions that update state
- Normal edges (fixed connections)
- Conditional edges (dynamic routing based on logic)

**Flow:**
```
START → node_1 → decide_mood() → [node_2 OR node_3] → END
```

**Run:**
```bash
python 01_simple_graph.py
```

**Key Learning:**
- Nodes return dictionaries matching state schema
- Conditional edges return node names (strings)
- State values are overwritten by default

---

### 02_chain_with_tool_calling.py
**Concepts:** Messages, Reducers, Tool Binding, Chat Models

A LangGraph chain demonstrating:
- Messages as state (HumanMessage, AIMessage)
- add_messages reducer (append vs overwrite)
- Tool binding to LLMs
- Function calling (LLM decides when to use tools)

**Flow:**
```
START → tool_calling_llm → END
```

**Run:**
```bash
python 02_chain_with_tool_calling.py
```

**Key Learning:**
- Reducers define HOW state updates merge
- add_messages preserves conversation history
- LLMs can request tool calls (but don't execute them yet)
- Nodes must return dicts matching state keys

---

## Concepts Explained

### State
The data structure passed between nodes. Defined using TypedDict.

```python
class State(TypedDict):
    messages: list[AnyMessage]
    counter: int
```

### Nodes
Python functions that:
1. Take state as input
2. Process/transform data
3. Return dict with updated state values

```python
def my_node(state: State):
    return {"counter": state["counter"] + 1}
```

### Edges
Connections between nodes:
- **Normal edges**: Always go from A → B
- **Conditional edges**: Choose next node based on logic

```python
builder.add_edge(START, "node_1")  # Normal
builder.add_conditional_edges("node_1", router_function)  # Conditional
```

### Reducers
Functions that control how new state values merge with old ones.

**Default behavior** (overwrite):
```python
old_state = {"messages": [msg1, msg2]}
new_value = {"messages": [msg3]}
result = {"messages": [msg3]}  # Lost msg1, msg2!
```

**With add_messages reducer** (append):
```python
old_state = {"messages": [msg1, msg2]}
new_value = {"messages": [msg3]}
result = {"messages": [msg1, msg2, msg3]}  # All preserved!
```

### Messages
Structured conversation format:
- **HumanMessage**: User input
- **AIMessage**: LLM response
- **SystemMessage**: Model instructions
- **ToolMessage**: Tool execution results

### Tool Calling
Binding Python functions to LLMs so they can request tool usage:

```python
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

llm_with_tools = llm.bind_tools([multiply])
```

LLM decides when to call tools based on user input.

---

## Progression

**01_simple_graph.py** → Basic LangGraph mechanics
- State, nodes, edges, routing

**02_chain_with_tool_calling.py** → Real-world patterns
- Messages, reducers, tools, chat models

**Next (Module 2):** Agent with tool execution loop

---

## Running the Examples

### Prerequisites
```bash
pip install langgraph langchain-openai python-dotenv
```

### Environment Setup
Create `.env` file:
```
OPENAI_API_KEY=your_key_here
```

### Run
```bash
cd module-1
python 01_simple_graph.py
python 02_chain_with_tool_calling.py
```

---

## Key Differences: Chain vs Agent

| Aspect | Chain | Agent |
|--------|-------|-------|
| Flow | Fixed (START → LLM → END) | Dynamic (loops back) |
| Tool execution | Requests only | Full execution |
| Complexity | Simple, predictable | Complex, flexible |
| Use case | Prepare tool calls | Autonomous task completion |

**Chain (Module 1):**
```
User → LLM → Tool Call Request → END
```

**Agent (Next):**
```
User → LLM → Execute Tool → LLM sees result → ... → END
```

---

## Common Questions

**Q: Why return `{"messages": [msg]}` instead of just `msg`?**
A: Nodes must return dicts matching state schema. If state has `messages` key, return `{"messages": ...}`.

**Q: Why wrap in list `[msg]`?**
A: The `add_messages` reducer expects a list. Even single messages should be wrapped.

**Q: Can I return only some state keys?**
A: YES! Only return the keys you want to UPDATE. Others remain unchanged.

**Q: What's the difference between node returns and conditional edge returns?**
A: Nodes return dicts `{"key": value}`, conditional edges return strings `"node_name"`.

**Q: When to use add_messages vs default behavior?**
A: Use `add_messages` when you want to preserve history (conversations). Use default when you want to replace values (counters, flags).

---

## Next Steps

After completing Module 1, you'll understand:
- ✅ How LangGraph executes (state → nodes → edges)
- ✅ How to structure conversation history
- ✅ How LLMs decide when to use tools
- ✅ The difference between chains and agents

**Next:** Build a full agent that executes tools and loops until task completion.

---

## Resources

- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **LangChain Academy:** https://academy.langchain.com/
- **GitHub Repo:** https://github.com/KlementMultiverse/langgraph-git

---

**Module 1 Complete!** 🎉

You've learned the foundations. Now you're ready to build real agents.
