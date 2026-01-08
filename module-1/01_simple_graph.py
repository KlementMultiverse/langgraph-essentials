"""
Simple Graph - Module 1

A basic LangGraph with 3 nodes and conditional routing.
Flow: START → node_1 → decide_mood() → node_2 OR node_3 → END
"""

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import random
from typing import Literal


# ============================================================================
# STEP 1: DEFINE STATE
# ============================================================================

class State(TypedDict):
    graph_state: str

# Q: How does the program know THIS is the state?
# A: You tell it explicitly when you create the graph:
#    builder = StateGraph(State)  ← You pass State here!
#    You can have multiple TypedDict classes, but only the one you pass
#    to StateGraph() becomes the state for that graph.


# ============================================================================
# STEP 2: DEFINE NODES
# ============================================================================

def node_1(state):
    print("---Node 1---")
    return {"graph_state": state['graph_state'] + " I am"}

# Q: Are we modifying the SAME variable or creating a NEW one?
# A: Creating a NEW string! Here's what happens:
#    1. Read old value: state['graph_state']
#    2. Create new string: old_value + " I am" (NEW object in memory)
#    3. Return: {"graph_state": new_string}
#    4. LangGraph replaces old value with new value
#    Same KEY name, different VALUE object.

# Q: Why not return state['graph_state'] = value?
# A: That's invalid Python syntax! The correct way is:
#    return {"graph_state": value}  ← Return a dict
#    LangGraph then does the assignment internally for you.

# Q: Must we return keys that are in State?
# A: YES! The key "graph_state" MUST exist in your State schema.
#    If you return a key not in State, it might error or be ignored.


def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] + " happy!"}


def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] + " sad!"}


# ============================================================================
# STEP 3: DEFINE CONDITIONAL EDGE
# ============================================================================

def decide_mood(state) -> Literal["node_2", "node_3"]:
    # Often, we will use state to decide on the next node to visit
    user_input = state['graph_state']

    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:
        # 50% of the time, we return Node 2
        return "node_2"

    # 50% of the time, we return Node 3
    return "node_3"

# NOTE: Conditional edges return a STRING (node name), not a dict!
# - Nodes return: {"key": value}  ← Updates state
# - Conditional edges return: "node_name"  ← Routing decision


# ============================================================================
# STEP 4: BUILD GRAPH
# ============================================================================

# Build graph
builder = StateGraph(State)  # ← Tell LangGraph to use State schema
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Compile
graph = builder.compile()


# ============================================================================
# STEP 5: RUN GRAPH
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SIMPLE GRAPH DEMO")
    print("=" * 60)

    # Run the graph
    result = graph.invoke({"graph_state": "Hi, this is Lance."})

    print("\nFinal result:")
    print(result)

    # Q: Can we use values from the returned state?
    # A: YES! Extract and use them:
    final_message = result['graph_state']
    print(f"\nExtracted message: {final_message}")

    if "happy" in final_message:
        print("→ Took the happy path!")
    elif "sad" in final_message:
        print("→ Took the sad path!")
