"""
LangGraph Academy - Module 2: State Schema
===========================================

This demonstrates 3 ways to define state schemas in LangGraph:
1. TypedDict (type hints only, no runtime validation)
2. Dataclass (cleaner syntax, no runtime validation)
3. Pydantic (runtime validation with type checking)

Author: Learning from LangChain Academy
Date: 2026-01-10
"""

import random
from typing import Literal
from typing_extensions import TypedDict
from dataclasses import dataclass
from pydantic import BaseModel, field_validator, ValidationError
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display


# =============================================================================
# METHOD 1: TypedDict - Type Hints Only (No Runtime Validation)
# =============================================================================

class TypedDictState(TypedDict):
    """
    TypedDict provides type hints but does NOT enforce them at runtime.
    You can still assign invalid values without errors.
    """
    name: str
    mood: Literal["happy", "sad"]


# =============================================================================
# METHOD 2: Dataclass - Cleaner Syntax (No Runtime Validation)
# =============================================================================

@dataclass
class DataclassState:
    """
    Dataclass offers concise syntax for storing structured data.
    Also does NOT enforce types at runtime.
    """
    name: str
    mood: Literal["happy", "sad"]


# =============================================================================
# METHOD 3: Pydantic - Runtime Validation (RECOMMENDED for Production)
# =============================================================================

class PydanticState(BaseModel):
    """
    Pydantic validates data at runtime, catching errors before they happen.
    Best choice for production applications.
    """
    name: str
    mood: str  # Will be validated as "happy" or "sad"

    @field_validator('mood')
    @classmethod
    def validate_mood(cls, value):
        """Ensure mood is either 'happy' or 'sad'"""
        if value not in ["happy", "sad"]:
            raise ValueError("Each mood must be either 'happy' or 'sad'")
        return value


# =============================================================================
# Graph Nodes - Work with all 3 state types
# =============================================================================

def node_1(state):
    """
    First node - appends to name
    Works differently based on state type:
    - TypedDict: state['name']
    - Dataclass/Pydantic: state.name
    """
    print("---Node 1---")
    # For TypedDict, use state['name']
    # For Dataclass/Pydantic, use state.name
    if isinstance(state, dict):
        return {"name": state['name'] + " is ... "}
    else:
        return {"name": state.name + " is ... "}


def node_2(state):
    """Node 2 - sets mood to happy"""
    print("---Node 2---")
    return {"mood": "happy"}


def node_3(state):
    """Node 3 - sets mood to sad"""
    print("---Node 3---")
    return {"mood": "sad"}


def decide_mood(state) -> Literal["node_2", "node_3"]:
    """
    Conditional edge - randomly routes to node_2 or node_3
    50/50 split between happy and sad
    """
    if random.random() < 0.5:
        return "node_2"  # 50% chance
    return "node_3"  # 50% chance


# =============================================================================
# Build Graph with TypedDict
# =============================================================================

def build_typed_dict_graph():
    """Build graph using TypedDict state"""
    builder = StateGraph(TypedDictState)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)
    builder.add_node("node_3", node_3)

    # Logic
    builder.add_edge(START, "node_1")
    builder.add_conditional_edges("node_1", decide_mood)
    builder.add_edge("node_2", END)
    builder.add_edge("node_3", END)

    return builder.compile()


# =============================================================================
# Build Graph with Dataclass
# =============================================================================

def build_dataclass_graph():
    """Build graph using Dataclass state"""
    # Modify node_1 for dataclass access pattern
    def node_1_dataclass(state):
        print("---Node 1---")
        return {"name": state.name + " is ... "}

    builder = StateGraph(DataclassState)
    builder.add_node("node_1", node_1_dataclass)
    builder.add_node("node_2", node_2)
    builder.add_node("node_3", node_3)

    # Logic
    builder.add_edge(START, "node_1")
    builder.add_conditional_edges("node_1", decide_mood)
    builder.add_edge("node_2", END)
    builder.add_edge("node_3", END)

    return builder.compile()


# =============================================================================
# Build Graph with Pydantic
# =============================================================================

def build_pydantic_graph():
    """Build graph using Pydantic state (with validation!)"""
    # Modify node_1 for pydantic access pattern
    def node_1_pydantic(state):
        print("---Node 1---")
        return {"name": state.name + " is ... "}

    builder = StateGraph(PydanticState)
    builder.add_node("node_1", node_1_pydantic)
    builder.add_node("node_2", node_2)
    builder.add_node("node_3", node_3)

    # Logic
    builder.add_edge(START, "node_1")
    builder.add_conditional_edges("node_1", decide_mood)
    builder.add_edge("node_2", END)
    builder.add_edge("node_3", END)

    return builder.compile()


# =============================================================================
# MAIN - Run Examples
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LangGraph State Schema Examples")
    print("=" * 70)

    # Example 1: TypedDict
    print("\n1️⃣  TypedDict Example (No Runtime Validation)")
    print("-" * 70)
    graph_typed = build_typed_dict_graph()
    result = graph_typed.invoke({"name": "Lance", "mood": "sad"})
    print(f"Result: {result}")

    # TypedDict allows invalid values!
    print("\n⚠️  TypedDict ALLOWS invalid mood:")
    invalid_state = TypedDictState(name="John", mood="mad")  # This works!
    print(f"Created invalid state: {invalid_state}")

    # Example 2: Dataclass
    print("\n\n2️⃣  Dataclass Example (No Runtime Validation)")
    print("-" * 70)
    graph_dataclass = build_dataclass_graph()
    result = graph_dataclass.invoke(DataclassState(name="Lance", mood="happy"))
    print(f"Result: {result}")

    # Dataclass also allows invalid values!
    print("\n⚠️  Dataclass ALLOWS invalid mood:")
    invalid_dataclass = DataclassState(name="Jane", mood="angry")  # This works!
    print(f"Created invalid state: {invalid_dataclass}")

    # Example 3: Pydantic
    print("\n\n3️⃣  Pydantic Example (WITH Runtime Validation) ✅")
    print("-" * 70)
    graph_pydantic = build_pydantic_graph()
    result = graph_pydantic.invoke(PydanticState(name="Lance", mood="sad"))
    print(f"Result: {result}")

    # Pydantic catches invalid values!
    print("\n✅ Pydantic PREVENTS invalid mood:")
    try:
        invalid_pydantic = PydanticState(name="Bob", mood="furious")
    except ValidationError as e:
        print(f"Validation Error Caught: {e}")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAY:")
    print("Use Pydantic for production apps - it catches errors at runtime!")
    print("=" * 70)
