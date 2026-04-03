# Graph Pipeline Registry

**Module:** `nemo_retriever.graph.graph_pipeline_registry`
**Source:** `nemo_retriever/src/nemo_retriever/graph/graph_pipeline_registry.py`

## What is it?

The Graph Pipeline Registry is a central store for **golden pipeline graphs** —
team-approved, named graph definitions that can be imported and reused
anywhere in the codebase instead of building up custom `Graph` objects from
scratch every time.

Each entry in the registry is a **blueprint**: a factory function plus
metadata (name, version, description, tags).  Calling `registry.build(name)`
produces a fresh `Graph` instance from that blueprint, ready to hand to an
executor or modify further.

## Why does it exist?

Before the registry, every call-site that needed a pipeline had to manually
assemble operators into a `Graph` with `>>` chains, duplicate configuration
across files, and hope the result stayed in sync with the "canonical" version.
The registry solves this by providing:

- **A single source of truth** — define a pipeline once, import it everywhere.
- **Rich introspection** — print tree views, dump every node's kwargs, generate
  full reports, all without running the pipeline.
- **Configuration overrides** — build a graph and tweak specific node kwargs
  without rewriting the whole chain.
- **Graph diffing** — compare two graph instances and see exactly what changed
  (structural differences, added/removed/changed kwargs).
- **JSON serialization** — save graphs to disk and reload them, useful for
  sharing configs, auditing, and reproducibility.

## Quick start

### 1. Register a pipeline

Use the `default_registry` singleton so all parts of the codebase share the
same set of pipelines.

**Decorator form (preferred):**

```python
from nemo_retriever.graph.graph_pipeline_registry import default_registry
from nemo_retriever.graph import Graph

@default_registry.register(
    "pdf-extract-basic",
    description="Minimal PDF extraction pipeline",
    version="1.0.0",
    tags=["pdf", "extraction"],
)
def _build_pdf_basic():
    from nemo_retriever.pdf.split import PDFSplitActor
    from nemo_retriever.pdf.extract import PDFExtractionActor

    return (
        Graph()
        >> PDFSplitActor()
        >> PDFExtractionActor(method="pdfium", dpi=300)
    )
```

**Imperative form:**

```python
def my_factory():
    return Graph() >> SomeOperator() >> AnotherOperator()

default_registry.register_graph(
    "my-pipeline",
    my_factory,
    description="Does something useful",
    version="2.1.0",
    tags=["text"],
)
```

### 2. Build a graph from the registry

```python
graph = default_registry.build("pdf-extract-basic")
# graph is a fresh Graph instance, ready for an executor
```

### 3. Build with overrides

Need the same pipeline but with different DPI?  No need to re-define it:

```python
graph = default_registry.build_with_overrides(
    "pdf-extract-basic",
    overrides={
        "PDFExtractionActor": {"dpi": 150},
    },
)
```

## Inspecting graphs

The registry and its standalone utility functions give you several ways to
understand what a graph looks like without running it.

### Summary

```python
from nemo_retriever.graph.graph_pipeline_registry import format_graph_summary

graph = default_registry.build("pdf-extract-basic")
print(format_graph_summary(graph))
```

```
Graph Summary
  Roots (1) : ['PDFSplitActor']
  Leaves (1): ['PDFExtractionActor']
  Total nodes    : 2
  Max depth      : 1
```

### Tree view

```python
from nemo_retriever.graph.graph_pipeline_registry import format_graph_tree

print(format_graph_tree(graph, show_kwargs=True))
```

```
PDFSplitActor  (nemo_retriever.pdf.split.PDFSplitActor)
└── PDFExtractionActor  (nemo_retriever.pdf.extract.PDFExtractionActor)
      ╰ dpi = 300
      ╰ method = 'pdfium'
```

### Full report

```python
# Through the registry (includes blueprint metadata):
default_registry.print_graph("pdf-extract-basic")

# Or directly on any Graph instance:
from nemo_retriever.graph.graph_pipeline_registry import print_graph
print_graph(graph)
```

### Listing the registry

```python
default_registry.print_summary()
```

```
Name                                Version    Nodes  Depth  Tags
----------------------------------------------------------------------
pdf-extract-basic                   1.0.0          2      1  pdf, extraction
```

### Querying nodes and kwargs

```python
from nemo_retriever.graph.graph_pipeline_registry import (
    find_node,
    get_node_kwargs,
    list_all_kwargs,
    collect_nodes,
    leaf_nodes,
)

# Find a specific node
node = find_node(graph, "PDFExtractionActor")

# Get its kwargs as a dict
kwargs = get_node_kwargs(graph, "PDFExtractionActor")
# {'method': 'pdfium', 'dpi': 300, ...}

# Get kwargs for every node at once
all_kw = list_all_kwargs(graph)
# {'PDFSplitActor': {...}, 'PDFExtractionActor': {...}}

# List all leaf nodes
leaves = leaf_nodes(graph)
```

## Modifying graph configuration

These functions mutate a graph instance **in-place**, so build a fresh copy
from the registry first if you need to preserve the original.

```python
from nemo_retriever.graph.graph_pipeline_registry import (
    update_node_kwargs,
    remove_node_kwargs,
    replace_node_kwargs,
    clone_graph,
)

graph = default_registry.build("pdf-extract-basic")

# Merge new values into existing kwargs
update_node_kwargs(graph, "PDFExtractionActor", {"dpi": 150, "extract_text": True})

# Remove specific keys
remove_node_kwargs(graph, "PDFExtractionActor", ["extract_text"])

# Replace kwargs entirely
replace_node_kwargs(graph, "PDFExtractionActor", {"method": "ocr", "dpi": 200})
```

To modify all nodes that share a name (e.g. if the same operator appears
twice), pass `all_matches=True`.

`clone_graph(graph)` creates an independent deep-copy via serialization
round-trip, useful when you want to derive a variant without touching the
original.

## Comparing graphs (diff)

When debugging or reviewing changes, diff two graphs to see exactly what
is different — structurally, in operator classes, and in kwargs.

```python
graph_a = default_registry.build("pdf-extract-basic")
graph_b = default_registry.build_with_overrides(
    "pdf-extract-basic",
    overrides={"PDFExtractionActor": {"dpi": 150}},
)

# Get a structured diff object
from nemo_retriever.graph.graph_pipeline_registry import diff_graphs
result = diff_graphs(graph_a, graph_b)
print(result.format())
```

```
========================================================================
GRAPH COMPARISON REPORT
========================================================================
  Identical        : False
  Structural match : True
  Nodes (A / B)    : 2 / 2
  Roots (A)        : ['PDFSplitActor']
  Roots (B)        : ['PDFSplitActor']

------------------------------------------------------------------------
NODE DIFFS
------------------------------------------------------------------------

  Position: root[0]/PDFSplitActor -> PDFExtractionActor
    Node     : 'PDFExtractionActor'
    ~ Changed kwargs:
        dpi: 300 -> 150
========================================================================
```

You can also diff by name directly through the registry:

```python
default_registry.print_diff("pipeline-a", "pipeline-b")
```

The `GraphDiff` object exposes structured data (`node_diffs`, `nodes_only_in_a`,
`nodes_only_in_b`, etc.) for programmatic inspection beyond the formatted
report.

## Serialization

### Save / load a single graph

```python
from nemo_retriever.graph.graph_pipeline_registry import save_graph, load_graph

graph = default_registry.build("pdf-extract-basic")
save_graph(graph, "pipeline.json")

restored = load_graph("pipeline.json")
```

### Save / load through the registry

```python
# Export every registered graph to one file
default_registry.save_all("all_pipelines.json")

# Import into a (possibly different) registry
other_registry = GraphPipelineRegistry()
loaded_names = other_registry.load_all("all_pipelines.json")

# Single-graph save/load through the registry
default_registry.save_graph("pdf-extract-basic", "pdf_basic.json")
default_registry.load_graph("pdf_basic.json", name="pdf-basic-imported")
```

The JSON format captures each node's operator class (as a fully qualified
import path), its `operator_kwargs`, and the tree structure.  Non-JSON-native
types (`Path`, `set`, class references, callables) are encoded with tagged
wrappers and restored automatically on load.

If an operator class cannot be instantiated during deserialization (e.g. the
dependency is not installed), a `_PlaceholderOperator` is substituted so the
graph structure is still available for inspection and diffing.

## Migrating existing code

If you currently build graphs inline like this:

```python
# before — ad-hoc graph construction
graph = Graph()
graph = graph >> PDFSplitActor() >> PDFExtractionActor(method="pdfium", dpi=300)
executor = InprocessExecutor(graph)
executor.ingest(data)
```

Move the construction into a registered factory, then consume it:

```python
# step 1 — register once (e.g. in a pipelines module)
@default_registry.register("pdf-extract-basic", description="...", tags=["pdf"])
def _build():
    return Graph() >> PDFSplitActor() >> PDFExtractionActor(method="pdfium", dpi=300)

# step 2 — use everywhere
graph = default_registry.build("pdf-extract-basic")
executor = InprocessExecutor(graph)
executor.ingest(data)
```

If the call-site needs to tweak parameters, use `build_with_overrides` instead
of rewriting the chain:

```python
graph = default_registry.build_with_overrides(
    "pdf-extract-basic",
    overrides={"PDFExtractionActor": {"dpi": 150}},
)
```

## API reference (quick)

### Standalone utility functions

All of these work on any `Graph` instance — no registry required.

| Function | Purpose |
|----------|---------|
| `walk_nodes(graph)` | Iterate `(node, depth)` pairs via DFS |
| `collect_nodes(graph)` | Ordered list of all unique nodes |
| `node_count(graph)` | Total unique node count |
| `max_depth(graph)` | Longest root-to-leaf path |
| `find_node(graph, name)` | First node matching name (or `None`) |
| `find_nodes(graph, name)` | All nodes matching name |
| `leaf_nodes(graph)` | All nodes with no children |
| `get_node_kwargs(graph, name)` | Kwargs dict for a named node |
| `list_all_kwargs(graph)` | `{node_name: kwargs}` for every node |
| `format_graph_tree(graph)` | Tree-view string |
| `format_graph_summary(graph)` | Summary string |
| `format_node_details(node)` | Detailed single-node string |
| `format_full_report(graph)` | Complete report string |
| `print_graph(graph)` | Print full report to stdout |
| `update_node_kwargs(graph, name, updates)` | Merge kwargs in-place |
| `remove_node_kwargs(graph, name, keys)` | Drop kwargs keys in-place |
| `replace_node_kwargs(graph, name, kwargs)` | Replace kwargs entirely |
| `diff_graphs(a, b)` | Structural + kwarg diff (`GraphDiff`) |
| `print_diff(a, b)` | Print formatted diff |
| `serialize_graph(graph)` | Graph to JSON-safe dict |
| `deserialize_graph(data)` | Dict back to `Graph` |
| `save_graph(graph, path)` | Write graph JSON to file |
| `load_graph(path)` | Read graph JSON from file |
| `clone_graph(graph)` | Deep-copy via serialization |

### `GraphPipelineRegistry` methods

| Method | Purpose |
|--------|---------|
| `register(name, ...)` | Decorator for registering a factory |
| `register_graph(name, factory, ...)` | Imperative registration |
| `unregister(name)` | Remove a blueprint |
| `build(name)` | Build a fresh `Graph` |
| `build_with_overrides(name, overrides)` | Build + apply kwarg patches |
| `get_blueprint(name)` | Access `GraphBlueprint` metadata |
| `list_names()` | All registered names |
| `list_blueprints(tag=...)` | Blueprints, optionally filtered |
| `print_graph(name)` | Full inspection to stdout |
| `print_summary()` | Compact table of all graphs |
| `get_graph_info(name)` | Full report as string |
| `diff(name_a, name_b)` | Diff two registered graphs |
| `print_diff(name_a, name_b)` | Print diff to stdout |
| `save_all(path)` | Export entire registry to JSON |
| `load_all(path)` | Import graphs from JSON |
| `save_graph(name, path)` | Export one graph to JSON |
| `load_graph(path, name=...)` | Import one graph from JSON |
