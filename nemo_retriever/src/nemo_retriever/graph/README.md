# NeMo Retriever Graph

The `nemo_retriever.graph` package contains the graph-based execution model used to build explicit ingestion pipelines from composable operators.

It is useful when you want to:

- Build a pipeline stage-by-stage instead of using a higher-level fluent API
- Reuse operators across multiple workflows
- Run the same graph in-process or with Ray Data
- Add your own custom logic as a first-class pipeline stage

## Core Concepts

The graph package is built around a few small pieces:

- `AbstractOperator`: base class for a pipeline stage
- `Node`: wraps an operator and tracks downstream children
- `Graph`: owns one or more root nodes and executes them
- `InprocessExecutor`: runs a graph locally on pandas DataFrames
- `RayDataExecutor`: runs a graph as a Ray Data pipeline
- `UDFOperator`: wraps a plain Python function as an operator

Most users interact with the graph by chaining operators with `>>`.

## Smallest Example

```python
from nemo_retriever.graph import Graph, UDFOperator


def double(x):
    return x * 2


graph = Graph()
graph.add_root(UDFOperator(double, name="Double"))
result = graph.execute(3)

print(result)  # [6]
```

Because `AbstractOperator.__rshift__` returns a `Graph`, chaining works naturally:

```python
from nemo_retriever.graph import UDFOperator


graph = (
    UDFOperator(lambda x: x + 1, name="AddOne")
    >> UDFOperator(lambda x: x * 10, name="TimesTen")
)

print(graph.execute(5))  # [60]
```

## Building a Multi-Stage Graph

Here is a more realistic example with a few named stages:

```python
from nemo_retriever.graph import UDFOperator


graph = (
    UDFOperator(lambda text: text.strip(), name="Trim")
    >> UDFOperator(lambda text: text.lower(), name="Lower")
    >> UDFOperator(lambda text: f"normalized::{text}", name="Prefix")
)

result = graph.execute("  Hello World  ")
print(result)  # ['normalized::hello world']
```

The returned value is a list because a graph can have multiple leaf nodes.

## Writing a Custom Operator

Use a custom operator when you want reusable logic, configuration, or access to the full `preprocess -> process -> postprocess` lifecycle.

```python
from typing import Any

from nemo_retriever.graph import AbstractOperator


class AddSuffixOperator(AbstractOperator):
    def __init__(self, suffix: str = "_done") -> None:
        super().__init__()
        self.suffix = suffix

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return str(data).strip()

    def process(self, data: Any, **kwargs: Any) -> Any:
        return f"{data}{self.suffix}"

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

```



```python
from nemo_retriever.graph import UDFOperator


graph = (
    UDFOperator(lambda text: text.upper(), name="Upper")
    >> AddSuffixOperator("_READY")
)

print(graph.execute("hello"))  # ['HELLO_READY']
```

## Operator Lifecycle

Each operator runs in three steps:

1. `preprocess(data, **kwargs)`
2. `process(data, **kwargs)`
3. `postprocess(data, **kwargs)`

`run()` calls those three in order, and both executors use that same lifecycle.

This pattern is helpful when you want to:

- Normalize or validate input in `preprocess`
- Perform the main computation in `process`
- Clean up or reshape output in `postprocess`

## Custom Operator Tips

- Keep `process()` focused on the main transformation
- Use `__init__()` for stable configuration like thresholds, URLs, or model names
- Accept `**kwargs` in lifecycle methods so your operator stays executor-friendly
- Return a value that the next stage can consume directly
- If the operator will run in `RayDataExecutor`, make sure construction is safe on workers

## Using `UDFOperator`

`UDFOperator` is the easiest way to adopt the graph system when you do not need a full class yet.

It lets you wrap a plain Python callable:

```python
from nemo_retriever.graph import Graph, UDFOperator


def uppercase_and_prefix(text: str) -> str:
    return f"PROCESSED: {text.upper()}"


graph = Graph()
graph.add_root(UDFOperator(uppercase_and_prefix, name="UppercasePrefix"))
print(graph.execute("hello world"))  # ['PROCESSED: HELLO WORLD']
```

You can also chain it with custom operators:

```python
graph = (
    UDFOperator(lambda x: x * 4, name="MultiplyByFour")
    >> UDFOperator(lambda x: x + 3, name="AddThree")
    >> UDFOperator(lambda x: f"{x}_done", name="Finalize")
)

print(graph.execute(2))  # ['11_done']
```

## When To Use `UDFOperator` vs Custom Operator

Use `UDFOperator` when:

- You want to prototype quickly
- The logic is small and self-contained
- You only need one transformation step
- You want a low-friction way to join existing graph stages

Use a custom `AbstractOperator` subclass when:

- The stage has configuration or internal state
- You want reusable code with a clear name
- You need `preprocess` or `postprocess`
- The implementation is complex enough that a class is easier to maintain

## Executing a Graph

### Direct execution

`Graph.execute()` is the simplest option and is good for small in-memory workflows:

```python
result = graph.execute("sample input")
```

### In-process execution

Use `InprocessExecutor` for linear graphs that should run locally on pandas DataFrames:

```python
from nemo_retriever.graph import InprocessExecutor


executor = InprocessExecutor(graph)
result_df = executor.ingest(["/path/to/file.pdf"])
```

### Ray Data execution

Use `RayDataExecutor` for linear graphs that should run with Ray Data:

```python
from nemo_retriever.graph import RayDataExecutor


executor = RayDataExecutor(graph, batch_size=8)
result_ds = executor.ingest(["/path/to/*.pdf"])
```

## Notes About Graph Shape

The base `Graph` type supports multiple roots and fan-out.

`InprocessExecutor` and `RayDataExecutor` currently support linear graphs only:

- one root
- one child per node

If you use direct `Graph.execute()`, branching graphs are supported and leaf outputs are collected into a list.

## Imports

Recommended imports:

```python
from nemo_retriever.graph import (
    AbstractOperator,
    Graph,
    InprocessExecutor,
    Node,
    RayDataExecutor,
    UDFOperator,
)
```

`nemo_retriever.graph` is the graph API location used by the codebase.

## NimClient and custom NIM endpoints

The `NimClient` class provides a unified interface for connecting to and interacting with NVIDIA NIM Microservices.
Use it to create custom NIM integrations in [NeMo Retriever Library](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/) pipelines and user-defined functions (UDFs).

The NimClient architecture consists of two main components:

1. **NimClient**: The client class that handles communication with NIM endpoints via gRPC or HTTP protocols
2. **ModelInterface**: An abstract base class that defines how to format input data, parse output responses, and process inference results for specific models

For advanced usage patterns, refer to the existing model interfaces in [`nemo_retriever/models/nim/primitives/model_interface/`](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/src/nemo_retriever/models/nim/primitives/model_interface).

### Quick start

For ingest and pipeline APIs used with NimClient in UDFs, refer to the [Python API guide](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/nemo-retriever-api-reference.md).

#### Basic NimClient creation

```python
from nemo_retriever.models.nim.util import create_inference_client
from nemo_retriever.models.nim.primitives import ModelInterface

# Create a custom model interface (refer to examples below)
model_interface = MyCustomModelInterface()

# Define endpoints (gRPC, HTTP)
endpoints = ("grpc://my-nim-service:8001", "http://my-nim-service:8000")

# Create the client
client = create_inference_client(
    endpoints=endpoints,
    model_interface=model_interface,
    auth_token="your-ngc-api-key",  # Optional
    infer_protocol="grpc",          # Optional: "grpc" or "http"
    timeout=120.0,                  # Optional: request timeout
    max_retries=5                   # Optional: retry attempts
)

# Perform inference
data = {"input": "your input data"}
results = client.infer(data, model_name="your-model-name")
```

#### Using environment variables

```python
import os
from nemo_retriever.models.nim.util import create_inference_client

# Use environment variables for configuration
auth_token = os.getenv("NGC_API_KEY")
grpc_endpoint = os.getenv("NIM_GRPC_ENDPOINT", "grpc://localhost:8001")
http_endpoint = os.getenv("NIM_HTTP_ENDPOINT", "http://localhost:8000")

client = create_inference_client(
    endpoints=(grpc_endpoint, http_endpoint),
    model_interface=model_interface,
    auth_token=auth_token
)
```

### Creating custom model interfaces

To integrate a new NIM, create a custom `ModelInterface` subclass that implements the required methods.

For ingest pipeline APIs used with custom NIMs, refer to the [Python API guide](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/nemo-retriever-api-reference.md).

#### Basic model interface template

```python
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from nemo_retriever.models.nim.primitives import ModelInterface

class MyCustomModelInterface(ModelInterface):
    """
    Custom model interface for My Custom NIM.
    """

    def __init__(self, model_name: str = "my-custom-model"):
        """Initialize the model interface."""
        self.model_name = model_name

    def name(self) -> str:
        """Return the name of this model interface."""
        return "MyCustomModel"

    def prepare_data_for_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "input_text" not in data:
            raise KeyError("Input data must include 'input_text'")
        if not isinstance(data["input_text"], str):
            raise ValueError("input_text must be a string")
        return data

    def format_input(
        self,
        data: Dict[str, Any],
        protocol: str,
        max_batch_size: int,
        **kwargs
    ) -> Tuple[List[Any], List[Dict[str, Any]]]:
        if protocol == "http":
            return self._format_http_input(data, max_batch_size, **kwargs)
        elif protocol == "grpc":
            return self._format_grpc_input(data, max_batch_size, **kwargs)
        raise ValueError("Invalid protocol. Must be 'grpc' or 'http'")

    def _format_http_input(
        self,
        data: Dict[str, Any],
        max_batch_size: int,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        input_text = data["input_text"]
        payload = {
            "model": kwargs.get("model_name", self.model_name),
            "input": input_text,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
        }
        return [payload], [{"original_input": input_text}]

    def _format_grpc_input(
        self,
        data: Dict[str, Any],
        max_batch_size: int,
        **kwargs
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        input_text = data["input_text"]
        text_array = np.array([[input_text.encode("utf-8")]], dtype=np.object_)
        return [text_array], [{"original_input": input_text}]

    def parse_output(
        self,
        response: Any,
        protocol: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        if protocol == "http":
            return self._parse_http_response(response)
        if protocol == "grpc":
            return self._parse_grpc_response(response)
        raise ValueError("Invalid protocol. Must be 'grpc' or 'http'")

    def _parse_http_response(self, response: Dict[str, Any]) -> str:
        if isinstance(response, dict):
            if "choices" in response:
                return response["choices"][0].get("text", "")
            if "output" in response:
                return response["output"]
            raise RuntimeError("Unexpected response format")
        return str(response)

    def _parse_grpc_response(self, response: np.ndarray) -> str:
        if isinstance(response, np.ndarray):
            return response.flatten()[0].decode("utf-8")
        return str(response)

    def process_inference_results(self, output: Any, protocol: str, **kwargs) -> Any:
        if isinstance(output, str):
            return output.strip()
        return output
```

### Real-world examples

For parameter details on ingest and pipeline APIs, refer to the [Python API guide](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/nemo-retriever-api-reference.md).

#### Text generation model interface

```python
class TextGenerationModelInterface(ModelInterface):
    """Interface for text generation NIMs (e.g., LLaMA, GPT-style models)."""

    def name(self) -> str:
        return "TextGeneration"

    def prepare_data_for_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "prompt" not in data:
            raise KeyError("Input data must include 'prompt'")
        return data

    def format_input(self, data: Dict[str, Any], protocol: str, max_batch_size: int, **kwargs):
        prompt = data["prompt"]
        if protocol == "http":
            payload = {
                "model": kwargs.get("model_name", "llama-2-7b-chat"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "stream": False
            }
            return [payload], [{"prompt": prompt}]
        raise ValueError("Only HTTP protocol supported for this model")

    def parse_output(self, response: Any, protocol: str, data: Optional[Dict[str, Any]] = None, **kwargs):
        if protocol == "http" and isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
        return str(response)

    def process_inference_results(self, output: Any, protocol: str, **kwargs):
        return output.strip() if isinstance(output, str) else output
```

#### Image analysis model interface

```python
from nemo_retriever.common.api.util.image_processing.transforms import numpy_to_base64

class ImageAnalysisModelInterface(ModelInterface):
    """Interface for image analysis NIMs (e.g., vision models)."""

    def name(self) -> str:
        return "ImageAnalysis"

    def prepare_data_for_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "images" not in data:
            raise KeyError("Input data must include 'images'")
        if not isinstance(data["images"], list):
            data["images"] = [data["images"]]
        return data

    def format_input(self, data: Dict[str, Any], protocol: str, max_batch_size: int, **kwargs):
        images = data["images"]
        prompt = data.get("prompt", "Describe this image.")
        base64_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                base64_images.append(numpy_to_base64(img))
            elif isinstance(img, str) and img.startswith("data:image"):
                base64_images.append(img.split(",")[1])
            else:
                base64_images.append(str(img))

        batches = [base64_images[i:i + max_batch_size]
                  for i in range(0, len(base64_images), max_batch_size)]
        payloads = []
        batch_data_list = []

        for batch in batches:
            if protocol == "http":
                messages = []
                for img_b64 in batch:
                    messages.append({
                        "role": "user",
                        "content": f'{prompt} <img src="data:image/png;base64,{img_b64}" />'
                    })
                payload = {
                    "model": kwargs.get("model_name", "llava-1.5-7b-hf"),
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", 512),
                    "temperature": kwargs.get("temperature", 0.1)
                }
                payloads.append(payload)
                batch_data_list.append({"images": batch, "prompt": prompt})

        return payloads, batch_data_list

    def parse_output(self, response: Any, protocol: str, data: Optional[Dict[str, Any]] = None, **kwargs):
        if protocol == "http" and isinstance(response, dict):
            choices = response.get("choices", [])
            return [choice.get("message", {}).get("content", "") for choice in choices]
        return [str(response)]

    def process_inference_results(self, output: Any, protocol: str, **kwargs):
        if isinstance(output, list):
            return [result.strip() for result in output]
        return output
```

### Using NimClient in UDFs

For ingest pipeline APIs used in UDFs, refer to the [Python API guide](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/nemo-retriever-api-reference.md).

#### Basic UDF with NimClient

```python
from nemo_retriever.common.api.internal.primitives.ingest_control_message import IngestControlMessage
from nemo_retriever.models.nim.util import create_inference_client
import os

def analyze_document_with_nim(control_message: IngestControlMessage) -> IngestControlMessage:
    """UDF that uses a custom NIM to analyze document content."""
    model_interface = TextGenerationModelInterface()
    client = create_inference_client(
        endpoints=(
            os.getenv("ANALYSIS_NIM_GRPC", "grpc://analysis-nim:8001"),
            os.getenv("ANALYSIS_NIM_HTTP", "http://analysis-nim:8000")
        ),
        model_interface=model_interface,
        auth_token=os.getenv("NGC_API_KEY"),
        infer_protocol="http"
    )

    df = control_message.get_payload()

    for idx, row in df.iterrows():
        if row.get("content"):
            prompt = f"Analyze the following document content and provide a summary: {row['content'][:1000]}"
            try:
                results = client.infer(
                    data={"prompt": prompt},
                    model_name="llama-2-7b-chat",
                    max_tokens=256,
                    temperature=0.3
                )
                if results:
                    analysis = results[0] if isinstance(results, list) else results
                    df.at[idx, "custom_analysis"] = analysis
            except Exception as e:
                print(f"NIM inference failed: {e}")
                df.at[idx, "custom_analysis"] = "Analysis failed"

    control_message.payload(df)
    return control_message
```

#### Advanced UDF with batching

```python
def batch_image_analysis_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    """UDF that performs batched image analysis using NIM."""
    model_interface = ImageAnalysisModelInterface()
    client = create_inference_client(
        endpoints=(
            os.getenv("VISION_NIM_GRPC", "grpc://vision-nim:8001"),
            os.getenv("VISION_NIM_HTTP", "http://vision-nim:8000")
        ),
        model_interface=model_interface,
        auth_token=os.getenv("NGC_API_KEY")
    )

    df = control_message.get_payload()
    image_rows = []
    images = []

    for idx, row in df.iterrows():
        if "image_data" in row and row["image_data"]:
            image_rows.append(idx)
            images.append(row["image_data"])

    if images:
        try:
            results = client.infer(
                data={
                    "images": images,
                    "prompt": "Describe the content and key elements in this image."
                },
                model_name="llava-1.5-7b-hf",
                max_tokens=200
            )
            for idx, result in zip(image_rows, results):
                df.at[idx, "image_description"] = result
        except Exception as e:
            print(f"Batch image analysis failed: {e}")
            for idx in image_rows:
                df.at[idx, "image_description"] = "Analysis failed"

    control_message.payload(df)
    return control_message
```

### Configuration and best practices

#### Environment variables

```bash
# NIM endpoints
export MY_NIM_GRPC_ENDPOINT="grpc://my-nim-service:8001"
export MY_NIM_HTTP_ENDPOINT="http://my-nim-service:8000"

# Authentication
export NGC_API_KEY="your-ngc-api-key"

# Optional: timeouts and retries
export NIM_TIMEOUT=120
export NIM_MAX_RETRIES=5
```

#### Performance optimization

1. **Use gRPC when possible**: Generally faster than HTTP for high-throughput scenarios
2. **Batch processing**: Process multiple items together to reduce overhead
3. **Connection reuse**: Create NimClient instances once and reuse them
4. **Appropriate timeouts**: Set reasonable timeouts based on your model's response time
5. **Error handling**: Always handle inference failures gracefully

#### Error handling

```python
def robust_nim_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    """UDF with comprehensive error handling."""
    try:
        client = create_inference_client(
            endpoints=(grpc_endpoint, http_endpoint),
            model_interface=model_interface,
            auth_token=auth_token,
            timeout=60.0,
            max_retries=3
        )
    except Exception as e:
        print(f"Failed to create NIM client: {e}")
        return control_message

    df = control_message.get_payload()

    for idx, row in df.iterrows():
        try:
            results = client.infer(data=input_data, model_name="my-model")
            df.at[idx, "nim_result"] = results
        except TimeoutError:
            print(f"NIM request timed out for row {idx}")
            df.at[idx, "nim_result"] = "timeout"
        except Exception as e:
            print(f"NIM inference failed for row {idx}: {e}")
            df.at[idx, "nim_result"] = "error"

    control_message.payload(df)
    return control_message
```

### Troubleshooting

#### Common issues

* **Connection Errors** – Verify NIM service is running and endpoints are correct
* **Authentication Failures** – Check NGC_API_KEY is valid and properly set
* **Timeout Errors** – Increase timeout values or check NIM service performance
* **Format Errors** – Ensure your ModelInterface formats data correctly for your NIM
* **Memory Issues** – Use appropriate batch sizes to avoid memory exhaustion

#### NIM Triton limit memory

If you encounter memory issues, try increasing the `NIM_TRITON_CUDA_MEMORY_POOL_MB` parameter. This adjustment typically does not affect performance.

If memory issues persist, you can reduce the `NIM_TRITON_RATE_LIMIT` value — even down to 1. However, lowering this parameter affects performance.

#### Debugging tips

```python
import logging

logging.getLogger("nemo_retriever.models.nim").setLevel(logging.DEBUG)

model_interface = MyCustomModelInterface()
test_data = {"input": "test"}
prepared = model_interface.prepare_data_for_inference(test_data)
print(f"Prepared data: {prepared}")
formatted, batch_data = model_interface.format_input(prepared, "http", 1)
print(f"Formatted input: {formatted}")
```
