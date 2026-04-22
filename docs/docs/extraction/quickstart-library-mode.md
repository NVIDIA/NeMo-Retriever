# Deploy Without Containers (Library Mode) for NeMo Retriever Library

!!! note

    NVIDIA Ingest (nv-ingest) has been renamed NeMo Retriever Library.

Use the [Quick Start for NeMo Retriever Library](https://github.com/NVIDIA/NeMo-Retriever/blob/26.03/nemo_retriever/README.md) to set up and run the NeMo Retriever Library locally, so you can build a GPU‑accelerated, multimodal RAG ingestion pipeline that parses PDFs, HTML, text, audio, and video into LanceDB vector embeddings, integrates with Nemotron RAG models (locally or via NIM endpoints), which includes Ray‑based scaling with built‑in recall evaluation. Python 3.12 or later is required (see [Prerequisites](prerequisites.md)).

## `run_pipeline`

The primary Python entry point for launching the Ray-based ingestion pipeline in library mode is `run_pipeline` in `nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners`.

```python
import time

from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest_client.client.interface import Ingestor
from nv_ingest_client.client import NvIngestClient
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

def main():
    # Start the pipeline subprocess for library mode
    run_pipeline(block=False, disable_dynamic_scaling=True, run_in_subprocess=True)

    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost",
    )

    # gpu_cagra accelerated indexing is not available in milvus-lite
    # Provide a filename for milvus_uri to use milvus-lite
    milvus_uri = "milvus.db"
    collection_name = "test"
    sparse = False

    # do content extraction from files
    ingestor = (
        Ingestor(client=client)
        .files("data/multimodal_test.pdf")
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=True,
            table_output_format="markdown",
            extract_infographics=True,
            # extract_method="nemotron_parse", #Slower, but maximally accurate, especially for PDFs with pages that are scanned images
            text_depth="page",
        )
        .embed()
        .vdb_upload(
            collection_name=collection_name,
            milvus_uri=milvus_uri,
            sparse=sparse,
            # for llama-3.2 embedder, use 1024 for e5-v5
            dense_dim=2048,
        )
    )

    print("Starting ingestion..")
    t0 = time.time()

    # Return both successes and failures
    # Use for large batches where you want successful chunks/pages to be committed, while collecting detailed diagnostics for failures.
    results, failures = ingestor.ingest(show_progress=True, return_failures=True)

    # Return only successes
    # results = ingestor.ingest(show_progress=True)

    t1 = time.time()
    print(f"Total time: {t1 - t0} seconds")

    # results blob is directly inspectable
    if results:
        print(ingest_json_results_to_blob(results[0]))

    # (optional) Review any failures that were returned
    if failures:
        print(f"There were {len(failures)} failures. Sample: {failures[0]}")

if __name__ == "__main__":
    main()
```

!!! note

    For advanced visual parsing with library mode, uncomment `extract_method="nemotron_parse"` in the previous code. For more information, refer to [Advanced Visual Parsing](nemoretriever-parse.md).


You can see the extracted text that represents the content of the ingested test document.

```shell
Starting ingestion..
Total time: 9.243880033493042 seconds

TestingDocument
A sample document with headings and placeholder text
Introduction
This is a placeholder document that can be used for any purpose. It contains some 
headings and some placeholder text to fill the space. The text is not important and contains 
no real value, but it is useful for testing. Below, we will have some simple tables and charts 
that we can use to confirm Ingest is working as expected.
Table 1
This table describes some animals, and some activities they might be doing in specific 
locations.
Animal Activity Place
Gira@e Driving a car At the beach
Lion Putting on sunscreen At the park
Cat Jumping onto a laptop In a home o@ice
Dog Chasing a squirrel In the front yard
Chart 1
This chart shows some gadgets, and some very fictitious costs.

... document extract continues ...
```

## Step 3: Query Ingested Content

To query for relevant snippets of the ingested content, and use them with an LLM to generate answers, use the following code.

```python
import os
from openai import OpenAI
from nv_ingest_client.util.vdb.milvus import nvingest_retrieval

milvus_uri = "milvus.db"
collection_name = "test"
sparse=False

queries = ["Which animal is responsible for the typos?"]

retrieved_docs = nvingest_retrieval(
    queries,
    collection_name=collection_name,
    milvus_uri=milvus_uri,
    hybrid=sparse,
    top_k=1,
)

# simple generation example
extract = retrieved_docs[0][0].get("entity", retrieved_docs[0][0]).get("text", "")
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.environ["NVIDIA_API_KEY"]
)

prompt = f"Using the following content: {extract}\n\n Answer the user query: {queries[0]}"
print(f"Prompt: {prompt}")
completion = client.chat.completions.create(
  model="nvidia/nemotron-nano-12b-v2-vl",
  messages=[{"role":"user","content": prompt}],
)
response = completion.choices[0].message.content

print(f"Answer: {response}")
```

```shell
Prompt: Using the following content: Table 1
| This table describes some animals, and some activities they might be doing in specific locations. | This table describes some animals, and some activities they might be doing in specific locations. | This table describes some animals, and some activities they might be doing in specific locations. |
| Animal | Activity | Place |
| Giraffe | Driving a car | At the beach |
| Lion | Putting on sunscreen | At the park |
| Cat | Jumping onto a laptop | In a home office |
| Dog | Chasing a squirrel | In the front yard |

 Answer the user query: Which animal is responsible for the typos?
Answer: A clever query!

Based on the provided Table 1, I'd make an educated inference to answer your question. Since the activities listed are quite unconventional for the respective animals (e.g., a giraffe driving a car, a lion putting on sunscreen), it's likely that the table is using humor or hypothetical scenarios.

Given this context, the question "Which animal is responsible for the typos?" is probably a tongue-in-cheek inquiry, as there's no direct information in the table about typos or typing activities.

However, if we were to make a playful connection, we could look for an animal that's:

1. Typically found in a setting where typing might occur (e.g., an office).
2. Engaging in an activity that could potentially lead to typos (e.g., interacting with a typing device).

Based on these loose criteria, I'd jokingly point to:

**Cat** as the potential culprit, since it's:
        * Located "In a home office"
        * Engaged in "Jumping onto a laptop", which could theoretically lead to accidental keystrokes or typos if the cat were to start "walking" on the keyboard!

Please keep in mind that this response is purely humorous and interpretative, as the table doesn't explicitly mention typos or provide a straightforward answer to the question.
```



## Logging Configuration

The NeMo Retriever Library uses [Ray](https://docs.ray.io/en/latest/index.html) for logging. 
For details, refer to [Configure Ray Logging](ray-logging.md).

By default, library mode runs in quiet mode to minimize startup noise. 
Quiet mode automatically configures the following environment variables.

| Variable                             | Quiet Mode Value | Description |
|--------------------------------------|------------------|-------------|
| `INGEST_RAY_LOG_LEVEL`               | `PRODUCTION`     | Sets Ray logging to ERROR level to reduce noise. |
| `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO` | `0`              | Silences Ray accelerator warnings |
| `OTEL_SDK_DISABLED`                  | `true`           | Disables OpenTelemetry trace export errors |


If you want to see detailed startup logs for debugging, use one of the following options:

- Set `quiet=False` when you run the pipeline as shown following.

    ```python
    run_pipeline(block=False, disable_dynamic_scaling=True, run_in_subprocess=True, quiet=False)
    ```

- Set the environment variables manually before you run the pipeline as shown following.

    ```bash
    export INGEST_RAY_LOG_LEVEL=DEVELOPMENT  # or DEBUG for maximum verbosity
    ```



## Library Mode Communication and Advanced Examples

Communication in library mode is handled through a simplified, 3-way handshake message broker called `SimpleBroker`.

Attempting to run a library-mode process co-located with a Docker Compose deployment does not work by default. 
The Docker Compose deployment typically creates a firewall rule or port mapping that captures traffic to port `7671`,
which prevents the `SimpleBroker` from receiving messages. 
Always ensure that you use library mode in isolation, without an active containerized deployment listening on the same port.


### Example `launch_libmode_service.py`

This example launches the pipeline service in a subprocess, 
and keeps it running until it is interrupted (for example, by pressing `Ctrl+C`). 
It listens for ingestion requests on port `7671` from an external client.

```python
import logging
import os

from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest_api.util.logging.configuration import configure_logging as configure_local_logging

# Configure the logger
logger = logging.getLogger(__name__)

local_log_level = os.getenv("INGEST_LOG_LEVEL", "DEFAULT")
if local_log_level in ("DEFAULT",):
    local_log_level = "INFO"

configure_local_logging(local_log_level)


def main():
    """
    Launch the libmode pipeline service using the embedded default configuration.
    """
    try:
        # Start pipeline and block until interrupted
        # Note: stdout/stderr cannot be passed when run_in_subprocess=True (not picklable)
        # Use quiet=False to see verbose startup logs
        _ = run_pipeline(
            block=True,
            disable_dynamic_scaling=True,
            run_in_subprocess=True,
        )
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
```

### Example `launch_libmode_and_run_ingestor.py`

This example starts the pipeline service in-process, 
and immediately runs an ingestion client against it in the same parent process.

```python
import logging
import os
import time

from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest_api.util.logging.configuration import configure_logging as configure_local_logging
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.client.interface import Ingestor
from nv_ingest_client.client import NvIngestClient

# Configure the logger
logger = logging.getLogger(__name__)

local_log_level = os.getenv("INGEST_LOG_LEVEL", "INFO")
if local_log_level in ("DEFAULT",):
    local_log_level = "INFO"

configure_local_logging(local_log_level)


def run_ingestor():
    """
    Set up and run the ingestion process to send traffic against the pipeline.
    """
    logger.info("Setting up Ingestor client...")
    client = NvIngestClient(
        message_client_allocator=SimpleClient, message_client_port=7671, message_client_hostname="localhost"
    )

    ingestor = (
        Ingestor(client=client)
        .files("./data/multimodal_test.pdf")
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=True,
            table_output_format="markdown",
            extract_infographics=False,
            text_depth="page",
        )
        .split(chunk_size=1024, chunk_overlap=150)
        .embed()
    )

    try:
        results, _ = ingestor.ingest(show_progress=False, return_failures=True)
        logger.info("Ingestion completed successfully.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise

    print("\nIngest done.")
    print(f"Got {len(results)} results.")


def main():
    """
    Launch the libmode pipeline service and run the ingestor against it.
    Uses the embedded default libmode pipeline configuration.
    """
    pipeline = None
    try:
        # Start pipeline in subprocess
        # Note: stdout/stderr cannot be passed when run_in_subprocess=True (not picklable)
        # Use quiet=False to see verbose startup logs
        pipeline = run_pipeline(
            block=False,
            disable_dynamic_scaling=True,
            run_in_subprocess=True,
        )
        time.sleep(10)
        run_ingestor()
        # Run other code...
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
    finally:
        if pipeline:
            pipeline.stop()
            logger.info("Shutting down pipeline...")


if __name__ == "__main__":
    main()
```



## The `run_pipeline` Function Reference

The `run_pipeline` function is the main entry point to start the NeMo Retriever Library pipeline. 
It can run in-process or as a subprocess.

The `run_pipeline` function accepts the following parameters.

| Parameter                | Type                   | Default | Required? | Description                                     |
|--------------------------|------------------------|---------|-----------|-------------------------------------------------|
| pipeline_config            | PipelineConfigSchema | —       | No        | A configuration object that specifies how the pipeline should be constructed. Default `None`; auto-loaded when `libmode=True`. |
| run_in_subprocess        | bool                   | False   | No        | `True` to launch the pipeline in a separate Python subprocess. `False` to run in the current process. |
| block                    | bool                   | True    | No        | `True` to run the pipeline synchronously. The function returns after it finishes. `False` to return an interface for external pipeline control. |
| disable_dynamic_scaling  | bool                   | None    | No        | `True` to disable autoscaling regardless of global settings. `None` to use the global default behavior. |
| dynamic_memory_threshold | float                  | None    | No        | A value between `0.0` and `1.0`. If dynamic scaling is enabled, triggers autoscaling when memory usage crosses this threshold. |
| stdout                   | TextIO                 | None    | No        | Redirect the subprocess `stdout` to a file or stream. If `None`, defaults to `/dev/null`. |
| stderr                   | TextIO                 | None    | No        | Redirect subprocess `stderr` to a file or stream. If `None`, defaults to `/dev/null`. |
| libmode                  | bool                   | True    | No        | `True` to load the default library mode pipeline configuration when `ingest_config` is `None`. |
| quiet                    | bool                   | None    | No        | `True` to suppress verbose startup logs (PRODUCTION preset). `None` defaults to `True` when `libmode=True`. Set to `False` for verbose output. |


The `run_pipeline` function returns the following values, depending on the parameters that you set:

- **run_in_subprocess=False and block=True**  — The function returns a `float` that represents the elapsed time in seconds.
- **run_in_subprocess=False and block=False** — The function returns a `RayPipelineInterface` object.
- **run_in_subprocess=True  and block=True**  — The function returns `0.0`.
- **run_in_subprocess=True  and block=False** — The function returns a `RayPipelineSubprocessInterface` object.


The following table matches the function signature in source (defaults and optionality). **None of these parameters are required** in the sense of having no default; omit them to use the defaults shown.

| Parameter | Required | Type (default) | Description |
|-----------|----------|----------------|-------------|
| `pipeline_config` | No | `Optional[PipelineConfigSchema]` (`None`) | Validated pipeline configuration. If `None` and `libmode=True`, the default library-mode pipeline is loaded automatically. If `None` and `libmode=False`, a `ValueError` is raised—you must pass a configuration. |
| `block` | No | `bool` (`True`) | If `True`, the call blocks until the pipeline finishes. If `False`, returns immediately with a handle object (see [Return type](#return-type)). |
| `disable_dynamic_scaling` | No | `Optional[bool]` (`None`) | If set, overrides the same field from the pipeline configuration. |
| `dynamic_memory_threshold` | No | `Optional[float]` (`None`) | If set, overrides the same field from the pipeline configuration. |
| `run_in_subprocess` | No | `bool` (`False`) | If `True`, runs the pipeline in a separate Python subprocess (`multiprocessing.Process`). If `False`, runs in the current process. |
| `stdout` | No | `Optional[TextIO]` (`None`) | When using a subprocess, optional stream for child stdout; if `None`, stdout is discarded. |
| `stderr` | No | `Optional[TextIO]` (`None`) | When using a subprocess, optional stream for child stderr; if `None`, stderr is discarded. |
| `libmode` | No | `bool` (`True`) | If `True` and `pipeline_config` is `None`, loads the default library-mode pipeline. If `False`, `pipeline_config` must be provided. |
| `quiet` | No | `Optional[bool]` (`None`) | If `True`, reduces logging noise for library use. If `None`, defaults to `True` when `libmode=True`. |

### Return type

`run_pipeline` returns a **union** of three possible types, depending on `block` and `run_in_subprocess`:

| Mode | Return type | Notes |
|------|-------------|--------|
| In-process, `block=True` | `float` | Elapsed time in seconds. |
| In-process, `block=False` | `RayPipelineInterface` | Handle to control the in-process pipeline (defined in `nv_ingest.framework.orchestration.ray.primitives.ray_pipeline`). |
| Subprocess, `block=False` | `RayPipelineSubprocessInterface` | Handle to control the subprocess-based pipeline (same module). **This is not** `RayPipelineInterface`; the two classes are separate implementations of `PipelineInterface`. Use `isinstance(..., RayPipelineSubprocessInterface)` when you launch with `run_in_subprocess=True` and `block=False`. |
| Subprocess, `block=True` | `float` | Returns `0.0` when blocking in subprocess mode. |

For the authoritative contract (including raised exceptions), refer to the docstring on `run_pipeline` in `src/nv_ingest/framework/orchestration/ray/util/pipeline/pipeline_runners.py`.
