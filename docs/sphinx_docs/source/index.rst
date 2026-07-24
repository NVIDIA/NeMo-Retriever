===============
API Reference
===============

This page provides API references for the `nv-ingest-api`, `nv-ingest-client`, and `nv-ingest` modules. 

Basic Usage
---------------

The following will provide a basic example of submitting a job using the Python API

To begin, get the pipeline running in library mode with
:func:`~nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners.PipelineCreationSchema`,
which creates a config object for the pipeline and
:func:`~nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners.run_pipeline`, which
starts the pipline

.. code-block:: python

    from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
    from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import PipelineCreationSchema

    # Start the pipeline subprocess for library mode
    config = PipelineCreationSchema()

    run_pipeline(config, block=False, disable_dynamic_scaling=True, run_in_subprocess=True)

Then, connect the client to the pipeline with :func:`~nv_ingest_client.client.NvIngestClient`

.. code-block:: python

    from nv_ingest_client.client import NvIngestClient

    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost"
    )


Next, use the :class:`~nv_ingest_client.client.Ingestor` class to ingest the test PDF (click the
Ingestor link to see descriptions of the available tasks)

.. code-block:: python

    import time
    from nv_ingest_client.client import NvIngestClient

    # LanceDB: local database directory and table name for embedded vector storage
    lancedb_uri = "./lancedb_data"
    table_name = "test"

    # do content extraction from files                                
    ingestor = (
        Ingestor(client=client)
        .files("data/multimodal_test.pdf")
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=True,
            paddle_output_format="markdown",
            extract_infographics=True,
            # extract_method="nemotron_parse",  # Alternate PDF extraction method
            text_depth="page"
        ).embed()
        .vdb_upload(
            vdb_op="lancedb",
            uri=lancedb_uri,
            table_name=table_name,
            hybrid=False,
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

To inspect the results, use :func:`~nv_ingest_client.util.process_json_files.ingest_json_results_to_blob`

.. code-block:: python

    from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

    # results blob is directly inspectable
    print(ingest_json_results_to_blob(results[0]))

To query the ingested LanceDB table, use :meth:`~nemo_retriever.vdb.lancedb.LanceDB.retrieval`
(precomputed query vectors) or :meth:`~nemo_retriever.retriever.Retriever.query` (embeds the query string for you).

.. code-block:: python

    from nemo_retriever.vdb.lancedb import LanceDB
    from nemo_retriever.retriever import Retriever

    table_path = "./lancedb_data"
    table_name = "test"

    # Option 1: precomputed query vectors
    vdb = LanceDB(uri=table_path, table_name=table_name)
    query_vectors = [[0.0] * 2048]  # replace with vectors from the same embedder used at ingest
    retrieved_docs = vdb.retrieval(query_vectors, top_k=1)

    # Option 2: natural-language query (Retriever embeds the query for you)
    retriever = Retriever(
        vdb_kwargs={"uri": table_path, "table_name": table_name},
    )
    hits = retriever.query("Which animal is responsible for the typos?", top_k=1)


.. toctree::
    :maxdepth: 2
    :caption: NV-Ingest Packages

    nv-ingest-api/modules.rst
    nv-ingest-client/modules.rst
    nv-ingest/modules.rst
    nemo-retriever/modules.rst
    
