# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example using MultiTypeExtractOperator in a graph pipeline.

This demonstrates processing a mixed folder of files with different types.
"""

from nemo_retriever.utils.pipeline import MultiTypeExtractOperator


def main():
    # Create the multi-type extract operator
    extract_op = MultiTypeExtractOperator(
        extract_params={
            "extract_text": True,
            "extract_tables": True,
            "extract_charts": True,
            "max_tokens": 512,  # For text/HTML
        }
    )

    # For demo, assume a folder with mixed files
    # In practice, provide a real folder path
    folder_path = "/path/to/mixed/files"  # Replace with actual path

    # Create graph: Extract -> (next operator, e.g., embed)
    graph = extract_op  # Since it's the root

    # Note: In a real graph, you'd chain more operators, but execute() here would run the extract
    # For demo, just show the structure
    print(f"Graph: {graph}")

    # To run: results = graph.execute(folder_path)
    # But since ingest() returns Ray dataset, and execute returns list, adjust as needed

    print("MultiTypeExtractOperator created. To run: graph.execute('/path/to/folder')")
    graph.execute(folder_path)  # This will run the extraction on the folder


if __name__ == "__main__":
    main()
