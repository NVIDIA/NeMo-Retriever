# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script for MultiTypeExtractOperator with sample data.
"""

from nemo_retriever.graph import MultiTypeExtractOperator


def main():
    # Use the nv-ingest data folder which has mixed file types
    data_folder = "/raid/nv-ingest/data"

    # Create the operator with basic extract params
    extract_op = MultiTypeExtractOperator(
        extract_params={
            "extract_text": True,
            "extract_tables": False,  # Keep it simple for testing
            "extract_charts": False,
            "extract_infographics": False,
            "max_tokens": 512,  # For text/HTML
        }
    )

    print(f"Testing MultiTypeExtractOperator with folder: {data_folder}")

    # Run the operator
    try:
        results = extract_op.run(data_folder)
        print(f"Extraction completed successfully! Got {len(results)} results.")
        if results:
            print("Sample result:")
            print(results[0])
        else:
            print("No results returned.")
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
