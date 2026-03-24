# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example demonstrating UserDefinedFunctionOperator in a graph pipeline.

This script shows how to create a custom UDF operator and chain it with
other operators in a graph, then execute the pipeline.
"""

from nemo_retriever.graph import AbstractOperator, UDFOperator


# Example UDF: a simple transformation function
def uppercase_and_prefix(text: str) -> str:
    """Convert text to uppercase and add a prefix."""
    return f"PROCESSED: {text.upper()}"


# Additional operators for the chain (must inherit from AbstractOperator)
class AddSuffixOperator(AbstractOperator):
    """Simple operator to add a suffix."""

    def __init__(self, suffix: str = "_end"):
        super().__init__()
        self.suffix = suffix

    def preprocess(self, data):
        return data

    def process(self, data):
        return str(data) + self.suffix

    def postprocess(self, data):
        return data


class MultiplyLengthOperator(AbstractOperator):
    """Operator that multiplies the length of the string by a factor."""

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def preprocess(self, data):
        return data

    def process(self, data):
        return len(str(data)) * self.factor

    def postprocess(self, data):
        return data


def main():
    # Create the UDF operator
    udf_op = UDFOperator(uppercase_and_prefix, name="UppercasePrefix")

    # Create other operators
    suffix_op = AddSuffixOperator("_final")
    length_op = MultiplyLengthOperator(3)

    # Build the graph: UDF >> Suffix >> Length multiplier
    graph = udf_op >> suffix_op >> length_op

    # Execute with sample data
    input_data = "hello world"
    results = graph.execute(input_data)

    print(f"Input: {input_data}")
    print(f"Output: {results}")

    # Expected: "PROCESSED: HELLO WORLD_final" -> length 28 -> 28 * 3 = 84
    assert results == [84], f"Expected [84], got {results}"
    print("Graph execution successful!")


if __name__ == "__main__":
    main()
