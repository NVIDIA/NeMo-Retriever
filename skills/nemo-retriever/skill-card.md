## Description: <br>
Use when the user wants to search, index, or answer questions over a folder of PDFs (or other documents) — including building a RAG / search index over PDFs, looking up information across many PDFs, or running the `retriever` CLI (ingest, query, pipeline, recall, eval, etc.). <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who need to search, index, or answer questions over collections of PDFs and documents using a local RAG/vector-search pipeline powered by the retriever CLI. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [NeMo Retriever Library Documentation](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/) <br>
- [CLI reference: retriever ingest](references/cli/ingest.md) <br>
- [CLI reference: retriever query](references/cli/query.md) <br>
- [Installation guide](references/install.md) <br>
- [Query workflow](references/query.md) <br>
- [Setup guide](references/setup.md) <br>
- [Pitfalls and recovery](references/pitfalls.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, JSON, Synthesized answers] <br>
**Output Format:** [Markdown with inline bash code blocks and JSON query results] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Query results are JSON arrays sorted by vector distance; final answers are synthesized from retrieved context] <br>

## Evaluation Tasks: <br>
Evaluated through NVSkills-Eval 3-Tier framework (profile: external). Tier 1: 9 static validation checks (21 findings, passed with observations). Tier 2: 2 deduplication checks (0 findings, passed). Overall verdict: PASS. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>



## Skill Version(s): <br>
25.3.0-1014-gb7fdbb45 (source: git describe) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
