## Description: <br>
Use when the user wants to search, index, or answer questions over a folder of PDFs (or other documents) — including building a RAG / search index over PDFs, looking up information across many PDFs, or running the `retriever` CLI (ingest, query, pipeline, recall, eval, etc.). <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who need to search, index, or answer questions across PDF and document collections using RAG and vector search via the retriever CLI. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [NeMo Retriever Library Documentation](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/) <br>
- [Install Guide](references/install.md) <br>
- [Setup Guide](references/setup.md) <br>
- [Query Workflow](references/query.md) <br>
- [Pitfalls and Recovery](references/pitfalls.md) <br>
- [CLI: ingest](references/cli/ingest.md) <br>
- [CLI: query](references/cli/query.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, JSON] <br>
**Output Format:** [JSON] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Tasks: <br>
NVSkills-Eval 3-Tier evaluation (external profile); Tier 1 static validation (9 checks, 20 findings), Tier 2 deduplication (2 checks, 1 finding). Tier 3 live agent evaluation not available in this report. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>



## Skill Version(s): <br>
3fa00d94 (source: git SHA, committed 2026-05-28) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
