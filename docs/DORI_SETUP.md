# NVIDIA DORI Setup

DORI installs and routes agent skill collections.
The NVIDIA documentation Skill Library provides documentation-focused skills that DORI can load.

Use this guide only when the user explicitly asks to install or configure NVIDIA DORI.
Routine documentation work follows [NVIDIA DORI Routing](../AGENTS.md#nvidia-dori-routing) and does not require setup.
Before inspecting or installing private components, confirm that the user has the required access.
If access is unavailable, stop setup and use the checked-in [Writing Style Guide](AGENTS.md#writing-style-guide).
Access confirmation does not approve installation or host configuration.

Use your organization's maintained setup instructions through an approved private channel for installation sources, versions, and canonical collection source identities.
Keep concrete internal hostnames, repository paths, package sources, and deployment-specific instructions out of tracked files and public PR text.

## Verify the Skill Library Source

Compare the installed collection's source identity with the canonical Skill Library identity supplied by trusted host configuration or your organization's maintained setup instructions.
A matching display name, description, or source substring is insufficient.
Use the canonical identity for both verification and installation.
If the canonical identity is unavailable or the installed source cannot be verified, use the [Writing Style Guide](AGENTS.md#writing-style-guide).
Keep concrete private source identities and collection metadata out of tracked files and public PR text.

## Inspect the Environment

1. Check for DORI MCP tools.
   - If the current agent exposes `dori_handle` or `dori_route`, do not reconfigure the host.
   - When `dori_collections` is available, verify the installed collection using [Verify the Skill Library Source](#verify-the-skill-library-source).
   - If the Skill Library is missing, identify it as the only missing component and continue to [Confirm Changes](#confirm-changes).
2. When DORI MCP tools are unavailable, inspect the command-line interface (CLI).
   - Run `command -v dori` (or the host equivalent).
   - If the CLI exists, run `dori collections list --json`.
   - Verify the installed collection using [Verify the Skill Library Source](#verify-the-skill-library-source).
3. Identify the host from explicit runtime context.
   Do not infer the host from the model name or repository files.
4. Run `dori setup auto --dry-run` as a cross-check when the CLI exists.
   - If auto-detection conflicts with the explicit host, use the explicit host.
   - If no explicit host exists and auto-detection is uncertain, ask which host is running.

Use the following host commands:

| Explicit Host | Setup Command |
|---|---|
| Codex CLI or Desktop | `dori setup codex` |
| Cursor | `dori setup cursor --scope user` |
| Claude Code | `dori setup claude-code --scope user` |
| Claude Desktop | `dori setup claude` |
| VS Code with GitHub Copilot | `dori setup vscode --scope user` |
| Kiro | `dori setup kiro` |
| Google Antigravity | `dori setup antigravity` |

Keep the listed `--scope user` option.
Project or combined scope requires separate repository-owner authorization because it can create a repository MCP configuration file.

## Confirm Changes

Report each missing component.
Before an installation or host configuration change, ask:

> DORI setup is incomplete: `<missing components>`.
> Do you want me to install or configure these components in your user environment?

Continue only after explicit approval.
The user's private-source access confirmation does not approve these changes.
If the user declines, use the [Writing Style Guide](AGENTS.md#writing-style-guide).

## Install Missing Components

When DORI MCP is unavailable and `dori` is missing, require an existing `uv` command.

- If `uv` is missing, stop and refer the user to your organization's maintained setup instructions.
  Do not download or execute an installer script.
- If `uv` exists, use the pinned DORI version and package source specified in those instructions for your host.

Obtain the approved Skill Library source through the same trusted configuration or setup instructions used for verification.
In the following examples, replace `<approved Skill Library source>` with that source without committing the concrete value.

When DORI MCP is unavailable and the Skill Library is missing, run:

```bash
dori install "<approved Skill Library source>" --all --yes
```

When DORI MCP is available but the Skill Library is missing:

1. Run `dori_collections(action="install", source="<approved Skill Library source>")`.
2. Run `dori_refresh`.
3. Verify the source with `dori_collections(action="list")` using [Verify the Skill Library Source](#verify-the-skill-library-source).

Do not depend on a shell-visible CLI or reconfigure the host on the DORI MCP path.

## Configure and Validate the Host

Complete this section only when DORI MCP is unavailable.
After the CLI becomes available, run `dori setup auto --dry-run` if it did not run during inspection.
If auto-detection conflicts with the explicit host, use the explicit host.
If no explicit host exists and auto-detection is uncertain, ask which host is running.
After approval, run the setup command for the resolved host.
Then perform the following checks:

1. Run the selected command with `--validate`.
2. Run `dori doctor health --json`.
3. Require a passing host validation and `"ok": true` health.

Follow the activation action that DORI reports.
The action can require an application restart, a new session, a window reload, or enabling the MCP server.

Until the current agent exposes DORI tools, continue the original task with the [Writing Style Guide](AGENTS.md#writing-style-guide).

## Protect Credentials and Repository State

- Never search for, request, print, copy, export, or embed a token, password, cookie, SSH key, or credential-bearing URL.
- Let `uv`, Git, and DORI use credentials that the user already configured.
  If access is denied or authentication is missing, stop and refer to your organization's maintained setup instructions.
- Do not create repository-scoped identity or authorization files.
  Confirm private-source access only for an explicit setup request.
- Do not bypass approval controls for writes outside the repository.
- Do not create or commit project-scoped DORI state or MCP configuration without separate repository-owner authorization.
- Do not retry a failed installation in the same task.
