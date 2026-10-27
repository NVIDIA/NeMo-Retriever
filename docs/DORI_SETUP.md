# NVIDIA DORI Setup

Use this guide only when the user explicitly asks to install or configure NVIDIA DORI.
Routine documentation work follows [NVIDIA DORI Routing](../AGENTS.md#nvidia-dori-routing) and does not require setup.

Follow these rules for an explicit setup request:

1. Use your organization's maintained setup instructions through an approved private channel.
   Keep private installation and configuration details outside this public repository.
   Do not copy internal links, repository paths, package sources, or operational commands into tracked files or public PR text.
2. Confirm the user has the access required by those instructions before inspecting or installing private components.
   Access confirmation does not approve installation or host configuration.
3. Describe proposed installations or host configuration changes and obtain the user's approval before applying them.
   Do not create repository-scoped identity, authorization, or MCP configuration files without separate repository-owner authorization.
4. Use credentials the user already configured.
   Never request, print, copy, or commit credentials or credential-bearing URLs.
   If authentication fails, stop setup and refer the user to their organization's setup instructions.
5. Verify that the host exposes the DORI tools and an identifiable NVIDIA documentation Skill Library collection.
   Keep private collection metadata and validation output out of tracked files and public PR text.

If setup instructions, access, approval, or tools are unavailable, continue documentation work with the checked-in [Writing Style Guide](AGENTS.md#writing-style-guide).
