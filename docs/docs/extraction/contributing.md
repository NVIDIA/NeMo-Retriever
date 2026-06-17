# Contributing to NeMo Retriever Library

External contributions will be welcome soon, and they are greatly appreciated. For repository policy, coding standards, and the contribution process, refer to **[Contributing to NeMo Retriever](https://github.com/NVIDIA/NeMo-Retriever/blob/main/CONTRIBUTING.md)** on GitHub.

The sections below describe how to configure your machine and Git remotes so you can work on documentation (or code) against **[NVIDIA/NeMo-Retriever](https://github.com/NVIDIA/NeMo-Retriever)** using a fork and a separate publishing clone.

---

## Set up your writing and development environment

### SSH authentication (one time for each computer)

1. **Create an SSH key** on your computer. Follow steps 1–3 in [Generating a new SSH key and adding it to the ssh-agent](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent). (You only need the key-generation steps; you can skip configuring ssh-agent if your organization prefers not to use it.)

2. **Add the public key to GitHub** using [Adding a new SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

### Commit signing for GitHub (one time for each computer)

1. **Create a GPG key** following [Generating a new GPG key](https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key).

2. **Tell Git which key to use:**

    ```bash
    git config --global user.signingkey YOUR_KEY_ID
    ```

3. **Sign every commit by default (recommended if your org requires signed commits):**

    ```bash
    git config --global commit.gpgsign true
    ```

4. **Optional — sign a single commit:**

    ```bash
    git commit -S -m "your message"
    ```

    or

    ```bash
    git commit --gpg-sign -m "your message"
    ```

5. **Optional — skip signing for one commit:**

    ```bash
    git commit --no-gpg-sign -m "Unsigned commit"
    ```

---

## Set up your writing and development clone (fork)

You do day-to-day work in a clone of **your fork**, with `upstream` pointing at NVIDIA’s repo.

1. **Get access** to **[https://github.com/NVIDIA/NeMo-Retriever](https://github.com/NVIDIA/NeMo-Retriever)** (and permission to fork it, per your organization).

2. **Create a fork**

    - Open the **Fork** menu, then choose **Create a new fork**.
    - Accept the default repository name (`NeMo-Retriever`) unless your org requires another name.
    - **Deselect** “Copy the main branch only” if you need other branches locally; you can recover later with `git fetch upstream --tags` (see below).
    - Click **Create fork**.

3. **Clone the fork** onto your machine:

    - Pick a parent folder, for example `C:\_work\NeMo-Retriever-fork` or `C:\_repositories\NeMo-Retriever-fork`.
    - Open a terminal in that folder, then clone:

        ```bash
        git clone git@github.com:<your-github-username>/NeMo-Retriever.git
        ```

    - Enter the repository directory (default folder name is usually `NeMo-Retriever`):

        ```bash
        cd NeMo-Retriever
        ```

    - **Add NVIDIA’s repo as `upstream`:**

        ```bash
        git remote add upstream https://github.com/NVIDIA/NeMo-Retriever.git
        ```

    - If the fork was created with **only** the default branch, fetch the rest from upstream when needed:

        ```bash
        git fetch upstream --tags
        ```

Confirm remotes:

```bash
git remote -v
```

You should see `origin` pointing at your fork and `upstream` at `NVIDIA/NeMo-Retriever`.

---

## Set up your publishing clone (canonical repo)

Some workflows use a **second clone** of the **official** repository (not your fork) for publishing or internal automation.

1. Choose a **different** directory from your fork clone. On Windows, your team may require this clone inside **WSL**; follow internal guidance.

2. Clone NVIDIA’s repository:

    ```bash
    git clone git@github.com:NVIDIA/NeMo-Retriever.git
    ```

After setup you typically have **two** working copies: one from your fork (with `upstream` configured) and one straight from `NVIDIA/NeMo-Retriever`.

---

## Make a documentation change

### Target branches

Decide where the change lands:

- **`main` only**
- A **release** branch only (for example `release/25.9.0`)
- **Both** `main` and a release branch — commit to `main` first, then [cherry-pick](https://git-scm.com/docs/git-cherry-pick) the commits onto the release branch.

### Keep your fork and local clone in sync with NVIDIA

From your **fork** clone, on each branch you care about (example uses `main`; substitute `develop` or a release branch as needed):

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

Use a **space** between the remote name and the branch: `git push origin main`. (`git push origin/main` is invalid and Git will report an error.)

Repeat `checkout` / `fetch` / `merge` / `push` for every branch you maintain (`main`, `develop`, release branches, and so on).

---

## Build the documentation

NeMo Retriever Library documentation is published to **two different URLs**. Both use the same CI build ([NRL documentation — GitHub Pages](https://github.com/NVIDIA/NeMo-Retriever/actions/workflows/nrl-docs-github-pages.yml)); only the final deploy step differs.

Content merged to **`main`** is what the GitHub Pages workflow builds. That HTML is also the source for the public **26.05** release on **docs.nvidia.com** (`26.5.0/`). When a step needs a version number—for example the S3 prefix or a verification URL—use **`26.5.0`** (the docs.nvidia.com path for release **26.05**).

| Target | Live URL | How it goes live |
|--------|----------|------------------|
| **GitHub Pages** (staging / public preview) | [nvidia.github.io/NeMo-Retriever](https://nvidia.github.io/NeMo-Retriever/) | **Automatic** — workflow deploys after a successful run on `main` |
| **docs.nvidia.com** (official NVIDIA docs, 26.05 / `26.5.0`) | [docs.nvidia.com/nemo/retriever/26.5.0/](https://docs.nvidia.com/nemo/retriever/26.5.0/extraction/overview/) (also [latest/](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/)) | **Manual** — download the **`github-pages`** workflow artifact and upload HTML to S3 under `26.5.0/` |

Mike, the `docs-site` branch, and local `make docs` are **not** part of the current publish workflow.

### Run the documentation build (CI)

1. Merge your documentation changes to `main` on [NVIDIA/NeMo-Retriever](https://github.com/NVIDIA/NeMo-Retriever), **or** run the workflow manually:
    - Open [NRL documentation — GitHub Pages](https://github.com/NVIDIA/NeMo-Retriever/actions/workflows/nrl-docs-github-pages.yml)
    - Click **Run workflow** → choose branch **`main`** → **Run workflow**
2. Wait for the run to finish successfully (build + deploy jobs both green).

The workflow runs `mkdocs build -f mkdocs.yml --strict` and produces the static HTML site under `docs/site/`.

**If you are building for staging, QA, VDR, or peer review on GitHub Pages:** after the workflow completes, check [nvidia.github.io/NeMo-Retriever](https://nvidia.github.io/NeMo-Retriever/). **Stop here** — do not download the artifact or upload to S3.

### Local build for preview (optional)

Build locally when you want to review HTML before merging, or when CI is unavailable.

#### Get into the build environment

(Windows only) Open WSL (`wsl.exe`) and ensure you are in the correct WSL distro for this work.

```bash
cd NeMo-Retriever
```

(One time only) From the repository root:

```bash
python -m pip install --upgrade pip
pip install -r docs/requirements.txt
pip install -e ./nemo_retriever
```

#### Build and preview

From the `docs/` directory:

```bash
cd docs
DISABLE_MKDOCS_2_WARNING=true mkdocs build -f mkdocs.yml --strict
```

Output is written to `docs/site/`. To serve it locally:

```bash
mkdocs serve -f mkdocs.yml
```

Browse to the URL printed in the shell (typically `http://127.0.0.1:8000/`). Press **Ctrl+C** to stop the server.

You can also run `make nrl-github-pages` from `docs/` for the same MkDocs build.

**STOP** if you are building for staging only. Do not continue to [Production publish to docs.nvidia.com](#production-publish-to-docsnvidiacom).

### Production publish to docs.nvidia.com { #production-publish-to-docsnvidiacom }

Use this procedure when copying the CI-built HTML from the GitHub Pages workflow to **docs.nvidia.com** for release **26.05** (`26.5.0`). You download the **`github-pages`** artifact from a successful run on **`main`**, extract it, and upload the files to S3 under **`developer/docs/nemo/retriever/26.5.0/`**.

#### Download the built HTML from GitHub Actions

1. Open [NRL documentation — GitHub Pages](https://github.com/NVIDIA/NeMo-Retriever/actions/workflows/nrl-docs-github-pages.yml).
2. Click the **successful** run that contains the documentation you want to publish.
3. Scroll to **Artifacts** (or the deployment summary) and download **`github-pages`** (a `.zip` file).
4. Extract the download:
    - Unzip `github-pages.zip` — it contains **`artifact.tar`**.
    - Extract `artifact.tar` into a working folder (for example `nrl-docs-publish/`).

    On Windows (PowerShell), from the folder that contains the zip:

    ```powershell
    Expand-Archive -Path "github-pages.zip" -DestinationPath pages-artifact
    tar -xf pages-artifact/artifact.tar -C nrl-docs-publish
    ```

    On Linux or macOS:

    ```bash
    unzip github-pages.zip -d pages-artifact
    mkdir -p nrl-docs-publish
    tar -xf pages-artifact/artifact.tar -C nrl-docs-publish
    ```

5. Spot-check the extracted site:
    - `nrl-docs-publish/index.html` exists
    - `nrl-docs-publish/extraction/overview/index.html` opens in a browser

The extracted tree is the same flat MkDocs output that GitHub Pages serves — `index.html`, `extraction/`, `assets/`, and related paths at the top level. It should match what is already in S3 under **`26.5.0/`**: five top-level files plus folders `assets/`, `extraction/`, `license/`, `reference/`, and `search/`.

#### Copy the HTML files to S3

Publish documentation live on **docs.nvidia.com** by uploading the extracted site to the Brightspot S3 prefix for release **26.05** (`26.5.0`).

1. Go to [myapplications.microsoft.com](https://myapplications.microsoft.com/) and open **AWS NVIDIA Accounts**. Sign in with the account your team uses for documentation publishing.
2. Open the S3 bucket **`brightspot-assets-prod`**.
3. Navigate to **`developer/docs/nemo/retriever/26.5.0/`**. This prefix maps to [https://docs.nvidia.com/nemo/retriever/26.5.0/](https://docs.nvidia.com/nemo/retriever/26.5.0/).
4. **Delete** the existing objects under **`26.5.0/`** if you are replacing a prior upload for the same release.
5. Click **Upload** and add **the contents** of `nrl-docs-publish/` (not the `nrl-docs-publish` folder itself). S3 keys should mirror your local layout, for example:
    - `developer/docs/nemo/retriever/26.5.0/index.html`
    - `developer/docs/nemo/retriever/26.5.0/extraction/overview/index.html`
    - `developer/docs/nemo/retriever/26.5.0/assets/...`
6. Confirm in the S3 console that the **`26.5.0/`** prefix lists the same top-level files and folders as your local extract (10 objects at the root if you count five files and five folders, plus everything nested under the folders).
7. **`latest/`** aliases the current GA release. Repeat steps 4–6 under **`developer/docs/nemo/retriever/latest/`** with the same file set so [docs.nvidia.com/nemo/retriever/latest/](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/) serves the same **26.05** build (or follow your team's copy procedure for that alias).

#### Purge the Akamai cache

S3 updates are not visible on docs.nvidia.com until the CDN cache is purged.

1. Open [control.akamai.com](https://control.akamai.com/) (or your team's Akamai entry point).
2. Submit a cache purge for **`/nemo/retriever`** (or the path your docs team specifies).
3. Wait for the confirmation email (often **~40 minutes**). Do not announce the release as live until the purge completes.
4. Verify in a private browser window:
    - [docs.nvidia.com/nemo/retriever/26.5.0/extraction/overview/](https://docs.nvidia.com/nemo/retriever/26.5.0/extraction/overview/)
    - [docs.nvidia.com/nemo/retriever/latest/extraction/overview/](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/)

---

## Related

- [Contributing to NeMo Retriever](https://github.com/NVIDIA/NeMo-Retriever/blob/main/CONTRIBUTING.md) — authoritative contribution guidelines in the repository
