# Contributing to NeMo Retriever Library

External contributions will be welcome soon, and they are greatly appreciated. For repository policy, coding standards, and the contribution process, refer to **[Contributing to NeMo Retriever](https://github.com/NVIDIA/NeMo-Retriever/blob/main/CONTRIBUTING.md)** on GitHub.

The sections below describe how to configure your machine and Git remotes so you can work on documentation (or code) against **[NVIDIA/NeMo-Retriever](https://github.com/NVIDIA/NeMo-Retriever)** using a fork and a separate publishing clone.

---

## Set up your writing and development environment

### SSH authentication (one time for each computer)

1. **Create an SSH key** on your computer. Follow steps 1–3 in [Generating a new SSH key and adding it to the ssh-agent](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent). (You only need the key-generation steps; you can skip configuring ssh-agent if your organization prefers not to use it.)

2. **Add the public key to GitHub** using [Adding a new SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

### Developer Certificate of Origin sign-off

The repository requires a Developer Certificate of Origin (DCO) sign-off on every contributed commit. Sign-off is not the same as GPG commit signing. The `-s` or `--signoff` option adds a `Signed-off-by` trailer. The `-S` or `--gpg-sign` option attaches a cryptographic signature. A valid GPG signature does not add a DCO trailer.

For the full DCO policy, refer to [Developer Certificate of Origin (DCO)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/CONTRIBUTING.md#developer-certificate-of-origin-dco) in the repository contributing guide.

To sign off a commit, include `--signoff` (or `-s`) when you create the commit:

```bash
git commit --signoff -m "your message"
```

Git appends a trailer that matches your `user.name` and `user.email` configuration:

```text
Signed-off-by: Your Name <your@email.com>
```

Use `--signoff` on every commit that you contribute, including GPG-signed commits and commits that skip GPG signing.

### Commit signing for GitHub (one time for each computer)

GPG commit signing is optional unless your organization requires it. Configure it independently of DCO sign-off.

1. **Create a GPG key** following [Generating a new GPG key](https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key).

2. **Tell Git which key to use:**

    ```bash
    git config --global user.signingkey YOUR_KEY_ID
    ```

3. **Sign every commit by default (recommended if your organization requires signed commits):**

    ```bash
    git config --global commit.gpgsign true
    ```

4. **Optional: Sign a single commit with GPG, and include DCO sign-off:**

    ```bash
    git commit --signoff -S -m "your message"
    ```

    or

    ```bash
    git commit --signoff --gpg-sign -m "your message"
    ```

5. **Optional: Skip GPG signing for one commit. Still include DCO sign-off:**

    ```bash
    git commit --signoff --no-gpg-sign -m "Unsigned commit"
    ```

    The `--no-gpg-sign` option skips the GPG signature only. It does not skip the required DCO `--signoff`.

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

## Related

- [Contributing to NeMo Retriever](https://github.com/NVIDIA/NeMo-Retriever/blob/main/CONTRIBUTING.md) — authoritative contribution guidelines in the repository
