# Authentication and API keys

NeMo Retriever uses different credentials depending on what you are doing:

- **`NVIDIA_API_KEY`** — Authorizes HTTP calls to [NVIDIA-hosted NIMs](https://build.nvidia.com/) (for example `ai.api.nvidia.com` and `integrate.api.nvidia.com`). Obtain this key from [build.nvidia.com](https://build.nvidia.com/). Keys typically start with `nvapi-`.
- **NGC personal key** — Used when you install the [NeMo Retriever Helm chart](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md) so the cluster can authenticate to NGC Helm repos, pull images from `nvcr.io`, and provide `NGC_API_KEY` to in-cluster NIM workloads.

You may need one or both, for example if you deploy with Helm from NGC and also call hosted inference APIs.

## NVIDIA API key (`NVIDIA_API_KEY`) { #nvidia-api-key }

Use this key when you run the NeMo Retriever Library from Python, call remote NIM URLs, or use any workflow that calls NVIDIA-hosted inference without supplying a separate per-service secret.

1. Sign in at [build.nvidia.com](https://build.nvidia.com/) with your NVIDIA developer account.
2. Open [API keys](https://build.nvidia.com/settings/api-keys) (profile menu → **Settings** → **API keys**, or use that link after you are signed in).
3. Create a key, copy it when it is shown (you may not be able to read the full secret again later), and set it in your environment:

```bash
export NVIDIA_API_KEY="nvapi-..."
```

On Windows PowerShell you can use `$env:NVIDIA_API_KEY = "nvapi-..."`.

For a full list of related variables, see [Environment configuration variables](environment-config.md).

!!! note

    The `NVIDIA_API_KEY` from build.nvidia.com is not the same string as your NGC personal key used for Helm and `nvcr.io` access. Do not substitute one for the other unless your tooling explicitly documents that mapping.

## NGC personal key (Helm and `nvcr.io`)

Many public assets on NGC can be used without authentication. For a Kubernetes deployment, the cluster must still pull NIM and microservice images from `nvcr.io` and may need NGC API access; the Helm chart expects credentials derived from an NGC personal key.

To create a key, go to [https://org.ngc.nvidia.com/setup/api-keys](https://org.ngc.nvidia.com/setup/api-keys).

When you create an NGC key, select the following for **Services Included**.

- **NGC Catalog**
- **Public API Endpoints**

!!! important

    Early Access participants must also select **Private Registry**.

![Generate Personal Key](images/generate_personal_key.png)


## Using your NGC key with Helm

Configure your key through the chart values described in the [NeMo Retriever Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md): for example `imagePullSecret.create` / `imagePullSecret.password` for pulls from `nvcr.io`, `nimApiKey` (inline value or `existingSecret`) for the retriever service, and `nims.ngcApiKey` when `nims.enabled=true`. Exact paths are versioned—use the **Secrets** section in that README and [`values.yaml`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/values.yaml) as the source of truth.
