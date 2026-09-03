# Authentication and API keys

NeMo Retriever uses different credentials depending on what you are doing:

- **`NVIDIA_API_KEY`** — Authorizes HTTP calls to [NVIDIA-hosted NIMs](https://build.nvidia.com/) (for example `ai.api.nvidia.com` and `integrate.api.nvidia.com`). Obtain this key from [build.nvidia.com](https://build.nvidia.com/). Keys typically start with `nvapi-`.
- **NGC personal key** — Used when you install the [NeMo Retriever Helm chart](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.2/nemo_retriever/helm/README.md) so the cluster can authenticate to NGC Helm repos, pull images from `nvcr.io`, and provide `NGC_API_KEY` to in-cluster NIM workloads.

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

The SDK and CLI do not load a `.env` file automatically. For a full list of related variables and how to source a `.env` file into the shell, refer to [Environment configuration variables](environment-config.md).

For `LiteLLMClient` and `LLMJudge` with a `nvidia_nim/...` model, pass `api_key="os.environ/NVIDIA_API_KEY"`. LiteLLM does not read `NVIDIA_API_KEY` for that provider.

Hosted object-detection NIMs (Page Elements, Table Structure, Graphic Elements) cap inline base64 image payloads at about 180,000 characters (roughly 180 KB). This key authorizes those hosted inference calls. For size limits and what to do when an image exceeds the cap, refer to [Hosted Page Elements NIM image size limits](troubleshoot.md#hosted-page-elements-nim-image-size-limits).

!!! note

    The `NVIDIA_API_KEY` from build.nvidia.com is not the same string as your NGC personal key used for Helm and `nvcr.io` access. Do not substitute one for the other unless your tooling explicitly documents that mapping.

## Credential references in persisted graphs { #credential-references-in-persisted-graphs }

Persisted pipeline graphs never contain literal API keys. Configure a graph with an explicit worker-side environment reference such as:

```python
api_key="os.environ/NVIDIA_API_KEY"
```

Use the provider's own variable name, for example `os.environ/OPENAI_API_KEY` for an OpenAI model. The reference is stored in graph JSON and resolved only when the operator is constructed or invoked on the worker.

For LiteLLM `nvidia_nim/...` models, use `os.environ/NVIDIA_API_KEY`. The worker resolves that reference and passes the value as `api_key`. If you omit `api_key`, LiteLLM looks up `NVIDIA_NIM_API_KEY` instead.

Literal keys remain available for non-persisted local execution, but attempting to serialize one raises an error. This prevents graph persistence from silently substituting an NVIDIA credential for another provider's key.

For how persisted graphs store credential references, refer to [Persisted graphs are trusted configuration](nemo-retriever-api-reference.md#persisted-graphs-are-trusted-configuration) in the Python API guide.

## NGC personal key (Helm and `nvcr.io`) { #ngc-personal-key }

Many public assets on NGC can be used without authentication. For a Kubernetes deployment, the cluster must still pull NIM and microservice images from `nvcr.io` and may need NGC API access; the Helm chart expects credentials derived from an NGC personal key.

To create a key, go to [https://org.ngc.nvidia.com/setup/api-keys](https://org.ngc.nvidia.com/setup/api-keys).

When you create an NGC key, select the following for **Services Included**.

- **NGC Catalog**
- **Public API Endpoints**

!!! important

    Early Access participants must also select **Private Registry**.

![Generate Personal Key](images/generate_personal_key.png)

After you copy the key, set it in your environment. The Helm example below reads `$NGC_API_KEY`. If that variable is empty, Helm fails because `ngcImagePullSecret.password` is required when `create=true`.

```bash
export NGC_API_KEY="<ngc-personal-key>"
```

On Windows PowerShell you can use `$env:NGC_API_KEY = "<ngc-personal-key>"`.

## Using your NGC key with Helm { #using-your-ngc-key-with-helm }

Set the chart values in the [Secrets](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.2/nemo_retriever/helm/README.md#secrets) section of the Helm chart README so the chart renders `ngc-secret` and `ngc-api`:

- `ngcImagePullSecret.create` and `ngcImagePullSecret.password` create the `ngc-secret` dockerconfigjson Secret for pulls from `nvcr.io`.
- `ngcApiSecret.create` and `ngcApiSecret.password` create the `ngc-api` Secret with `NGC_API_KEY` and `NGC_CLI_API_KEY`. The service container maps `NGC_API_KEY` and `NVIDIA_API_KEY` from the Secret `NGC_API_KEY` key when the Secret exists.
- Overriding `ngcImagePullSecret.name` or `ngcApiSecret.name` also updates every rendered NIMCache and NIMService unless you set a non-empty per-NIM `image.pullSecrets` or `authSecret` override.

```bash
helm install retriever ./nemo_retriever/helm \
  --set ngcImagePullSecret.create=true \
  --set ngcImagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.create=true \
  --set ngcApiSecret.password=$NGC_API_KEY
```

Helm accepts unknown `--set` paths without error. Paths such as `imagePullSecret`, `nimApiKey`, and `nims.ngcApiKey` do not create either Secret.

For defaults and additional fields, refer to [`values.yaml`](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.2/nemo_retriever/helm/values.yaml).
