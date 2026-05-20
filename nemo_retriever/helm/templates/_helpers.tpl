{{/*
=============================================================================
Naming helpers
=============================================================================
*/}}

{{/*
nemo-retriever.name
  The chart name, optionally overridden by .Values.nameOverride.
*/}}
{{- define "nemo-retriever.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
nemo-retriever.fullname
  Default fully qualified app name.  Defaults to <release>-<chart> but
  collapses to just <release> when the release name already contains the
  chart name (idiomatic Helm pattern).
*/}}
{{- define "nemo-retriever.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
nemo-retriever.chart
  Standard Helm chart label value: <name>-<version>, sanitized.
*/}}
{{- define "nemo-retriever.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
nemo-retriever.serviceAccountName
  Name of the ServiceAccount to use for the service Deployment.
*/}}
{{- define "nemo-retriever.serviceAccountName" -}}
{{- if .Values.service.serviceAccount.create -}}
{{- default (include "nemo-retriever.fullname" .) .Values.service.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.service.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
=============================================================================
Label helpers
=============================================================================
*/}}

{{/*
nemo-retriever.labels
  Common labels applied to every object in the chart.
*/}}
{{- define "nemo-retriever.labels" -}}
helm.sh/chart: {{ include "nemo-retriever.chart" . }}
{{ include "nemo-retriever.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: nemo-retriever
{{- end -}}

{{/*
nemo-retriever.selectorLabels
  Selector labels for the service Deployment.  Stable across upgrades.
*/}}
{{- define "nemo-retriever.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nemo-retriever.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: service
{{- end -}}

{{/*
=============================================================================
PVC + Secret name helpers
=============================================================================
*/}}

{{- define "nemo-retriever.pvcName" -}}
{{- if .Values.persistence.existingClaim -}}
{{- .Values.persistence.existingClaim -}}
{{- else -}}
{{- printf "%s-data" (include "nemo-retriever.fullname" .) -}}
{{- end -}}
{{- end -}}

{{- define "nemo-retriever.retrieverResultsPvcName" -}}
{{- if .Values.retrieverResults.existingClaim -}}
{{- .Values.retrieverResults.existingClaim -}}
{{- else -}}
{{- printf "%s-retriever-results" (include "nemo-retriever.fullname" .) -}}
{{- end -}}
{{- end -}}

{{- define "nemo-retriever.configMapName" -}}
{{- printf "%s-config" (include "nemo-retriever.fullname" .) -}}
{{- end -}}

{{/*
=============================================================================
Pull secret helpers
=============================================================================

Combine the chart-managed NGC pull Secret (when ngcImagePullSecret.create=true)
with any pre-existing pull secrets listed in .Values.imagePullSecrets and
emit them in the form expected by a Pod spec.
*/}}
{{- define "nemo-retriever.imagePullSecrets" -}}
{{- $secrets := list -}}
{{- if .Values.ngcImagePullSecret.create -}}
{{- $secrets = append $secrets (dict "name" .Values.ngcImagePullSecret.name) -}}
{{- end -}}
{{- range .Values.imagePullSecrets -}}
{{- $secrets = append $secrets . -}}
{{- end -}}
{{- if $secrets -}}
imagePullSecrets:
{{- range $secrets }}
  - name: {{ .name }}
{{- end }}
{{- end -}}
{{- end -}}

{{/*
nemo-retriever.ngcImagePullSecret
  Base64-encoded docker-config JSON for the chart-managed NGC pull Secret.
  Honours the user-supplied `dockerconfigjson` (assumed already encoded)
  when present, otherwise assembles one from registry/username/password.
*/}}
{{- define "nemo-retriever.ngcImagePullSecret" -}}
{{- if .Values.ngcImagePullSecret.dockerconfigjson -}}
{{- .Values.ngcImagePullSecret.dockerconfigjson -}}
{{- else -}}
{{- $registry := required "ngcImagePullSecret.registry required when create=true and dockerconfigjson is empty" .Values.ngcImagePullSecret.registry -}}
{{- $username := required "ngcImagePullSecret.username required when create=true and dockerconfigjson is empty" .Values.ngcImagePullSecret.username -}}
{{- $password := required "ngcImagePullSecret.password required when create=true and dockerconfigjson is empty" .Values.ngcImagePullSecret.password -}}
{{- $auth := printf "%s:%s" $username $password | b64enc -}}
{{- $cfg := dict "auths" (dict $registry (dict "username" $username "password" $password "auth" $auth)) -}}
{{- $cfg | toJson | b64enc -}}
{{- end -}}
{{- end -}}

{{/*
=============================================================================
Split-topology helpers (gateway / realtime / batch)
=============================================================================
*/}}

{{/*
nemo-retriever.role.fullname
  Resource name for a topology role, e.g. <fullname>-gateway.
  Usage: {{ include "nemo-retriever.role.fullname" (dict "context" $ "role" "gateway") }}
*/}}
{{- define "nemo-retriever.role.fullname" -}}
{{- printf "%s-%s" (include "nemo-retriever.fullname" .context) .role | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
nemo-retriever.role.selectorLabels
  Stable selector labels for a topology-role Deployment / Service.
*/}}
{{- define "nemo-retriever.role.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nemo-retriever.name" .context }}
app.kubernetes.io/instance: {{ .context.Release.Name }}
app.kubernetes.io/component: {{ .role }}
{{- end -}}

{{/*
nemo-retriever.role.labels
  Full labels for a topology-role resource.
*/}}
{{- define "nemo-retriever.role.labels" -}}
helm.sh/chart: {{ include "nemo-retriever.chart" .context }}
{{ include "nemo-retriever.role.selectorLabels" . }}
{{- if .context.Chart.AppVersion }}
app.kubernetes.io/version: {{ .context.Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .context.Release.Service }}
app.kubernetes.io/part-of: nemo-retriever
{{- end -}}

{{/*
nemo-retriever.role.configMapName
  ConfigMap name for a topology role.
*/}}
{{- define "nemo-retriever.role.configMapName" -}}
{{- printf "%s-config" (include "nemo-retriever.role.fullname" .) -}}
{{- end -}}

{{/*
=============================================================================
NIM Operator endpoint resolution
=============================================================================

The NIM Operator creates a Kubernetes Service with the same name as the
NIMService resource. The chart hardcodes that name per-NIM (matching the
file name under templates/nims/<model>.yaml) so the retriever-service
config can address each NIM as `http://<service-name>:<port><invokePath>`.

Mapping (key -> Service name, default invokePath):
  page_elements   -> nemotron-page-elements-v3      /v1/infer
  table_structure -> nemotron-table-structure-v1    /v1/infer
  ocr             -> nemotron-ocr-v1                /v1/infer
  vlm_embed       -> llama-nemotron-embed-vl-1b-v2  /v1/embeddings
*/}}

{{/*
nemo-retriever.nimOperator.url
  In-cluster invocation URL for one operator-managed NIM. Returns the empty
  string when the NIM is disabled OR when the NIM Operator CRDs are absent,
  so callers can fall back to an externally configured URL.

  Usage:
    {{ include "nemo-retriever.nimOperator.url" (dict
         "context" $
         "key" "page_elements"
         "serviceName" "nemotron-page-elements-v3"
         "invokePath" "/v1/infer") }}
*/}}
{{- define "nemo-retriever.nimOperator.url" -}}
{{- $ctx := .context -}}
{{- $key := .key -}}
{{- $cfg := index $ctx.Values.nimOperator $key -}}
{{- if and (and (and $cfg $cfg.enabled) $ctx.Values.nims.enabled) ($ctx.Capabilities.APIVersions.Has "apps.nvidia.com/v1alpha1") -}}
{{- $port := 8000 -}}
{{- if and $cfg.expose $cfg.expose.service $cfg.expose.service.port -}}
{{- $port = int $cfg.expose.service.port -}}
{{- end -}}
{{- printf "http://%s:%d%s" .serviceName $port .invokePath -}}
{{- end -}}
{{- end -}}

{{/*
nemo-retriever.nim.endpointURL
  Resolves the URL the retriever-service should call for a given NIM.

  Resolution order:
    1. Explicit override in .Values.serviceConfig.nimEndpoints.<configKey>
       (always wins).
    2. The operator-managed in-cluster URL when both nimOperator.<key>.enabled
       and the apps.nvidia.com/v1alpha1 CRDs are present.
    3. Empty string (the service treats this as "no endpoint configured").

  Usage:
    {{ include "nemo-retriever.nim.endpointURL" (dict
         "context" $
         "key" "page_elements"
         "serviceName" "nemotron-page-elements-v3"
         "configKey" "pageElementsInvokeUrl"
         "invokePath" "/v1/infer") }}
*/}}
{{- define "nemo-retriever.nim.endpointURL" -}}
{{- $ctx := .context -}}
{{- $explicit := index $ctx.Values.serviceConfig.nimEndpoints .configKey -}}
{{- if $explicit -}}
{{- $explicit -}}
{{- else -}}
{{- include "nemo-retriever.nimOperator.url" (dict "context" $ctx "key" .key "serviceName" .serviceName "invokePath" .invokePath) -}}
{{- end -}}
{{- end -}}
