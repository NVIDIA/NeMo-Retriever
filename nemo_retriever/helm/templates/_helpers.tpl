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

Combine the chart-managed NGC pull Secret with any pre-existing pull secrets
listed in .Values.imagePullSecrets and emit them in the form expected by a
Pod spec.  The NGC secret is injected when the chart creates it
(ngcImagePullSecret.create=true) OR when the user pre-created it and
supplied the name (ngcImagePullSecret.create=false + name set).
*/}}
{{- define "nemo-retriever.imagePullSecrets" -}}
{{- $secrets := list -}}
{{- if or .Values.ngcImagePullSecret.create .Values.ngcImagePullSecret.name -}}
{{- $secrets = append $secrets (dict "name" (default "ngc-secret" .Values.ngcImagePullSecret.name)) -}}
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
Tracing helpers
=============================================================================
*/}}

{{- define "nemo-retriever.suffixedFullname" -}}
{{- $base := include "nemo-retriever.fullname" .context -}}
{{- $suffix := .suffix -}}
{{- $maxBaseLen := int (sub 63 (len $suffix)) -}}
{{- printf "%s%s" ($base | trunc $maxBaseLen | trimSuffix "-") $suffix | trimSuffix "-" -}}
{{- end -}}

{{- define "nemo-retriever.zipkin.fullname" -}}
{{- include "nemo-retriever.suffixedFullname" (dict "context" . "suffix" "-zipkin") -}}
{{- end -}}

{{- define "nemo-retriever.otel.fullname" -}}
{{- include "nemo-retriever.suffixedFullname" (dict "context" . "suffix" "-otel") -}}
{{- end -}}

{{- define "nemo-retriever.otel.configMapName" -}}
{{- include "nemo-retriever.suffixedFullname" (dict "context" . "suffix" "-otel-config") -}}
{{- end -}}

{{- define "nemo-retriever.otel.ports" -}}
{{- $otel := .Values.topology.otel | default dict -}}
{{- if not (kindIs "map" $otel) -}}
{{- fail "topology.otel must be a map" -}}
{{- end -}}
{{- $ports := get $otel "ports" -}}
{{- if not (kindIs "map" $ports) -}}
{{- fail "topology.otel.ports must be a map when topology.otel.enabled=true" -}}
{{- end -}}
{{- range $portName := list "otlpGrpc" "otlpHttp" "prometheus" -}}
{{- if not (get $ports $portName) -}}
{{- fail (printf "topology.otel.ports.%s is required when topology.otel.enabled=true" $portName) -}}
{{- end -}}
{{- end -}}
{{- toYaml $ports -}}
{{- end -}}

{{- define "nemo-retriever.zipkin.image" -}}
{{- $zipkin := .Values.topology.zipkin | default dict -}}
{{- if not (kindIs "map" $zipkin) -}}
{{- fail "topology.zipkin must be a map" -}}
{{- end -}}
{{- $image := get $zipkin "image" -}}
{{- if not (kindIs "map" $image) -}}
{{- fail "topology.zipkin.image must be a map when topology.zipkin.enabled=true" -}}
{{- end -}}
{{- range $fieldName := list "repository" "tag" "pullPolicy" -}}
{{- if not (get $image $fieldName) -}}
{{- fail (printf "topology.zipkin.image.%s is required when topology.zipkin.enabled=true" $fieldName) -}}
{{- end -}}
{{- end -}}
{{- toYaml $image -}}
{{- end -}}

{{- define "nemo-retriever.zipkin.port" -}}
{{- $zipkin := .Values.topology.zipkin | default dict -}}
{{- if not (kindIs "map" $zipkin) -}}
{{- fail "topology.zipkin must be a map" -}}
{{- end -}}
{{- $port := get $zipkin "port" -}}
{{- if not $port -}}
{{- fail "topology.zipkin.port is required when topology.zipkin.enabled=true" -}}
{{- end -}}
{{- $port -}}
{{- end -}}


{{- define "nemo-retriever.zipkin.endpoint" -}}
{{- $zipkin := .Values.topology.zipkin | default dict -}}
{{- if not (kindIs "map" $zipkin) -}}
{{- fail "topology.zipkin must be a map" -}}
{{- end -}}
{{- $exporter := get $zipkin "exporter" | default dict -}}
{{- if not (kindIs "map" $exporter) -}}
{{- fail "topology.zipkin.exporter must be a map" -}}
{{- end -}}
{{- $endpoint := get $exporter "endpoint" -}}
{{- if $endpoint -}}
{{- tpl $endpoint . -}}
{{- else -}}
{{- printf "http://%s:%v/api/v2/spans" (include "nemo-retriever.zipkin.fullname" .) (include "nemo-retriever.zipkin.port" .) -}}
{{- end -}}
{{- end -}}

{{- define "nemo-retriever.otel.config" -}}
{{- $otel := .Values.topology.otel | default dict -}}
{{- if not (kindIs "map" $otel) -}}
{{- fail "topology.otel must be a map" -}}
{{- end -}}
{{- $configValue := get $otel "config" -}}
{{- if not (kindIs "map" $configValue) -}}
{{- fail "topology.otel.config must be a map when topology.otel.enabled=true" -}}
{{- end -}}
{{- $config := deepCopy $configValue -}}
{{- $zipkin := .Values.topology.zipkin | default dict -}}
{{- if not (kindIs "map" $zipkin) -}}
{{- fail "topology.zipkin must be a map" -}}
{{- end -}}
{{- $zipkinExporterConfig := get $zipkin "exporter" | default dict -}}
{{- if not (kindIs "map" $zipkinExporterConfig) -}}
{{- fail "topology.zipkin.exporter must be a map" -}}
{{- end -}}
{{- $zipkinInjectionEnabled := and (get $zipkinExporterConfig "enabled") (or (get $zipkin "enabled") (get $zipkinExporterConfig "endpoint")) -}}
{{- if $zipkinInjectionEnabled -}}
{{- $service := get $config "service" | default dict -}}
{{- $pipelines := get $service "pipelines" | default dict -}}
{{- $traces := get $pipelines "traces" -}}
{{- if not $traces -}}
{{- fail "topology.zipkin.exporter.enabled requires topology.otel.config.service.pipelines.traces with non-empty receivers; provide that traces pipeline or set topology.zipkin.exporter.enabled=false" -}}
{{- end -}}
{{- $traceReceivers := get $traces "receivers" -}}
{{- if not $traceReceivers -}}
{{- fail "topology.zipkin.exporter.enabled requires topology.otel.config.service.pipelines.traces with non-empty receivers; provide that traces pipeline or set topology.zipkin.exporter.enabled=false" -}}
{{- end -}}
{{- $receivers := get $config "receivers" | default dict -}}
{{- range $receiverName := $traceReceivers -}}
{{- if or (not (hasKey $receivers $receiverName)) (eq (get $receivers $receiverName) nil) -}}
{{- fail (printf "topology.otel.config.service.pipelines.traces trace receiver %q is missing or null in topology.otel.config.receivers; fix topology.otel.config or set topology.zipkin.exporter.enabled=false" $receiverName) -}}
{{- end -}}
{{- end -}}
{{- $processors := get $config "processors" | default dict -}}
{{- $traceProcessors := get $traces "processors" | default list -}}
{{- range $processorName := $traceProcessors -}}
{{- if or (not (hasKey $processors $processorName)) (eq (get $processors $processorName) nil) -}}
{{- fail (printf "topology.otel.config.service.pipelines.traces trace processor %q is missing or null in topology.otel.config.processors; fix topology.otel.config or set topology.zipkin.exporter.enabled=false" $processorName) -}}
{{- end -}}
{{- end -}}
{{- $exporters := get $config "exporters" | default dict -}}
{{- $zipkinExporter := get $exporters "zipkin" | default dict -}}
{{- if not (kindIs "map" $zipkinExporter) -}}
{{- fail "topology.otel.config.exporters.zipkin must be a map; fix topology.otel.config or set topology.zipkin.exporter.enabled=false" -}}
{{- end -}}
{{- $zipkinExporter = mergeOverwrite (deepCopy $zipkinExporter) (dict "endpoint" (include "nemo-retriever.zipkin.endpoint" .)) -}}
{{- $_ := set $exporters "zipkin" $zipkinExporter -}}
{{- $_ := set $config "exporters" $exporters -}}
{{- $traceExporters := get $traces "exporters" | default list -}}
{{- if not (has "zipkin" $traceExporters) -}}
{{- $traceExporters = append $traceExporters "zipkin" -}}
{{- end -}}
{{- range $exporterName := $traceExporters -}}
{{- if or (not (hasKey $exporters $exporterName)) (eq (get $exporters $exporterName) nil) -}}
{{- fail (printf "topology.otel.config.service.pipelines.traces trace exporter %q is missing or null in topology.otel.config.exporters; fix topology.otel.config or set topology.zipkin.exporter.enabled=false" $exporterName) -}}
{{- end -}}
{{- end -}}
{{- $_ := set $traces "exporters" $traceExporters -}}
{{- $_ := set $pipelines "traces" $traces -}}
{{- $_ := set $service "pipelines" $pipelines -}}
{{- $_ := set $config "service" $service -}}
{{- end -}}
{{- toYaml $config -}}
{{- end -}}

{{- define "nemo-retriever.serviceOtelEnv" -}}
{{- $root := .context -}}
{{- $role := .role -}}
{{- $serviceOtel := $root.Values.service.otel | default dict -}}
{{- $topologyOtel := $root.Values.topology.otel | default dict -}}
{{- if not (kindIs "map" $serviceOtel) -}}
{{- fail "service.otel must be a map" -}}
{{- end -}}
{{- if not (kindIs "map" $topologyOtel) -}}
{{- fail "topology.otel must be a map" -}}
{{- end -}}
{{- if and (get $topologyOtel "enabled") (get $serviceOtel "enabled") -}}
{{- $serviceOtelEnv := get $serviceOtel "env" | default dict -}}
{{- if not (kindIs "map" $serviceOtelEnv) -}}
{{- fail "service.otel.env must be a map" -}}
{{- end -}}
{{- $userEnvNames := dict -}}
{{- range $env := $root.Values.service.env -}}
{{- if and $env (hasKey $env "name") -}}
{{- $_ := set $userEnvNames $env.name true -}}
{{- end -}}
{{- end -}}
{{- $otelPorts := include "nemo-retriever.otel.ports" $root | fromYaml -}}
{{- $defaults := dict
  "OTEL_EXPORTER_OTLP_ENDPOINT" (printf "http://%s:%v" (include "nemo-retriever.otel.fullname" $root) (get $otelPorts "otlpGrpc"))
  "OTEL_SERVICE_NAME" (default "nemo-retriever-service" (get $serviceOtel "serviceName"))
  "OTEL_TRACES_EXPORTER" "otlp"
  "OTEL_METRICS_EXPORTER" "otlp"
  "OTEL_LOGS_EXPORTER" "none"
  "OTEL_PROPAGATORS" "tracecontext,baggage"
  "OTEL_RESOURCE_ATTRIBUTES" (printf "service.namespace=nemo-retriever,service.role=%s" $role)
  "OTEL_PYTHON_EXCLUDED_URLS" "health"
-}}
{{- $otelEnv := mergeOverwrite (deepCopy $defaults) (deepCopy $serviceOtelEnv) -}}
{{- range $name, $value := $otelEnv }}
{{- if not (hasKey $userEnvNames $name) }}
- name: {{ $name }}
  value: {{ $value | quote }}
{{- end }}
{{- end }}
{{- end -}}
{{- end -}}

{{- define "nemo-retriever.nimServiceOtelEnv" -}}
{{- $root := .context -}}
{{- $key := .key -}}
{{- $name := .name -}}
{{- $nim := index $root.Values.nimOperator $key -}}
{{- $chartOtel := $root.Values.nimOperator.otel | default dict -}}
{{- $nimOtel := $nim.otel | default dict -}}
{{- $topologyOtel := $root.Values.topology.otel | default dict -}}
{{- if not (kindIs "map" $chartOtel) -}}
{{- fail "nimOperator.otel must be a map" -}}
{{- end -}}
{{- if not (kindIs "map" $topologyOtel) -}}
{{- fail "topology.otel must be a map" -}}
{{- end -}}
{{- if not (kindIs "map" $nimOtel) -}}
{{- fail (printf "nimOperator.%s.otel must be a map" $key) -}}
{{- end -}}
{{- $enabled := get $chartOtel "enabled" -}}
{{- if and (hasKey $nimOtel "enabled") (not (eq (get $nimOtel "enabled") nil)) -}}
{{- $enabled = get $nimOtel "enabled" -}}
{{- end -}}
{{- if and (get $topologyOtel "enabled") $enabled -}}
{{- $chartNimOtelEnv := get $chartOtel "env" | default dict -}}
{{- $nimOtelEnv := get $nimOtel "env" | default dict -}}
{{- if not (kindIs "map" $chartNimOtelEnv) -}}
{{- fail "nimOperator.otel.env must be a map" -}}
{{- end -}}
{{- if not (kindIs "map" $nimOtelEnv) -}}
{{- fail (printf "nimOperator.%s.otel.env must be a map" $key) -}}
{{- end -}}
{{- $existingEnvNames := dict -}}
{{- $existingEnvValues := dict -}}
{{- range $env := $nim.env -}}
{{- if and $env (hasKey $env "name") -}}
{{- $_ := set $existingEnvNames $env.name true -}}
{{- if hasKey $env "value" -}}
{{- $_ := set $existingEnvValues $env.name $env.value -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- $otelPorts := include "nemo-retriever.otel.ports" $root | fromYaml -}}
{{- $defaultEndpoint := printf "http://%s:%v" (include "nemo-retriever.otel.fullname" $root) (get $otelPorts "otlpHttp") -}}
{{- $chartEndpoint := default $defaultEndpoint (get $chartOtel "endpoint") -}}
{{- $endpoint := default $chartEndpoint (get $nimOtel "endpoint") -}}
{{- $tritonPath := default "/v1/traces" (get $chartOtel "tritonPath") -}}
{{- $serviceName := default $name (get $nimOtel "serviceName") -}}
{{- $defaults := dict
  "NIM_ENABLE_OTEL" "true"
  "NIM_OTEL_SERVICE_NAME" $serviceName
  "NIM_OTEL_TRACES_EXPORTER" "otlp"
  "NIM_OTEL_METRICS_EXPORTER" "console"
  "NIM_OTEL_EXPORTER_OTLP_ENDPOINT" $endpoint
  "TRITON_OTEL_URL" (printf "%s%s" $endpoint $tritonPath)
  "TRITON_OTEL_RATE" "1"
-}}
{{- $explicitTritonUrl := or (hasKey $existingEnvNames "TRITON_OTEL_URL") (hasKey $chartNimOtelEnv "TRITON_OTEL_URL") (hasKey $nimOtelEnv "TRITON_OTEL_URL") -}}
{{- $existingEndpointWithoutLiteral := and (hasKey $existingEnvNames "NIM_OTEL_EXPORTER_OTLP_ENDPOINT") (not (hasKey $existingEnvValues "NIM_OTEL_EXPORTER_OTLP_ENDPOINT")) -}}
{{- $otelEnv := mergeOverwrite (deepCopy $defaults) (deepCopy $chartNimOtelEnv) (deepCopy $nimOtelEnv) -}}
{{- $finalEndpoint := get $otelEnv "NIM_OTEL_EXPORTER_OTLP_ENDPOINT" -}}
{{- if hasKey $existingEnvValues "NIM_OTEL_EXPORTER_OTLP_ENDPOINT" -}}
{{- $finalEndpoint = get $existingEnvValues "NIM_OTEL_EXPORTER_OTLP_ENDPOINT" -}}
{{- end -}}
{{- if not $explicitTritonUrl -}}
{{- if $existingEndpointWithoutLiteral -}}
{{- $_ := unset $otelEnv "TRITON_OTEL_URL" -}}
{{- else -}}
{{- $_ := set $otelEnv "TRITON_OTEL_URL" (printf "%s%s" $finalEndpoint $tritonPath) -}}
{{- end -}}
{{- end -}}
{{- range $envName, $envValue := $otelEnv }}
{{- if not (hasKey $existingEnvNames $envName) }}
- name: {{ $envName }}
  value: {{ $envValue | quote }}
{{- end }}
{{- end }}
{{- end -}}
{{- end -}}

{{/*
=============================================================================
NIMService GPU resources
=============================================================================

By default the chart sets ``spec.resources.limits.nvidia.com/gpu`` on
every NIMService (see ``nimOperator.nimServiceGpuLimit``) because the
NIM Operator does **not** reliably populate that field from the model
profile on all tested versions (for example v3.1.1 on A100/H100), which
otherwise leaves NIM pods without GPU access.

Helm and the operator may both server-side-apply the same field; a
later ``helm upgrade --install`` can then fail with an SSA conflict on
``.spec.resources.limits.nvidia.com/gpu``. See README §GPU limits and
``helm upgrade``.

Per-NIM ``nimOperator.<key>.resources`` replaces the whole block when
non-empty. When it is ``{}`` (the default), the chart-wide GPU limit
applies. Set ``nimOperator.nimServiceGpuLimit`` to ``null`` to omit the
``resources:`` block entirely (operator-only mode).
*/}}
{{- define "nemo-retriever.nimServiceResources" -}}
{{- $root := .context -}}
{{- $nimResources := .resources -}}
{{- $gpuLimit := $root.Values.nimOperator.nimServiceGpuLimit -}}
{{- if and $nimResources (gt (len $nimResources) 0) -}}
resources:
{{ toYaml $nimResources | indent 2 }}
{{- else if and (not (eq $gpuLimit nil)) $gpuLimit -}}
resources:
  limits:
    nvidia.com/gpu: {{ $gpuLimit }}
{{- end -}}
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
  page_elements                          -> nemotron-page-elements-v3                /v1/infer
  table_structure                        -> nemotron-table-structure-v1              /v1/infer
  ocr                                    -> nemotron-ocr-v1                          /v1/infer
  vlm_embed                              -> llama-nemotron-embed-vl-1b-v2            /v1/embeddings
  nemotron_3_nano_omni_30b_a3b_reasoning -> nemotron-3-nano-omni-30b-a3b-reasoning   /v1/chat/completions

Audio ASR (Parakeet) is configured directly via
  serviceConfig.nimEndpoints.audioGrpcEndpoint (no NIM Operator auto-wire).
*/}}

{{/*
Emit ``helm.sh/resource-policy: keep`` on NIMCache when
``nimOperator.nimCache.keepOnUninstall`` is true (default). Helm uninstall
then retains the cache CR (and its PVC) so model downloads are not discarded.
*/}}
{{- define "nemo-retriever.nimcache.keepPolicy" -}}
{{- if .Values.nimOperator.nimCache.keepOnUninstall }}
annotations:
  helm.sh/resource-policy: keep
{{- end }}
{{- end }}

{{/*
=============================================================================
NIMCache model-profile filter
=============================================================================

The NIM Operator's NIMCache CRD supports an optional
``spec.source.ngc.model`` block that restricts which model profiles the
cache job downloads.  Two filter dimensions are supported:

  spec.source.ngc.model.gpus      — list of {ids: [...], product: ...}
                                    selectors (PCI device IDs + display
                                    name); only profiles compatible with
                                    a listed GPU are downloaded.
  spec.source.ngc.model.profiles  — list of profile UUIDs; only those
                                    exact profiles are downloaded.

Without a filter the operator caches every profile applicable to the
GPUs it detects in the cluster, which on heterogeneous clusters (or any
cluster where the chart provisions ≥ 3 NIMs) wastes tens of GiB of PVC
storage, NGC bandwidth, and cache-job time.

Two knobs control the rendered ``model:`` block:

  .Values.nimOperator.modelProfile        — chart-wide default applied
                                            to every NIMCache that does
                                            not have its own override.
  .Values.nimOperator.<key>.modelProfile  — per-NIM override; when
                                            non-empty, REPLACES (does
                                            not merge with) the
                                            chart-wide default.

Both default to ``{}`` so the chart's behaviour is unchanged unless
the operator explicitly sets one of them. The mapping is rendered
verbatim under ``spec.source.ngc.model``, so the shape lines up 1:1
with the NIMCache CRD.

Usage inside ``templates/nims/<file>.yaml``:

  spec:
    source:
      ngc:
        modelPuller: "..."
        pullSecret: "..."
        authSecret: ...
        {{- include "nemo-retriever.nimcache.modelBlock"
              (dict "context" $ "key" "page_elements") | nindent 6 }}
*/}}
{{- define "nemo-retriever.nimcache.modelBlock" -}}
{{- $ctx := .context -}}
{{- $key := .key -}}
{{- $cfg := index $ctx.Values.nimOperator $key -}}
{{- $perNim := dict -}}
{{- if and $cfg (hasKey $cfg "modelProfile") -}}
{{- $perNim = ($cfg.modelProfile | default dict) -}}
{{- end -}}
{{- $global := ($ctx.Values.nimOperator.modelProfile | default dict) -}}
{{- $effective := dict -}}
{{- if $perNim -}}
{{- $effective = $perNim -}}
{{- else if $global -}}
{{- $effective = $global -}}
{{- end -}}
{{- if $effective -}}
model:
{{ toYaml $effective | indent 2 -}}
{{- end -}}
{{- end -}}

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
