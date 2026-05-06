{{/*
=============================================================================
Per-NIM resource bundle (StatefulSet + headless Service + cache VCT)
=============================================================================

StatefulSets with headless Services give each NIM pod a stable DNS name
(<name>-N.<name>.<ns>.svc.cluster.local) so the retriever service can
generate per-pod endpoint URLs and pin workers to individual replicas,
achieving the same application-level load spreading as the Docker setup.

Usage:
  {{ include "nemo-retriever.nim.bundle" (dict "context" $ "key" "pageElements" "shortName" "page-elements") }}

`key`        is the camelCase values.yaml key (e.g. "pageElements").
`shortName`  is the kebab-case suffix used in resource names ("page-elements").
*/}}

{{- define "nemo-retriever.nim.bundle" -}}
{{- $ctx := .context -}}
{{- $key := .key -}}
{{- $short := .shortName -}}
{{- $cfg := index $ctx.Values.nims $key -}}
{{- if and $ctx.Values.nims.enabled $cfg.enabled }}
{{- $merged := include "nemo-retriever.nim.merged" (dict "context" $ctx "key" $key) | fromYaml -}}
{{- $name := include "nemo-retriever.nim.fullname" (dict "context" $ctx "shortName" $short) -}}
{{- $port := int $merged.port -}}
{{- $labels := include "nemo-retriever.nim.labels" (dict "context" $ctx "shortName" $short) -}}
{{- $selector := include "nemo-retriever.nim.selectorLabels" (dict "context" $ctx "shortName" $short) }}
---
# Headless Service for StatefulSet pod DNS (enables per-pod endpoint URLs
# for application-level load spreading across NIM replicas).
apiVersion: v1
kind: Service
metadata:
  name: {{ $name }}
  labels:
    {{- $labels | nindent 4 }}
spec:
  type: ClusterIP
  clusterIP: None
  selector:
    {{- $selector | nindent 4 }}
  ports:
    - name: http
      protocol: TCP
      port: {{ $port }}
      targetPort: http
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: {{ $name }}
  labels:
    {{- $labels | nindent 4 }}
spec:
  serviceName: {{ $name }}
  replicas: {{ $merged.replicas }}
  selector:
    matchLabels:
      {{- $selector | nindent 6 }}
  template:
    metadata:
      labels:
        {{- $selector | nindent 8 }}
      {{- with $merged.podAnnotations }}
      annotations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
    spec:
      {{- include "nemo-retriever.nimImagePullSecrets" $ctx | nindent 6 }}
      {{- with $merged.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with $merged.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with $merged.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      # Triton/NIM containers need a generous /dev/shm. The emptyDir
      # below mounts at /dev/shm and replaces the default 64Mi.
      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: {{ $merged.shmSize | quote }}
      containers:
        - name: nim
          image: "{{ $cfg.image.repository }}:{{ $cfg.image.tag }}"
          imagePullPolicy: {{ default "IfNotPresent" $cfg.image.pullPolicy }}
          ports:
            - name: http
              containerPort: {{ $port }}
              protocol: TCP
          env:
            {{- with $merged.env }}
            {{- toYaml . | nindent 12 }}
            {{- end }}
            {{- if or $ctx.Values.nims.ngcApiKey.value $ctx.Values.nims.ngcApiKey.existingSecret }}
            - name: NGC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ include "nemo-retriever.ngcApiKeySecretName" $ctx }}
                  key: {{ $ctx.Values.nims.ngcApiKey.existingSecretKey }}
            {{- end }}
          {{- with $merged.envFrom }}
          envFrom:
            {{- toYaml . | nindent 12 }}
          {{- end }}
          resources:
            {{- toYaml $merged.resources | nindent 12 }}
          volumeMounts:
            - name: dshm
              mountPath: /dev/shm
            {{- if and $merged.cache $merged.cache.enabled }}
            - name: cache
              mountPath: {{ $merged.cache.mountPath }}
            {{- end }}
          {{- with $merged.startupProbe }}
          startupProbe:
            {{- toYaml . | nindent 12 }}
          {{- end }}
          {{- with $merged.livenessProbe }}
          livenessProbe:
            {{- toYaml . | nindent 12 }}
          {{- end }}
          {{- with $merged.readinessProbe }}
          readinessProbe:
            {{- toYaml . | nindent 12 }}
          {{- end }}
  {{- if and $merged.cache $merged.cache.enabled }}
  volumeClaimTemplates:
    - metadata:
        name: cache
      spec:
        accessModes:
          {{- toYaml $merged.cache.accessModes | nindent 10 }}
        resources:
          requests:
            storage: {{ $merged.cache.size | quote }}
        {{- if $merged.cache.storageClass }}
        {{- if eq "-" (toString $merged.cache.storageClass) }}
        storageClassName: ""
        {{- else }}
        storageClassName: {{ $merged.cache.storageClass | quote }}
        {{- end }}
        {{- end }}
  {{- end }}
{{- end }}
{{- end }}
