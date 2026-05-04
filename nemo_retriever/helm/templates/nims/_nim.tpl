{{/*
=============================================================================
Per-NIM resource bundle (Deployment + Service + optional cache PVC)
=============================================================================

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
apiVersion: v1
kind: Service
metadata:
  name: {{ $name }}
  labels:
    {{- $labels | nindent 4 }}
spec:
  type: ClusterIP
  selector:
    {{- $selector | nindent 4 }}
  ports:
    - name: http
      protocol: TCP
      port: {{ $port }}
      targetPort: http
---
{{- if and $merged.cache $merged.cache.enabled }}
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {{ printf "%s-cache" $name }}
  labels:
    {{- $labels | nindent 4 }}
spec:
  accessModes:
    {{- toYaml $merged.cache.accessModes | nindent 4 }}
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
---
{{- end }}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ $name }}
  labels:
    {{- $labels | nindent 4 }}
spec:
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
        {{- if and $merged.cache $merged.cache.enabled }}
        - name: cache
          persistentVolumeClaim:
            claimName: {{ printf "%s-cache" $name }}
        {{- end }}
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
{{- end }}
{{- end }}
