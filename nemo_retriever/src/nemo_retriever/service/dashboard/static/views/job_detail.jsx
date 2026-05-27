/* JobDetailView — single-job drill-down (J10).
 *
 * Renders:
 *   • aggregate header (status, counts, progress bar, label, timestamps)
 *   • throughput + cumulative-completion mini charts (pure SVG)
 *   • paginated document table backed by /v1/dashboard/api/jobs/{id}/documents
 *   • live SSE feed subscribed to /v1/ingest/job/{id}/events
 *
 * Charts are intentionally library-free (no Chart.js / Recharts) so the
 * dashboard ships as plain JSX over the React UMD bundle.
 */

function JobDetailView({ jobId, onBack }) {
  const [job, setJob] = React.useState(null);
  const [docs, setDocs] = React.useState([]);
  const [docTotal, setDocTotal] = React.useState(0);
  const [docTotalFiltered, setDocTotalFiltered] = React.useState(0);
  const [docOffset, setDocOffset] = React.useState(0);
  const [docLimit] = React.useState(100);
  const [docStatusFilter, setDocStatusFilter] = React.useState('');
  const [events, setEvents] = React.useState([]);
  const [throughput, setThroughput] = React.useState([]); // {t, completed, failed}
  const [sseStatus, setSseStatus] = React.useState('connecting');
  const [error, setError] = React.useState(null);
  const startedAtRef = React.useRef(null);

  // ------------------------------------------------------------------
  // Initial + on-status-change fetch of aggregate and docs.
  // ------------------------------------------------------------------
  const fetchAggregate = React.useCallback(async () => {
    try {
      const resp = await fetch(`/v1/dashboard/api/jobs/${jobId}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setJob(data);
      if (data.started_at && !startedAtRef.current) {
        startedAtRef.current = new Date(data.started_at).getTime();
      }
      setError(null);
    } catch (e) {
      setError(`aggregate: ${e}`);
    }
  }, [jobId]);

  const fetchDocs = React.useCallback(async () => {
    try {
      const params = new URLSearchParams({
        offset: String(docOffset),
        limit: String(docLimit),
      });
      if (docStatusFilter) params.set('status', docStatusFilter);
      const resp = await fetch(`/v1/dashboard/api/jobs/${jobId}/documents?${params}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setDocs(data.items || []);
      setDocTotal(data.total || 0);
      setDocTotalFiltered(data.total_filtered || 0);
      setError(null);
    } catch (e) {
      setError(`documents: ${e}`);
    }
  }, [jobId, docOffset, docLimit, docStatusFilter]);

  React.useEffect(() => { fetchAggregate(); }, [fetchAggregate]);
  React.useEffect(() => { fetchDocs(); }, [fetchDocs]);

  // ------------------------------------------------------------------
  // Live per-job SSE — subscribes to the J4 per-job stream.
  // We slot incoming events into both the recent-events log and the
  // throughput series; aggregate-level events refresh the header.
  // ------------------------------------------------------------------
  React.useEffect(() => {
    let es = null;
    let retryTimer = null;

    function connect() {
      setSseStatus('connecting');
      es = new EventSource(`/v1/ingest/job/${jobId}/events`);

      const pushEvent = (kind, payload) => {
        setEvents(prev => [
          { id: `${kind}:${Date.now()}:${Math.random()}`, kind, payload, t: Date.now() },
          ...prev,
        ].slice(0, 200));
      };

      // Generic catch-all: we don't know exactly which event names the
      // backend emits, so we listen for the SSE event types we know
      // about ("completed", "failed", "job_progress", "job_finalized",
      // "job_partial", "job_failed", "job_started", "job_created").
      // The "message" handler covers anything else.
      const handlerFor = (kind) => (e) => {
        try {
          const data = JSON.parse(e.data);
          pushEvent(kind, data);

          if (['completed', 'failed', 'processing', 'pending'].includes(kind)) {
            // Update the in-row status optimistically.
            setDocs(prev => prev.map(d =>
              d.id === data.id || d.id === data.document_id
                ? { ...d, status: kind, error: data.error, elapsed_s: data.elapsed_s, result_rows: data.result_rows }
                : d
            ));
            setThroughput(prev => {
              const tail = prev[prev.length - 1] || { completed: 0, failed: 0 };
              const next = {
                t: Date.now(),
                completed: tail.completed + (kind === 'completed' ? 1 : 0),
                failed: tail.failed + (kind === 'failed' ? 1 : 0),
              };
              return [...prev.slice(-119), next];
            });
            setSseStatus('connected');
          }

          if (kind.startsWith('job_')) {
            // Job lifecycle: refresh header from the event payload.
            setJob(prev => prev ? {
              ...prev,
              status: data.status || prev.status,
              counts: data.counts || prev.counts,
              expected_documents: data.expected_documents != null ? data.expected_documents : prev.expected_documents,
              started_at: data.started_at || prev.started_at,
              finalized_at: data.finalized_at || prev.finalized_at,
              elapsed_s: data.elapsed_s != null ? data.elapsed_s : prev.elapsed_s,
            } : prev);
            setSseStatus('connected');
          }
        } catch {}
      };

      const eventTypes = [
        'completed', 'failed', 'processing', 'pending',
        'job_created', 'job_started', 'job_progress',
        'job_finalized', 'job_partial', 'job_failed',
      ];
      for (const t of eventTypes) es.addEventListener(t, handlerFor(t));

      es.onopen = () => setSseStatus('connected');
      es.onerror = () => {
        setSseStatus('disconnected');
        es.close();
        retryTimer = setTimeout(connect, 3000);
      };
    }

    connect();
    return () => {
      if (es) es.close();
      if (retryTimer) clearTimeout(retryTimer);
    };
  }, [jobId]);

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------
  function statusBadge(status) {
    const cls = {
      completed: 'badge-green',
      failed: 'badge-red',
      partial_success: 'badge-yellow',
      processing: 'badge-yellow',
      running: 'badge-yellow',
      pending: 'badge-blue',
    }[status] || 'badge-dim';
    return React.createElement('span', { className: `badge ${cls}` }, status);
  }

  function fmtTime(iso) {
    if (!iso) return '—';
    return new Date(iso).toLocaleString();
  }

  function progressPct() {
    if (!job || !job.expected_documents) return 0;
    const c = (job.counts && job.counts.completed) || 0;
    const f = (job.counts && job.counts.failed) || 0;
    return Math.min(100, Math.round(((c + f) / job.expected_documents) * 100));
  }

  function downloadDoc(docId) {
    window.open(`/v1/ingest/job/${jobId}/document/${docId}`, '_blank');
  }

  // ------------------------------------------------------------------
  // Charts (pure-SVG so we don't depend on any chart library).
  // 'series' is an array of {t, completed, failed}; we plot cumulative
  // completed/failed counts as two line series within a 600x140 canvas.
  // ------------------------------------------------------------------
  function ThroughputChart({ series }) {
    const W = 600, H = 140, P = 24;
    if (series.length < 2) {
      return React.createElement('div', { className: 'empty-state', style: { padding: 24 } },
        'Waiting for events…'
      );
    }
    const tMin = series[0].t;
    const tMax = series[series.length - 1].t;
    const dt = Math.max(1, tMax - tMin);
    const maxY = Math.max(1, ...series.map(s => s.completed + s.failed));
    const x = (t) => P + ((t - tMin) / dt) * (W - 2 * P);
    const y = (v) => H - P - (v / maxY) * (H - 2 * P);
    const pathFor = (key, color) => {
      const d = series.map((s, i) =>
        `${i === 0 ? 'M' : 'L'} ${x(s.t).toFixed(1)} ${y(s[key]).toFixed(1)}`
      ).join(' ');
      return React.createElement('path', {
        d, fill: 'none', stroke: color, strokeWidth: 2,
      });
    };
    return React.createElement('svg', { width: W, height: H, style: { display: 'block' } },
      React.createElement('rect', {
        x: 0, y: 0, width: W, height: H, fill: 'var(--nv-surface)',
      }),
      pathFor('completed', 'var(--nv-green)'),
      pathFor('failed', 'var(--nv-red)'),
      React.createElement('text', {
        x: P, y: H - 4, fontSize: 10, fill: 'var(--nv-text-muted)',
      }, new Date(tMin).toLocaleTimeString()),
      React.createElement('text', {
        x: W - P - 60, y: H - 4, fontSize: 10, fill: 'var(--nv-text-muted)',
      }, new Date(tMax).toLocaleTimeString()),
      React.createElement('text', {
        x: P, y: P, fontSize: 11, fill: 'var(--nv-green)', fontWeight: 600,
      }, `Completed: ${series[series.length - 1].completed}`),
      React.createElement('text', {
        x: P + 130, y: P, fontSize: 11, fill: 'var(--nv-red)', fontWeight: 600,
      }, `Failed: ${series[series.length - 1].failed}`),
    );
  }

  const pct = progressPct();
  const c = job ? ((job.counts && job.counts.completed) || 0) : 0;
  const f = job ? ((job.counts && job.counts.failed) || 0) : 0;
  const p = job ? ((job.counts && job.counts.processing) || 0) : 0;
  const exp = job ? (job.expected_documents || 0) : 0;

  return React.createElement(React.Fragment, null,

    /* Header bar */
    React.createElement('div', {
      style: { display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, flexWrap: 'wrap' }
    },
      React.createElement('button', {
        className: 'btn',
        style: { padding: '6px 12px', fontSize: 12, background: 'var(--nv-surface)', color: 'var(--nv-text)' },
        onClick: () => onBack && onBack(),
      }, '← Back to jobs'),
      React.createElement('span', {
        className: 'mono',
        style: { fontSize: 13, color: 'var(--nv-text-muted)' }
      }, jobId),
      job && statusBadge(job.status),
      React.createElement('span', {
        className: `status-dot ${sseStatus === 'connected' ? 'ok' : sseStatus === 'connecting' ? 'unknown' : 'error'}`,
        style: { marginLeft: 'auto' },
      }),
      React.createElement('span', { style: { fontSize: 12, color: 'var(--nv-text-muted)' } },
        `SSE: ${sseStatus}`
      ),
    ),

    error && React.createElement('div', {
      className: 'card',
      style: { marginBottom: 16, color: 'var(--nv-red)' }
    }, error),

    /* Aggregate header card */
    job && React.createElement('div', {
      className: 'card',
      style: { marginBottom: 24 }
    },
      React.createElement('div', {
        style: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }
      },
        React.createElement('div', null,
          React.createElement('div', { style: { fontSize: 18, fontWeight: 600 } },
            job.label || `Job ${jobId.substring(0, 12)}…`
          ),
          React.createElement('div', { style: { fontSize: 12, color: 'var(--nv-text-muted)', marginTop: 4 } },
            `Created ${fmtTime(job.created_at)}  •  Started ${fmtTime(job.started_at)}  •  Finalized ${fmtTime(job.finalized_at)}`
          ),
        ),
        React.createElement('div', {
          className: 'mono',
          style: { fontSize: 13, color: 'var(--nv-text-muted)' }
        }, job.elapsed_s != null ? `${job.elapsed_s.toFixed(1)}s elapsed` : ''),
      ),
      React.createElement('div', { className: 'progress-bar', style: { height: 24 } },
        React.createElement('div', {
          className: 'progress-fill',
          style: { width: pct + '%' },
        }),
        React.createElement('div', { className: 'progress-label' },
          `${c + f} / ${exp} (${pct}%)`
        ),
      ),
      React.createElement('div', {
        style: { display: 'flex', gap: 24, marginTop: 16, fontSize: 13 }
      },
        React.createElement('span', null,
          React.createElement('span', { style: { color: 'var(--nv-green)', fontWeight: 600 } }, `${c}`),
          ' completed',
        ),
        React.createElement('span', null,
          React.createElement('span', { style: { color: 'var(--nv-yellow)', fontWeight: 600 } }, `${p}`),
          ' processing',
        ),
        React.createElement('span', null,
          React.createElement('span', { style: { color: 'var(--nv-red)', fontWeight: 600 } }, `${f}`),
          ' failed',
        ),
        React.createElement('span', null,
          React.createElement('span', { style: { color: 'var(--nv-blue)', fontWeight: 600 } }, `${exp - c - f - p}`),
          ' pending',
        ),
      ),
    ),

    /* Chart + recent events */
    React.createElement('div', {
      style: { display: 'grid', gridTemplateColumns: '1fr 360px', gap: 16, marginBottom: 24 }
    },
      React.createElement('div', { className: 'card' },
        React.createElement('div', { className: 'card-title' }, 'Cumulative throughput'),
        React.createElement(ThroughputChart, { series: throughput }),
      ),
      React.createElement('div', { className: 'card', style: { maxHeight: 220, overflowY: 'auto' } },
        React.createElement('div', { className: 'card-title' }, 'Live events'),
        events.length === 0
          ? React.createElement('div', { style: { fontSize: 12, color: 'var(--nv-text-muted)' } }, 'No events yet')
          : React.createElement('div', { style: { fontSize: 11 } },
              events.slice(0, 50).map(ev =>
                React.createElement('div', {
                  key: ev.id,
                  style: { padding: '4px 0', borderBottom: '1px solid var(--nv-border)' }
                },
                  React.createElement('span', { className: 'mono', style: { color: 'var(--nv-text-muted)', marginRight: 8 } },
                    new Date(ev.t).toLocaleTimeString()
                  ),
                  React.createElement('span', { style: {
                    color: ev.kind === 'completed' ? 'var(--nv-green)'
                      : ev.kind === 'failed' || ev.kind === 'job_failed' ? 'var(--nv-red)'
                      : ev.kind.startsWith('job_') ? 'var(--nv-blue)'
                      : 'var(--nv-text-muted)',
                    fontWeight: 600,
                  } }, ev.kind),
                  ev.payload && (ev.payload.id || ev.payload.document_id) &&
                    React.createElement('span', { className: 'mono', style: { color: 'var(--nv-text-muted)', marginLeft: 8 } },
                      `(${(ev.payload.id || ev.payload.document_id).substring(0, 8)}…)`
                    ),
                )
              )
            ),
      ),
    ),

    /* Documents table */
    React.createElement('div', { className: 'section' },
      React.createElement('div', {
        style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }
      },
        React.createElement('div', { className: 'section-title', style: { marginBottom: 0 } }, 'Documents'),
        React.createElement('div', { style: { display: 'flex', gap: 8, fontSize: 12, color: 'var(--nv-text-muted)' } },
          `${docs.length} of ${docTotalFiltered} (${docTotal} total)`,
          React.createElement('select', {
            className: 'input',
            style: { padding: '4px 8px', fontSize: 12 },
            value: docStatusFilter,
            onChange: (e) => { setDocOffset(0); setDocStatusFilter(e.target.value); },
          },
            React.createElement('option', { value: '' }, 'All statuses'),
            ['pending', 'processing', 'completed', 'failed'].map(s =>
              React.createElement('option', { key: s, value: s }, s)
            ),
          ),
          React.createElement('button', {
            className: 'btn btn-primary',
            style: { padding: '6px 12px', fontSize: 12 },
            onClick: () => { fetchAggregate(); fetchDocs(); },
          }, 'Refresh'),
        ),
      ),
      docs.length === 0
        ? React.createElement('div', { className: 'empty-state' }, 'No documents to show')
        : React.createElement('div', { className: 'table-wrap' },
            React.createElement('table', null,
              React.createElement('thead', null,
                React.createElement('tr', null,
                  ['Document ID', 'Filename', 'Status', 'Submitted', 'Elapsed', 'Rows', 'Error', ''].map(h =>
                    React.createElement('th', { key: h }, h)
                  )
                )
              ),
              React.createElement('tbody', null,
                docs.map(d =>
                  React.createElement('tr', { key: d.id },
                    React.createElement('td', { className: 'mono', style: { fontSize: 11 } },
                      (d.id || '').substring(0, 12) + '…'
                    ),
                    React.createElement('td', {
                      style: { maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
                      title: d.filename || '',
                    }, d.filename || '—'),
                    React.createElement('td', null, statusBadge(d.status)),
                    React.createElement('td', { title: d.submitted_at || '' },
                      d.submitted_at ? new Date(d.submitted_at).toLocaleTimeString() : '—'
                    ),
                    React.createElement('td', { className: 'mono' },
                      d.elapsed_s != null ? d.elapsed_s.toFixed(1) + 's' : '—'
                    ),
                    React.createElement('td', { className: 'mono' },
                      d.result_rows != null ? d.result_rows.toLocaleString() : '—'
                    ),
                    React.createElement('td', {
                      style: {
                        color: d.error ? 'var(--nv-red)' : 'inherit',
                        maxWidth: 300, overflow: 'hidden',
                        textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                      },
                      title: d.error || '',
                    }, d.error || '—'),
                    React.createElement('td', null,
                      ['completed', 'failed'].includes(d.status) &&
                        React.createElement('button', {
                          className: 'btn',
                          style: { padding: '2px 8px', fontSize: 11, background: 'var(--nv-surface)', color: 'var(--nv-text)' },
                          onClick: () => downloadDoc(d.id),
                        }, 'View'),
                    ),
                  )
                )
              )
            )
          )
    ),

    /* Pagination */
    docTotalFiltered > docLimit && React.createElement('div', {
      style: { display: 'flex', justifyContent: 'center', gap: 8, marginTop: 16 }
    },
      React.createElement('button', {
        className: 'btn',
        style: { padding: '6px 12px', fontSize: 12, opacity: docOffset === 0 ? 0.5 : 1 },
        disabled: docOffset === 0,
        onClick: () => setDocOffset(Math.max(0, docOffset - docLimit)),
      }, '← Prev'),
      React.createElement('span', { style: { padding: '6px 12px', fontSize: 12, color: 'var(--nv-text-muted)' } },
        `Page ${Math.floor(docOffset / docLimit) + 1} of ${Math.max(1, Math.ceil(docTotalFiltered / docLimit))}`
      ),
      React.createElement('button', {
        className: 'btn',
        style: { padding: '6px 12px', fontSize: 12, opacity: docOffset + docLimit >= docTotalFiltered ? 0.5 : 1 },
        disabled: docOffset + docLimit >= docTotalFiltered,
        onClick: () => setDocOffset(docOffset + docLimit),
      }, 'Next →'),
    ),
  );
}
