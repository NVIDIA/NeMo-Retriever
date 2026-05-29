/* Jobs — top-level list of job aggregates.
 *
 * Drives off the J8 dashboard endpoints:
 *   GET  /v1/dashboard/api/jobs/list?status=&offset=&limit=&sort=
 *   GET  /v1/dashboard/api/jobs               (SSE — both job_update and job_lifecycle)
 *
 * Clicking a row navigates to the job-detail view (J10) via hash route
 * "#job/<job_id>". This component intentionally renders one row per
 * *job*, not per *document* — drill-down lives on the detail page.
 */

function JobsView({ onOpenJob }) {
  const [jobs, setJobs] = React.useState([]);
  const [summary, setSummary] = React.useState({});
  const [statusFilter, setStatusFilter] = React.useState('');
  const [sort, setSort] = React.useState('created_desc');
  const [offset, setOffset] = React.useState(0);
  const [limit] = React.useState(50);
  const [total, setTotal] = React.useState(0);
  const [totalFiltered, setTotalFiltered] = React.useState(0);
  const [sseStatus, setSseStatus] = React.useState('connecting');
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState(null);
  const evtRef = React.useRef(null);

  // ------------------------------------------------------------------
  // REST page fetch — the source of truth for the visible window.
  // SSE only nudges us to re-fetch (or splice updates in place).
  // ------------------------------------------------------------------
  const fetchPage = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        offset: String(offset),
        limit: String(limit),
        sort,
      });
      if (statusFilter) params.set('status', statusFilter);
      const resp = await fetch(`/v1/dashboard/api/jobs/list?${params}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setJobs(data.jobs || []);
      setTotal(data.total || 0);
      setTotalFiltered(data.total_filtered || 0);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [offset, limit, sort, statusFilter]);

  React.useEffect(() => { fetchPage(); }, [fetchPage]);

  // ------------------------------------------------------------------
  // SSE connection (global firehose for the dashboard router).
  // We only listen for events that affect the visible page; on a
  // lifecycle transition we patch the matching row inline so the UI
  // updates without a full refetch.
  // ------------------------------------------------------------------
  React.useEffect(() => {
    let es = null;
    let retryTimer = null;

    function connect() {
      setSseStatus('connecting');
      es = new EventSource('/v1/dashboard/api/jobs');
      evtRef.current = es;

      es.addEventListener('snapshot', (e) => {
        try {
          const d = JSON.parse(e.data);
          setSummary(d.summary || {});
          setSseStatus('connected');
        } catch {}
      });

      // Per-document events: only used to bump completed/failed counts
      // on a visible job row so progress bars feel live.
      es.addEventListener('job_update', (e) => {
        try {
          const ev = JSON.parse(e.data);
          const jid = ev.job_id;
          if (!jid) return;
          setJobs(prev => prev.map(j => {
            if (j.job_id !== jid) return j;
            const counts = { ...(j.counts || {}) };
            if (ev.status === 'completed') counts.completed = (counts.completed || 0) + 1;
            if (ev.status === 'failed') counts.failed = (counts.failed || 0) + 1;
            if (ev.status === 'processing') counts.processing = (counts.processing || 0) + 1;
            return { ...j, counts };
          }));
        } catch {}
      });

      // Job-lifecycle events (J5): replace the entire row payload.
      es.addEventListener('job_lifecycle', (e) => {
        try {
          const ev = JSON.parse(e.data);
          if (!ev.job_id) return;
          setJobs(prev => {
            const idx = prev.findIndex(j => j.job_id === ev.job_id);
            const patched = {
              job_id: ev.job_id,
              status: ev.status,
              expected_documents: ev.expected_documents,
              counts: ev.counts || {},
              elapsed_s: ev.elapsed_s,
              started_at: ev.started_at,
              finalized_at: ev.finalized_at,
              label: ev.label,
              document_ids: idx >= 0 ? prev[idx].document_ids : [],
              created_at: idx >= 0 ? prev[idx].created_at : null,
            };
            if (idx >= 0) {
              const next = [...prev];
              next[idx] = { ...prev[idx], ...patched };
              return next;
            }
            // job_created for a brand-new job: prepend if we're on page 0
            // and sorting newest-first.
            if (ev.type === 'job_created' && offset === 0 && sort === 'created_desc') {
              return [patched, ...prev].slice(0, limit);
            }
            return prev;
          });
        } catch {}
      });

      es.addEventListener('heartbeat', (e) => {
        try {
          const d = JSON.parse(e.data);
          if (d.summary) setSummary(d.summary);
        } catch {}
      });

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
  }, [offset, sort, limit]);

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------
  function relativeTime(iso) {
    if (!iso) return '—';
    const diff = (Date.now() - new Date(iso).getTime()) / 1000;
    if (diff < 60) return Math.round(diff) + 's ago';
    if (diff < 3600) return Math.round(diff / 60) + 'm ago';
    if (diff < 86400) return Math.round(diff / 3600) + 'h ago';
    return Math.round(diff / 86400) + 'd ago';
  }

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

  function progressFor(j) {
    const exp = j.expected_documents || 0;
    if (!exp) return 0;
    const c = (j.counts && j.counts.completed) || 0;
    const f = (j.counts && j.counts.failed) || 0;
    return Math.min(100, Math.round(((c + f) / exp) * 100));
  }

  const totalJobs = summary.total_jobs || 0;
  const totalDocs = summary.total_documents || 0;
  const jobsCompleted = summary.completed || 0;
  const jobsFailed = summary.failed || 0;
  const jobsPartial = summary.partial_success || 0;
  const jobsRunning = (summary.running || 0) + (summary.processing || 0);

  return React.createElement(React.Fragment, null,

    /* SSE status + filters bar */
    React.createElement('div', {
      style: {
        display: 'flex', alignItems: 'center', gap: 12,
        marginBottom: 20, fontSize: 12, color: 'var(--nv-text-muted)',
        flexWrap: 'wrap',
      }
    },
      React.createElement('span', {
        className: `status-dot ${sseStatus === 'connected' ? 'ok' : sseStatus === 'connecting' ? 'unknown' : 'error'}`,
      }),
      `SSE: ${sseStatus}`,
      React.createElement('span', { style: { marginLeft: 16 } },
        `Showing ${jobs.length} of ${totalFiltered} jobs (${total} total)`
      ),
      React.createElement('div', { style: { marginLeft: 'auto', display: 'flex', gap: 8 } },
        React.createElement('select', {
          className: 'input',
          style: { padding: '4px 8px', fontSize: 12 },
          value: statusFilter,
          onChange: (e) => { setOffset(0); setStatusFilter(e.target.value); },
        },
          React.createElement('option', { value: '' }, 'All statuses'),
          ['pending', 'running', 'completed', 'failed', 'partial_success'].map(s =>
            React.createElement('option', { key: s, value: s }, s)
          ),
        ),
        React.createElement('select', {
          className: 'input',
          style: { padding: '4px 8px', fontSize: 12 },
          value: sort,
          onChange: (e) => { setOffset(0); setSort(e.target.value); },
        },
          React.createElement('option', { value: 'created_desc' }, 'Newest first'),
          React.createElement('option', { value: 'created_asc' }, 'Oldest first'),
          React.createElement('option', { value: 'finalized_desc' }, 'Recently finished'),
          React.createElement('option', { value: 'finalized_asc' }, 'Earliest finished'),
        ),
        React.createElement('button', {
          className: 'btn btn-primary',
          style: { padding: '6px 12px', fontSize: 12 },
          onClick: fetchPage,
        }, loading ? 'Loading…' : 'Refresh'),
      ),
    ),

    /* Stats row */
    React.createElement('div', { className: 'card-grid', style: { marginBottom: 24 } },
      [
        { label: 'Jobs',     value: totalJobs },
        { label: 'Documents', value: totalDocs },
        { label: 'Running',  value: jobsRunning },
        { label: 'Completed', value: jobsCompleted },
        { label: 'Failed',   value: jobsFailed },
        { label: 'Partial',  value: jobsPartial },
      ].map(s =>
        React.createElement('div', { key: s.label, className: 'card' },
          React.createElement('div', { className: 'card-title' }, s.label),
          React.createElement('div', { className: 'stat-value' }, s.value.toLocaleString()),
        )
      )
    ),

    error && React.createElement('div', {
      className: 'card',
      style: { marginBottom: 16, color: 'var(--nv-red)' }
    }, `Failed to load jobs: ${error}`),

    /* Job table */
    React.createElement('div', { className: 'section' },
      React.createElement('div', { className: 'section-title' }, 'Jobs'),
      jobs.length === 0 && !loading
        ? React.createElement('div', { className: 'empty-state' }, 'No jobs match the current filters')
        : React.createElement('div', { className: 'table-wrap' },
            React.createElement('table', null,
              React.createElement('thead', null,
                React.createElement('tr', null,
                  ['Job ID', 'Status', 'Label', 'Progress', 'Docs', 'Created', 'Elapsed'].map(h =>
                    React.createElement('th', { key: h }, h)
                  )
                )
              ),
              React.createElement('tbody', null,
                jobs.map(j => {
                  const exp = j.expected_documents || 0;
                  const c = (j.counts && j.counts.completed) || 0;
                  const f = (j.counts && j.counts.failed) || 0;
                  const pct = progressFor(j);
                  return React.createElement('tr', {
                    key: j.job_id,
                    style: { cursor: onOpenJob ? 'pointer' : 'default' },
                    onClick: () => onOpenJob && onOpenJob(j.job_id),
                  },
                    React.createElement('td', { className: 'mono', style: { fontSize: 11 } },
                      React.createElement('a', {
                        href: `#job/${j.job_id}`,
                        onClick: (e) => {
                          if (onOpenJob) {
                            e.preventDefault();
                            e.stopPropagation();
                            onOpenJob(j.job_id);
                          }
                        },
                        style: { color: 'var(--nv-green)', textDecoration: 'none' },
                      }, (j.job_id || '').substring(0, 12) + '…')
                    ),
                    React.createElement('td', null, statusBadge(j.status)),
                    React.createElement('td', {
                      style: { maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
                      title: j.label || '',
                    }, j.label || '—'),
                    React.createElement('td', { style: { width: 220 } },
                      React.createElement('div', { className: 'progress-bar', style: { height: 14 } },
                        React.createElement('div', {
                          className: 'progress-fill',
                          style: { width: pct + '%' },
                        }),
                        React.createElement('div', { className: 'progress-label', style: { fontSize: 10 } },
                          `${c + f} / ${exp} (${pct}%)`
                        ),
                      ),
                    ),
                    React.createElement('td', { className: 'mono', style: { fontSize: 11 } },
                      `${c}✓ ${f}✗ / ${exp}`
                    ),
                    React.createElement('td', { title: j.created_at || '' },
                      relativeTime(j.created_at)
                    ),
                    React.createElement('td', { className: 'mono' },
                      j.elapsed_s != null ? j.elapsed_s.toFixed(1) + 's' : '—'
                    ),
                  );
                })
              )
            )
          )
    ),

    /* Pagination */
    totalFiltered > limit && React.createElement('div', {
      style: { display: 'flex', justifyContent: 'center', gap: 8, marginTop: 16 }
    },
      React.createElement('button', {
        className: 'btn',
        style: { padding: '6px 12px', fontSize: 12, opacity: offset === 0 ? 0.5 : 1 },
        disabled: offset === 0,
        onClick: () => setOffset(Math.max(0, offset - limit)),
      }, '← Prev'),
      React.createElement('span', { style: { padding: '6px 12px', fontSize: 12, color: 'var(--nv-text-muted)' } },
        `Page ${Math.floor(offset / limit) + 1} of ${Math.max(1, Math.ceil(totalFiltered / limit))}`
      ),
      React.createElement('button', {
        className: 'btn',
        style: { padding: '6px 12px', fontSize: 12, opacity: offset + limit >= totalFiltered ? 0.5 : 1 },
        disabled: offset + limit >= totalFiltered,
        onClick: () => setOffset(offset + limit),
      }, 'Next →'),
    ),
  );
}
