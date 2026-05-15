/* Jobs — SSE-driven job tracker with worker bars, stats, and job table */

function JobsView() {
  const [jobs, setJobs] = React.useState([]);
  const [summary, setSummary] = React.useState({});
  const [sseStatus, setSseStatus] = React.useState('connecting');
  const evtRef = React.useRef(null);

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
          setJobs(d.jobs || []);
          setSummary(d.summary || {});
          setSseStatus('connected');
        } catch {}
      });

      es.addEventListener('job_update', (e) => {
        try {
          const ev = JSON.parse(e.data);
          setJobs(prev => {
            const idx = prev.findIndex(j => j.id === ev.id);
            const rec = {
              id: ev.id,
              status: ev.status,
              result_rows: ev.result_rows,
              elapsed_s: ev.elapsed_s,
              error: ev.error,
              submitted_at: idx >= 0 ? prev[idx].submitted_at : null,
              started_at: idx >= 0 ? prev[idx].started_at : null,
              completed_at: ['completed', 'failed'].includes(ev.status) ? new Date().toISOString() : null,
            };
            if (idx >= 0) {
              const next = [...prev];
              next[idx] = { ...prev[idx], ...rec };
              return next;
            }
            return [rec, ...prev];
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
  }, []);

  const workerConfig = summary || {};
  const total = workerConfig.total_tracked || 0;
  const completed = workerConfig.completed || 0;
  const processing = workerConfig.processing || 0;
  const failed = workerConfig.failed || 0;
  const pending = workerConfig.pending || 0;

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
      processing: 'badge-yellow',
      pending: 'badge-blue',
    }[status] || 'badge-dim';
    return React.createElement('span', { className: `badge ${cls}` }, status);
  }

  const completionPct = total > 0 ? Math.round(((completed + failed) / total) * 100) : 0;

  const sortedJobs = React.useMemo(() => {
    return [...jobs].sort((a, b) => {
      const order = { processing: 0, pending: 1, failed: 2, completed: 3 };
      const oa = order[a.status] ?? 4;
      const ob = order[b.status] ?? 4;
      if (oa !== ob) return oa - ob;
      return (b.submitted_at || '').localeCompare(a.submitted_at || '');
    });
  }, [jobs]);

  return React.createElement(React.Fragment, null,

    /* SSE status indicator */
    React.createElement('div', {
      style: {
        display: 'flex', alignItems: 'center', gap: 8,
        marginBottom: 20, fontSize: 12, color: 'var(--nv-text-muted)',
      }
    },
      React.createElement('span', {
        className: `status-dot ${sseStatus === 'connected' ? 'ok' : sseStatus === 'connecting' ? 'unknown' : 'error'}`,
      }),
      `SSE: ${sseStatus}`,
      React.createElement('span', { style: { marginLeft: 'auto' } },
        `${total} jobs tracked`
      ),
    ),

    /* Progress bar */
    React.createElement('div', { className: 'section' },
      React.createElement('div', { className: 'section-title' }, 'Overall Progress'),
      React.createElement('div', { className: 'progress-bar' },
        React.createElement('div', {
          className: 'progress-fill',
          style: { width: completionPct + '%' },
        }),
        React.createElement('div', { className: 'progress-label' },
          `${completed + failed} / ${total} (${completionPct}%)`
        ),
      ),
    ),

    /* Stats row */
    React.createElement('div', { className: 'card-grid', style: { marginBottom: 24 } },
      [
        { label: 'Total',      value: total,      cls: '' },
        { label: 'Completed',  value: completed,  cls: 'badge-green' },
        { label: 'Processing', value: processing, cls: 'badge-yellow' },
        { label: 'Failed',     value: failed,     cls: 'badge-red' },
        { label: 'Pending',    value: pending,    cls: 'badge-blue' },
      ].map(s =>
        React.createElement('div', { key: s.label, className: 'card' },
          React.createElement('div', { className: 'card-title' }, s.label),
          React.createElement('div', { className: 'stat-value' }, s.value.toLocaleString()),
        )
      )
    ),

    /* Job table */
    React.createElement('div', { className: 'section' },
      React.createElement('div', { className: 'section-title' }, 'Jobs'),
      sortedJobs.length === 0
        ? React.createElement('div', { className: 'empty-state' }, 'No jobs tracked yet')
        : React.createElement('div', { className: 'table-wrap' },
            React.createElement('table', null,
              React.createElement('thead', null,
                React.createElement('tr', null,
                  ['ID', 'Status', 'Submitted', 'Elapsed', 'Rows', 'Error'].map(h =>
                    React.createElement('th', { key: h }, h)
                  )
                )
              ),
              React.createElement('tbody', null,
                sortedJobs.slice(0, 500).map(j =>
                  React.createElement('tr', { key: j.id },
                    React.createElement('td', { className: 'mono', style: { fontSize: 11 } },
                      (j.id || '').substring(0, 12) + '…'
                    ),
                    React.createElement('td', null, statusBadge(j.status)),
                    React.createElement('td', { title: j.submitted_at || '' },
                      relativeTime(j.submitted_at)
                    ),
                    React.createElement('td', { className: 'mono' },
                      j.elapsed_s != null ? j.elapsed_s.toFixed(1) + 's' : '—'
                    ),
                    React.createElement('td', { className: 'mono' },
                      j.result_rows != null ? j.result_rows.toLocaleString() : '—'
                    ),
                    React.createElement('td', {
                      style: { color: j.error ? 'var(--nv-red)' : 'inherit', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
                      title: j.error || '',
                    }, j.error || '—'),
                  )
                )
              )
            )
          )
    ),
  );
}
