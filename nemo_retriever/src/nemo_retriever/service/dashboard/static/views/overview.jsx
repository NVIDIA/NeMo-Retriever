/* Overview — cluster topology, job summary, system info */

function OverviewView() {
  const [data, setData] = React.useState(null);
  const [error, setError] = React.useState(null);

  const fetchData = React.useCallback(() => {
    fetch('/v1/dashboard/api/overview')
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(setData)
      .catch(e => setError(e.message));
  }, []);

  React.useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 10000);
    return () => clearInterval(id);
  }, [fetchData]);

  if (error) {
    return React.createElement('div', { className: 'empty-state' },
      'Failed to load overview: ' + error
    );
  }
  if (!data) {
    return React.createElement('div', { className: 'empty-state' }, 'Loading…');
  }

  const backends = data.backends || {};
  const vdb = data.vectordb;
  const summary = data.job_summary || {};
  const workers = data.worker_config || {};
  const gw = data.gateway || {};

  const pods = [
    {
      name: 'Gateway',
      status: 'ok',
      details: [
        gw.realtime_url ? 'RT: ' + gw.realtime_url : null,
        gw.batch_url ? 'Batch: ' + gw.batch_url : null,
      ].filter(Boolean),
    },
    {
      name: 'Realtime',
      status: backends.realtime ? 'ok' : 'error',
      details: [
        `Workers: ${workers.realtime_workers || '—'}`,
        `Queue: ${workers.realtime_queue_size || '—'}`,
      ],
    },
    {
      name: 'Batch',
      status: backends.batch ? 'ok' : 'error',
      details: [
        `Workers: ${workers.batch_workers || '—'}`,
        `Queue: ${workers.batch_queue_size || '—'}`,
      ],
    },
    {
      name: 'VectorDB',
      status: vdb ? 'ok' : 'unknown',
      details: vdb ? [
        `Table: ${vdb.table || '—'}`,
        `Rows: ${(vdb.total_rows || 0).toLocaleString()}`,
      ] : ['Not connected'],
    },
  ];

  return React.createElement(React.Fragment, null,

    /* Topology */
    React.createElement('div', { className: 'section' },
      React.createElement('div', { className: 'section-title' }, 'Pod Topology'),
      React.createElement('div', { className: 'topology-grid' },
        pods.map(p =>
          React.createElement('div', { key: p.name, className: 'pod-card' },
            React.createElement('div', { className: 'pod-card-header' },
              React.createElement('span', { className: `status-dot ${p.status}` }),
              React.createElement('span', { className: 'pod-card-name' }, p.name),
            ),
            p.details.map((d, i) =>
              React.createElement('div', { key: i, className: 'pod-card-detail' }, d)
            ),
          )
        )
      )
    ),

    /* Job summary */
    React.createElement('div', { className: 'section' },
      React.createElement('div', { className: 'section-title' }, 'Job Summary'),
      React.createElement('div', { className: 'card-grid' },
        [
          { label: 'Total Tracked', value: summary.total_tracked || 0, cls: '' },
          { label: 'Completed',     value: summary.completed || 0,      cls: 'badge-green' },
          { label: 'Processing',    value: summary.processing || 0,     cls: 'badge-yellow' },
          { label: 'Failed',        value: summary.failed || 0,         cls: 'badge-red' },
          { label: 'Pending',       value: summary.pending || 0,        cls: 'badge-blue' },
        ].map(s =>
          React.createElement('div', { key: s.label, className: 'card' },
            React.createElement('div', { className: 'card-title' }, s.label),
            React.createElement('div', { className: 'stat-value' }, s.value.toLocaleString()),
          )
        )
      )
    ),

    /* System info */
    React.createElement('div', { className: 'section' },
      React.createElement('div', { className: 'section-title' }, 'System Configuration'),
      React.createElement('div', { className: 'card' },
        React.createElement('table', null,
          React.createElement('tbody', null,
            [
              ['Service Mode', data.mode],
              ['Realtime Workers', workers.realtime_workers],
              ['Realtime Queue Size', workers.realtime_queue_size],
              ['Batch Workers', workers.batch_workers],
              ['Batch Queue Size', workers.batch_queue_size],
            ].map(([k, v]) =>
              React.createElement('tr', { key: k },
                React.createElement('td', { style: { fontWeight: 600, width: '220px' } }, k),
                React.createElement('td', { className: 'mono' }, v != null ? String(v) : '—'),
              )
            )
          )
        )
      )
    ),
  );
}
