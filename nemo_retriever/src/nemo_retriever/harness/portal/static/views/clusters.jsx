/* ===== Clusters View ===== */
function ClustersView({ clusters, loading, onRefresh }) {
  const [showForm, setShowForm] = useState(false);
  const [editCluster, setEditCluster] = useState(null);
  const [detailCluster, setDetailCluster] = useState(null);
  const [healthChecking, setHealthChecking] = useState({});
  const pg = usePagination(clusters, 25);

  function handleCreate() { setEditCluster(null); setShowForm(true); }
  function handleEdit(cluster) { setEditCluster(cluster); setShowForm(true); }

  async function handleDelete(id, name) {
    if (!confirm(`Delete cluster "${name}"? This cannot be undone.`)) return;
    try {
      await fetch(`/api/clusters/${id}`, { method: "DELETE" });
      onRefresh();
    } catch {}
  }

  async function handleHealthCheck(id) {
    setHealthChecking(prev => ({ ...prev, [id]: true }));
    try {
      await fetch(`/api/clusters/${id}/health-check`, { method: "POST" });
      onRefresh();
    } catch {} finally {
      setHealthChecking(prev => ({ ...prev, [id]: false }));
    }
  }

  function statusBadge(status) {
    const colors = {
      online: { bg: 'rgba(118,185,0,0.12)', color: 'var(--nv-green)' },
      error: { bg: 'rgba(255,80,80,0.12)', color: '#ff6666' },
      unknown: { bg: 'rgba(255,255,255,0.06)', color: 'var(--nv-text-dim)' },
    };
    const c = colors[status] || colors.unknown;
    return (
      <span style={{ fontSize: '11px', fontWeight: 600, padding: '2px 8px', borderRadius: '4px', background: c.bg, color: c.color, textTransform: 'uppercase' }}>
        {status}
      </span>
    );
  }

  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <button className="btn btn-primary" onClick={handleCreate}><IconPlus /> Register Cluster</button>
        <button className="btn btn-secondary btn-icon" onClick={onRefresh} title="Refresh"><IconRefresh /></button>
      </div>
      <div className="card">
        <div style={{ overflowX: 'auto' }}>
          <table className="runs-table">
            <thead>
              <tr>
                <th>Status</th><th>Name</th><th>API Server</th><th>Namespace</th>
                <th>Auth</th><th>Run Mode</th><th>GPU Type</th>
                <th style={{ textAlign: 'right' }}>GPUs</th><th>Last Check</th><th>Tags</th><th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan="11" style={{ textAlign: 'center', padding: '60px', color: 'var(--nv-text-muted)' }}>
                  <div className="spinner spinner-lg" style={{ margin: '0 auto 12px' }}></div><div>Loading clusters…</div>
                </td></tr>
              ) : clusters.length === 0 ? (
                <tr><td colSpan="11" style={{ textAlign: 'center', padding: '60px', color: 'var(--nv-text-muted)' }}>
                  <div style={{ marginBottom: '8px', fontSize: '15px' }}>No clusters registered</div>
                  <div style={{ fontSize: '12px', color: 'var(--nv-text-dim)' }}>
                    Register a Kubernetes cluster to enable remote job execution.
                  </div>
                </td></tr>
              ) : pg.pageData.map(c => {
                const isError = c.status === 'error';
                const rowBg = isError ? 'rgba(255,80,80,0.04)' : 'transparent';
                return (
                  <tr key={c.id} style={{ background: rowBg }}>
                    <td>{statusBadge(c.status)}</td>
                    <td><span style={{ color: '#fff', fontWeight: 500, cursor: 'pointer', borderBottom: '1px dashed rgba(255,255,255,0.3)' }} onClick={() => setDetailCluster(c)}>{c.name}</span></td>
                    <td className="mono" style={{ fontSize: '12px', color: 'var(--nv-text-muted)' }}>{c.api_server_url}</td>
                    <td style={{ fontSize: '12px', color: 'var(--nv-text-muted)' }}>{c.namespace}</td>
                    <td style={{ fontSize: '11px' }}>
                      <span style={{ padding: '1px 6px', borderRadius: '4px', background: 'rgba(255,255,255,0.06)', color: 'var(--nv-text-dim)' }}>
                        {c.auth_method}
                      </span>
                    </td>
                    <td style={{ fontSize: '12px' }}>{c.default_run_mode}</td>
                    <td style={{ fontSize: '12px', color: 'var(--nv-text-muted)' }}>{c.gpu_type || "\u2014"}</td>
                    <td style={{ textAlign: 'right', fontSize: '12px' }}>{c.gpu_count || 0}</td>
                    <td style={{ fontSize: '12px', color: 'var(--nv-text-dim)' }}>{c.last_health_check ? fmtTs(c.last_health_check) : "\u2014"}</td>
                    <td>{(c.tags || []).map(t => <span key={t} className="badge" style={{ marginRight: '4px' }}>{t}</span>)}</td>
                    <td>
                      <div style={{ display: 'flex', gap: '4px' }}>
                        <button className="btn btn-ghost btn-sm" onClick={() => handleHealthCheck(c.id)}
                          disabled={healthChecking[c.id]} title="Test Connection">
                          {healthChecking[c.id] ? <span className="spinner"></span> : <IconRefresh />}
                        </button>
                        <button className="btn btn-ghost btn-sm" onClick={() => handleEdit(c)} title="Edit"><IconEdit /></button>
                        <button className="btn btn-ghost btn-sm" onClick={() => handleDelete(c.id, c.name)} title="Delete"
                          style={{ color: '#ff6666' }}><IconTrash /></button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {!loading && clusters.length > 0 && <Pagination page={pg.page} totalPages={pg.totalPages} totalItems={pg.totalItems} pageSize={pg.pageSize} onPageChange={pg.setPage} onPageSizeChange={pg.setPageSize} />}
      </div>
      {showForm && (
        <ClusterFormModal
          cluster={editCluster}
          onClose={() => setShowForm(false)}
          onSaved={() => { setShowForm(false); onRefresh(); }}
        />
      )}
      {detailCluster && (
        <ClusterDetailModal
          cluster={detailCluster}
          onClose={() => setDetailCluster(null)}
        />
      )}
    </>
  );
}

function ClusterDetailModal({ cluster, onClose }) {
  const [pods, setPods] = useState(null);
  const [configs, setConfigs] = useState(null);
  const [podsLoading, setPodsLoading] = useState(true);
  const [configsLoading, setConfigsLoading] = useState(false);
  const [podsError, setPodsError] = useState("");
  const [configsError, setConfigsError] = useState("");
  const [activeTab, setActiveTab] = useState("pods");
  const [expandedPod, setExpandedPod] = useState(null);
  const [expandedConfig, setExpandedConfig] = useState(null);

  useEffect(() => {
    setPodsLoading(true);
    setPodsError("");
    fetch(`/api/clusters/${cluster.id}/pods`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(data => setPods(data.pods || []))
      .catch(e => setPodsError(e.message))
      .finally(() => setPodsLoading(false));
  }, [cluster.id]);

  function loadConfigs() {
    if (configs !== null) return;
    setConfigsLoading(true);
    setConfigsError("");
    fetch(`/api/clusters/${cluster.id}/config`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(data => setConfigs(data.configmaps || []))
      .catch(e => setConfigsError(e.message))
      .finally(() => setConfigsLoading(false));
  }

  function podAge(startTime) {
    if (!startTime) return "\u2014";
    const ms = Date.now() - new Date(startTime).getTime();
    const secs = Math.floor(ms / 1000);
    if (secs < 60) return `${secs}s`;
    const mins = Math.floor(secs / 60);
    if (mins < 60) return `${mins}m`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ${mins % 60}m`;
    const days = Math.floor(hrs / 24);
    return `${days}d ${hrs % 24}h`;
  }

  function phaseBadge(phase) {
    const styles = {
      Running: { bg: 'rgba(118,185,0,0.12)', color: '#76b900' },
      Succeeded: { bg: 'rgba(100,180,255,0.12)', color: '#64b4ff' },
      Pending: { bg: 'rgba(255,200,0,0.12)', color: '#ffc800' },
      Failed: { bg: 'rgba(255,80,80,0.12)', color: '#ff6666' },
    };
    const s = styles[phase] || { bg: 'rgba(150,150,150,0.1)', color: '#aaa' };
    return <span style={{ fontSize: '11px', fontWeight: 600, padding: '2px 8px', borderRadius: '4px', background: s.bg, color: s.color }}>{phase}</span>;
  }

  const tabStyle = (active) => ({
    fontSize: '13px', fontWeight: 600, padding: '8px 16px', cursor: 'pointer',
    borderBottom: active ? '2px solid var(--nv-green)' : '2px solid transparent',
    color: active ? '#fff' : 'var(--nv-text-muted)', transition: 'color 0.15s',
  });

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{ maxWidth: '900px', maxHeight: '85vh', display: 'flex', flexDirection: 'column' }} onClick={e => e.stopPropagation()}>
        <div className="modal-head">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <IconCloud />
            <div>
              <h2 style={{ fontSize: '16px', fontWeight: 700, color: '#fff' }}>{cluster.name}</h2>
              <span style={{ fontSize: '12px', color: 'var(--nv-text-muted)' }}>{cluster.namespace} — {cluster.api_server_url}</span>
            </div>
          </div>
          <button className="btn btn-ghost btn-icon" onClick={onClose} style={{ borderRadius: '50%' }}><IconX /></button>
        </div>

        <div style={{ display: 'flex', gap: '0', borderBottom: '1px solid var(--nv-border)', padding: '0 24px' }}>
          <span style={tabStyle(activeTab === "pods")} onClick={() => setActiveTab("pods")}>
            Pods {pods ? `(${pods.length})` : ''}
          </span>
          <span style={tabStyle(activeTab === "config")} onClick={() => { setActiveTab("config"); loadConfigs(); }}>
            Configuration
          </span>
        </div>

        <div style={{ flex: 1, overflow: 'auto', padding: '20px 24px' }}>
          {activeTab === "pods" && (
            podsLoading ? (
              <div style={{ textAlign: 'center', padding: '40px', color: 'var(--nv-text-muted)' }}>
                <div className="spinner spinner-lg" style={{ margin: '0 auto 12px' }}></div>
                <div>Querying cluster pods…</div>
              </div>
            ) : podsError ? (
              <div style={{ padding: '20px', borderRadius: '8px', background: 'rgba(255,80,80,0.08)', border: '1px solid rgba(255,80,80,0.2)' }}>
                <div style={{ fontWeight: 600, color: '#ff6666', marginBottom: '6px' }}>Failed to fetch pods</div>
                <div style={{ fontSize: '13px', color: 'var(--nv-text-muted)' }}>{podsError}</div>
              </div>
            ) : pods && pods.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px', color: 'var(--nv-text-dim)' }}>
                No pods found in namespace <strong>{cluster.namespace}</strong>.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {(pods || []).map(pod => {
                  const isExpanded = expandedPod === pod.name;
                  return (
                    <div key={pod.name} style={{
                      borderRadius: '8px', border: '1px solid var(--nv-border)',
                      background: 'rgba(255,255,255,0.02)', overflow: 'hidden',
                    }}>
                      <div style={{
                        display: 'grid', gridTemplateColumns: '1fr auto auto auto auto',
                        gap: '12px', alignItems: 'center', padding: '12px 16px', cursor: 'pointer',
                      }} onClick={() => setExpandedPod(isExpanded ? null : pod.name)}>
                        <div>
                          <div style={{ fontSize: '13px', fontWeight: 600, color: '#fff', marginBottom: '2px' }}>
                            {pod.app_label || pod.name}
                          </div>
                          <div className="mono" style={{ fontSize: '11px', color: 'var(--nv-text-dim)' }}>{pod.name}</div>
                        </div>
                        <div>{phaseBadge(pod.phase)}</div>
                        <div style={{ fontSize: '12px', color: 'var(--nv-text-muted)', minWidth: '60px', textAlign: 'right' }}>
                          {podAge(pod.start_time)}
                        </div>
                        <div style={{ fontSize: '11px', color: 'var(--nv-text-dim)', minWidth: '80px' }}>
                          {pod.version || "\u2014"}
                        </div>
                        <span style={{ fontSize: '10px', color: 'var(--nv-text-dim)', transform: isExpanded ? 'rotate(90deg)' : 'rotate(0)', transition: 'transform 0.15s' }}>{"\u25B6"}</span>
                      </div>

                      {isExpanded && (
                        <div style={{ padding: '0 16px 16px', borderTop: '1px solid var(--nv-border)' }}>
                          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', padding: '12px 0' }}>
                            <div>
                              <div style={{ fontSize: '11px', fontWeight: 500, color: 'var(--nv-text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>Node</div>
                              <div style={{ fontSize: '13px', color: '#fff' }}>{pod.node || "\u2014"}</div>
                            </div>
                            <div>
                              <div style={{ fontSize: '11px', fontWeight: 500, color: 'var(--nv-text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>Component</div>
                              <div style={{ fontSize: '13px', color: '#fff' }}>{pod.component || "\u2014"}</div>
                            </div>
                            <div>
                              <div style={{ fontSize: '11px', fontWeight: 500, color: 'var(--nv-text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>Started</div>
                              <div style={{ fontSize: '13px', color: '#fff' }}>{pod.start_time ? fmtTs(pod.start_time) : "\u2014"}</div>
                            </div>
                            <div>
                              <div style={{ fontSize: '11px', fontWeight: 500, color: 'var(--nv-text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>Namespace</div>
                              <div style={{ fontSize: '13px', color: '#fff' }}>{pod.namespace}</div>
                            </div>
                          </div>

                          {pod.resources && (pod.resources.requests && Object.keys(pod.resources.requests).length > 0 || pod.resources.limits && Object.keys(pod.resources.limits).length > 0) && (
                            <div style={{ marginTop: '8px' }}>
                              <div style={{ fontSize: '11px', fontWeight: 500, color: 'var(--nv-text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>Resources</div>
                              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                                {pod.resources.requests && Object.keys(pod.resources.requests).length > 0 && (
                                  <div style={{ padding: '8px 12px', borderRadius: '6px', background: 'rgba(118,185,0,0.06)', border: '1px solid rgba(118,185,0,0.15)' }}>
                                    <div style={{ fontSize: '10px', color: 'var(--nv-green)', fontWeight: 600, marginBottom: '4px' }}>REQUESTS</div>
                                    {Object.entries(pod.resources.requests).map(([k, v]) => (
                                      <div key={k} style={{ fontSize: '12px', color: 'var(--nv-text-muted)' }}>{k}: <span style={{ color: '#fff' }}>{v}</span></div>
                                    ))}
                                  </div>
                                )}
                                {pod.resources.limits && Object.keys(pod.resources.limits).length > 0 && (
                                  <div style={{ padding: '8px 12px', borderRadius: '6px', background: 'rgba(255,165,0,0.06)', border: '1px solid rgba(255,165,0,0.15)' }}>
                                    <div style={{ fontSize: '10px', color: '#ffa500', fontWeight: 600, marginBottom: '4px' }}>LIMITS</div>
                                    {Object.entries(pod.resources.limits).map(([k, v]) => (
                                      <div key={k} style={{ fontSize: '12px', color: 'var(--nv-text-muted)' }}>{k}: <span style={{ color: '#fff' }}>{v}</span></div>
                                    ))}
                                  </div>
                                )}
                              </div>
                            </div>
                          )}

                          {pod.containers && pod.containers.length > 0 && (
                            <div style={{ marginTop: '12px' }}>
                              <div style={{ fontSize: '11px', fontWeight: 500, color: 'var(--nv-text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>Containers</div>
                              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                                {pod.containers.map(ct => (
                                  <div key={ct.name} style={{ display: 'grid', gridTemplateColumns: '1fr 2fr auto auto', gap: '8px', alignItems: 'center', padding: '6px 10px', borderRadius: '6px', background: 'rgba(0,0,0,0.2)' }}>
                                    <span style={{ fontSize: '12px', fontWeight: 500, color: '#fff' }}>{ct.name}</span>
                                    <span className="mono" style={{ fontSize: '11px', color: 'var(--nv-text-dim)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{ct.image}</span>
                                    <span style={{ fontSize: '11px', color: ct.ready ? '#76b900' : '#ffa500' }}>{ct.ready ? "Ready" : "Not Ready"}</span>
                                    {ct.restart_count > 0 && <span style={{ fontSize: '11px', color: '#ff6666' }}>{ct.restart_count} restart{ct.restart_count !== 1 ? 's' : ''}</span>}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {pod.labels && Object.keys(pod.labels).length > 0 && (
                            <div style={{ marginTop: '12px' }}>
                              <div style={{ fontSize: '11px', fontWeight: 500, color: 'var(--nv-text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>Labels</div>
                              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                                {Object.entries(pod.labels).map(([k, v]) => (
                                  <span key={k} style={{ fontSize: '10px', padding: '2px 6px', borderRadius: '4px', background: 'rgba(255,255,255,0.06)', color: 'var(--nv-text-dim)' }}>
                                    {k}={v}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )
          )}

          {activeTab === "config" && (
            configsLoading ? (
              <div style={{ textAlign: 'center', padding: '40px', color: 'var(--nv-text-muted)' }}>
                <div className="spinner spinner-lg" style={{ margin: '0 auto 12px' }}></div>
                <div>Fetching configuration…</div>
              </div>
            ) : configsError ? (
              <div style={{ padding: '20px', borderRadius: '8px', background: 'rgba(255,80,80,0.08)', border: '1px solid rgba(255,80,80,0.2)' }}>
                <div style={{ fontWeight: 600, color: '#ff6666', marginBottom: '6px' }}>Failed to fetch config</div>
                <div style={{ fontSize: '13px', color: 'var(--nv-text-muted)' }}>{configsError}</div>
              </div>
            ) : configs && configs.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px', color: 'var(--nv-text-dim)' }}>
                No configmaps found in namespace <strong>{cluster.namespace}</strong>.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {(configs || []).map(cm => {
                  const isExpanded = expandedConfig === cm.name;
                  return (
                    <div key={cm.name} style={{
                      borderRadius: '8px', border: '1px solid var(--nv-border)',
                      background: cm.helm_managed ? 'rgba(118,185,0,0.02)' : 'rgba(255,255,255,0.02)',
                      overflow: 'hidden',
                    }}>
                      <div style={{
                        display: 'grid', gridTemplateColumns: '1fr auto auto auto',
                        gap: '12px', alignItems: 'center', padding: '12px 16px', cursor: 'pointer',
                      }} onClick={() => setExpandedConfig(isExpanded ? null : cm.name)}>
                        <div>
                          <div style={{ fontSize: '13px', fontWeight: 600, color: '#fff', marginBottom: '2px' }}>{cm.name}</div>
                          {cm.app && <span style={{ fontSize: '11px', color: 'var(--nv-text-dim)' }}>{cm.app}</span>}
                        </div>
                        {cm.helm_managed && (
                          <span style={{ fontSize: '10px', padding: '2px 6px', borderRadius: '4px', background: 'rgba(118,185,0,0.12)', color: '#76b900', fontWeight: 600 }}>HELM</span>
                        )}
                        <span style={{ fontSize: '11px', color: 'var(--nv-text-dim)' }}>{cm.data_keys.length} key{cm.data_keys.length !== 1 ? 's' : ''}</span>
                        <span style={{ fontSize: '10px', color: 'var(--nv-text-dim)', transform: isExpanded ? 'rotate(90deg)' : 'rotate(0)', transition: 'transform 0.15s' }}>{"\u25B6"}</span>
                      </div>
                      {isExpanded && (
                        <div style={{ padding: '0 16px 16px', borderTop: '1px solid var(--nv-border)' }}>
                          {cm.chart && (
                            <div style={{ fontSize: '12px', color: 'var(--nv-text-muted)', padding: '8px 0' }}>
                              Chart: <span style={{ color: '#fff' }}>{cm.chart}</span>
                              {cm.created_at && <span style={{ marginLeft: '16px' }}>Created: {fmtTs(cm.created_at)}</span>}
                            </div>
                          )}
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginTop: '8px' }}>
                            {Object.entries(cm.data || {}).map(([key, value]) => (
                              <div key={key} style={{ borderRadius: '6px', background: 'rgba(0,0,0,0.2)', padding: '8px 12px' }}>
                                <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--nv-green)', marginBottom: '4px' }}>{key}</div>
                                <pre className="mono" style={{
                                  fontSize: '11px', color: 'var(--nv-text-muted)', margin: 0,
                                  whiteSpace: 'pre-wrap', wordBreak: 'break-all', maxHeight: '200px', overflow: 'auto',
                                }}>{value}</pre>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )
          )}
        </div>

        <div className="modal-foot" style={{ justifyContent: 'flex-end' }}>
          <button className="btn btn-secondary" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}


function ClusterFormModal({ cluster, onClose, onSaved }) {
  const isEdit = !!cluster;
  const [name, setName] = useState(cluster?.name || "");
  const [description, setDescription] = useState(cluster?.description || "");
  const [apiServerUrl, setApiServerUrl] = useState(cluster?.api_server_url || "");
  const [namespace, setNamespace] = useState(cluster?.namespace || "default");
  const [authMethod, setAuthMethod] = useState(cluster?.auth_method || "kubeconfig");
  const [kubeconfigContext, setKubeconfigContext] = useState(cluster?.kubeconfig_context || "");
  const [kubeconfigData, setKubeconfigData] = useState(cluster?.kubeconfig_data || "");
  const [serviceAccountToken, setServiceAccountToken] = useState(cluster?.service_account_token || "");
  const [caCertData, setCaCertData] = useState(cluster?.ca_cert_data || "");
  const [defaultRunMode, setDefaultRunMode] = useState(cluster?.default_run_mode || "batch");
  const [defaultImage, setDefaultImage] = useState(cluster?.default_image || "");
  const [nodeSelector, setNodeSelector] = useState(
    cluster?.node_selector ? (typeof cluster.node_selector === 'string' ? cluster.node_selector : JSON.stringify(cluster.node_selector, null, 2)) : ""
  );
  const [nodeHostnames, setNodeHostnames] = useState((cluster?.node_hostnames || []).join(", "));
  const [gpuType, setGpuType] = useState(cluster?.gpu_type || "");
  const [gpuCount, setGpuCount] = useState(cluster?.gpu_count || 0);
  const [tags, setTags] = useState((cluster?.tags || []).join(", "));
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [testResult, setTestResult] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!name.trim() || !apiServerUrl.trim()) return;
    setSubmitting(true);
    setError("");
    try {
      let parsedNodeSelector = null;
      if (nodeSelector.trim()) {
        try { parsedNodeSelector = JSON.parse(nodeSelector); } catch { parsedNodeSelector = nodeSelector; }
      }
      const payload = {
        name: name.trim(),
        description: description.trim() || null,
        api_server_url: apiServerUrl.trim(),
        namespace: namespace.trim() || "default",
        auth_method: authMethod,
        kubeconfig_context: kubeconfigContext.trim() || null,
        kubeconfig_data: kubeconfigData.trim() || null,
        service_account_token: serviceAccountToken.trim() || null,
        ca_cert_data: caCertData.trim() || null,
        default_run_mode: defaultRunMode,
        default_image: defaultImage.trim() || null,
        node_selector: parsedNodeSelector,
        node_hostnames: nodeHostnames.split(",").map(h => h.trim()).filter(Boolean),
        gpu_type: gpuType.trim() || null,
        gpu_count: gpuCount,
        tags: tags.split(",").map(t => t.trim()).filter(Boolean),
      };
      const url = isEdit ? `/api/clusters/${cluster.id}` : "/api/clusters";
      const method = isEdit ? "PUT" : "POST";
      const res = await fetch(url, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      onSaved();
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  async function handleTestConnection() {
    if (!isEdit) return;
    setTestResult(null);
    try {
      const res = await fetch(`/api/clusters/${cluster.id}/health-check`, { method: "POST" });
      const data = await res.json();
      setTestResult(data);
    } catch (err) {
      setTestResult({ status: "error", message: err.message });
    }
  }

  const labelStyle = { display: 'block', fontSize: '12px', fontWeight: 500, color: 'var(--nv-text-muted)', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.04em' };
  const hintStyle = { fontSize: '11px', color: 'var(--nv-text-dim)', marginTop: '4px', lineHeight: '1.5' };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{ maxWidth: '620px', maxHeight: '85vh', overflow: 'auto' }} onClick={e => e.stopPropagation()}>
        <div className="modal-head">
          <h2 style={{ fontSize: '16px', fontWeight: 700, color: '#fff' }}>{isEdit ? "Edit Cluster" : "Register Cluster"}</h2>
          <button className="btn btn-ghost btn-icon" onClick={onClose} style={{ borderRadius: '50%' }}><IconX /></button>
        </div>
        <form onSubmit={handleSubmit}>
          <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {error && <div style={{ padding: '10px 14px', borderRadius: '6px', background: 'rgba(255,80,80,0.1)', color: '#ff6666', fontSize: '13px' }}>{error}</div>}

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div>
                <label style={labelStyle}>Name</label>
                <input className="input" style={{ width: '100%' }} value={name} onChange={e => setName(e.target.value)} placeholder="e.g. prod-dgx-cluster" required />
              </div>
              <div>
                <label style={labelStyle}>Namespace</label>
                <input className="input" style={{ width: '100%' }} value={namespace} onChange={e => setNamespace(e.target.value)} placeholder="default" />
              </div>
            </div>

            <div>
              <label style={labelStyle}>API Server URL</label>
              <input className="input" style={{ width: '100%' }} value={apiServerUrl} onChange={e => setApiServerUrl(e.target.value)} placeholder="https://k8s-api.example.com:6443" required />
            </div>

            <div>
              <label style={labelStyle}>Description</label>
              <input className="input" style={{ width: '100%' }} value={description} onChange={e => setDescription(e.target.value)} placeholder="Optional description" />
            </div>

            <div>
              <label style={labelStyle}>Authentication Method</label>
              <select className="select" style={{ width: '100%' }} value={authMethod} onChange={e => setAuthMethod(e.target.value)}>
                <option value="kubeconfig">Kubeconfig</option>
                <option value="token">Service Account Token</option>
              </select>
            </div>

            {authMethod === "kubeconfig" && (
              <>
                <div>
                  <label style={labelStyle}>Kubeconfig Context</label>
                  <input className="input" style={{ width: '100%' }} value={kubeconfigContext} onChange={e => setKubeconfigContext(e.target.value)} placeholder="e.g. my-cluster-context" />
                  <div style={hintStyle}>Context name from kubeconfig. Leave empty to use the default context.</div>
                </div>
                <div>
                  <label style={labelStyle}>Kubeconfig Data (YAML)</label>
                  <textarea className="input" style={{ width: '100%', minHeight: '100px', fontFamily: 'var(--font-mono)' }} value={kubeconfigData} onChange={e => setKubeconfigData(e.target.value)} placeholder="Paste kubeconfig YAML content here..." />
                  <div style={hintStyle}>Full kubeconfig content. Stored securely in the portal database.</div>
                </div>
              </>
            )}

            {authMethod === "token" && (
              <>
                <div>
                  <label style={labelStyle}>Service Account Token</label>
                  <textarea className="input" style={{ width: '100%', minHeight: '60px', fontFamily: 'var(--font-mono)' }} value={serviceAccountToken} onChange={e => setServiceAccountToken(e.target.value)} placeholder="eyJhbGciOiJSUzI1NiIs..." />
                </div>
                <div>
                  <label style={labelStyle}>CA Certificate (PEM)</label>
                  <textarea className="input" style={{ width: '100%', minHeight: '60px', fontFamily: 'var(--font-mono)' }} value={caCertData} onChange={e => setCaCertData(e.target.value)} placeholder="-----BEGIN CERTIFICATE-----..." />
                </div>
              </>
            )}

            <div style={{ borderTop: '1px solid var(--nv-border)', paddingTop: '16px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div>
                  <label style={labelStyle}>Default Run Mode</label>
                  <select className="select" style={{ width: '100%' }} value={defaultRunMode} onChange={e => setDefaultRunMode(e.target.value)}>
                    <option value="batch">Batch (Ray)</option>
                    <option value="inprocess">In-Process</option>
                    <option value="service">Service</option>
                  </select>
                </div>
                <div>
                  <label style={labelStyle}>Default Container Image</label>
                  <input className="input" style={{ width: '100%' }} value={defaultImage} onChange={e => setDefaultImage(e.target.value)} placeholder="nvcr.io/nvidia/nemo-retriever:latest" />
                </div>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '16px' }}>
              <div>
                <label style={labelStyle}>GPU Type</label>
                <input className="input" style={{ width: '100%' }} value={gpuType} onChange={e => setGpuType(e.target.value)} placeholder="e.g. A100" />
              </div>
              <div>
                <label style={labelStyle}>GPU Count</label>
                <input className="input" type="number" min="0" style={{ width: '100%' }} value={gpuCount} onChange={e => setGpuCount(parseInt(e.target.value, 10) || 0)} />
              </div>
              <div>
                <label style={labelStyle}>Tags</label>
                <input className="input" style={{ width: '100%' }} value={tags} onChange={e => setTags(e.target.value)} placeholder="prod, dgx" />
              </div>
            </div>

            <div>
              <label style={labelStyle}>Node Hostnames</label>
              <input className="input" style={{ width: '100%' }} value={nodeHostnames} onChange={e => setNodeHostnames(e.target.value)} placeholder="gpu-node-01, gpu-node-02" />
              <div style={hintStyle}>Comma-separated hostnames of machines running this cluster. Local runners on these hosts will be excluded from job scheduling while the cluster has active jobs (prevents GPU memory contention).</div>
            </div>

            <div>
              <label style={labelStyle}>Node Selector (JSON)</label>
              <input className="input" style={{ width: '100%' }} value={nodeSelector} onChange={e => setNodeSelector(e.target.value)} placeholder='{"nvidia.com/gpu.product": "A100"}' />
              <div style={hintStyle}>Optional JSON object for Kubernetes node selector constraints.</div>
            </div>

            {isEdit && (
              <div style={{ borderTop: '1px solid var(--nv-border)', paddingTop: '16px' }}>
                <button type="button" className="btn btn-secondary" onClick={handleTestConnection}>
                  Test Connection
                </button>
                {testResult && (
                  <div style={{ marginTop: '8px', padding: '10px 14px', borderRadius: '6px', background: testResult.status === 'online' ? 'rgba(118,185,0,0.08)' : 'rgba(255,80,80,0.08)', fontSize: '12px' }}>
                    <strong style={{ color: testResult.status === 'online' ? 'var(--nv-green)' : '#ff6666' }}>{testResult.status}</strong>
                    {testResult.message && <span style={{ marginLeft: '8px', color: 'var(--nv-text-dim)' }}>{testResult.message}</span>}
                  </div>
                )}
              </div>
            )}
          </div>
          <div className="modal-foot">
            <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button type="submit" disabled={submitting || !name.trim() || !apiServerUrl.trim()} className="btn btn-primary" style={{ flex: 1, justifyContent: 'center' }}>
              {submitting ? <><span className="spinner" style={{ marginRight: '8px' }}></span>Saving…</> : (isEdit ? "Update Cluster" : "Register Cluster")}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
