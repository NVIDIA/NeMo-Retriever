/* ===== Alerts View ===== */
const METRIC_LABELS = { pages_per_sec:"Pages / sec", recall_1:"Recall@1", recall_5:"Recall@5", recall_10:"Recall@10", ingest_secs:"Ingest Time (s)", pages:"Pages", files:"Files", system:"System", runner_status:"Runner Status" };
const OPERATOR_LABELS = { "<":"less than", "<=":"at most", ">":"greater than", ">=":"at least", "==":"equal to", "!=":"not equal to" };

function AlertsView({ alertRules, alertEvents, alertRulesLoading, alertEventsLoading, onRefresh, onSelectRun, githubRepoUrl }) {
  const [tab, setTab] = useState("events");
  const [showForm, setShowForm] = useState(false);
  const [editRule, setEditRule] = useState(null);
  const [filterRule, setFilterRule] = useState("");
  const [filterAck, setFilterAck] = useState("unack");
  const pgRules = usePagination(alertRules, 25);

  function handleCreateRule() { setEditRule(null); setShowForm(true); }
  function handleEditRule(r) { setEditRule(r); setShowForm(true); }
  async function handleDeleteRule(id, name) {
    if (!confirm(`Delete alert rule "${name}"? This cannot be undone.`)) return;
    try { await fetch(`/api/alert-rules/${id}`, { method:"DELETE" }); onRefresh(); } catch {}
  }
  async function handleToggleRule(rule) {
    try {
      await fetch(`/api/alert-rules/${rule.id}`, {
        method:"PUT", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({ enabled: !rule.enabled }),
      });
      onRefresh();
    } catch {}
  }
  async function handleAcknowledge(eventId) {
    try { await fetch(`/api/alert-events/${eventId}/acknowledge`, { method:"POST" }); onRefresh(); } catch {}
  }
  async function handleAcknowledgeAll() {
    if (!confirm("Acknowledge all unread alerts?")) return;
    try { await fetch("/api/alert-events/acknowledge-all", { method:"POST" }); onRefresh(); } catch {}
  }

  const filteredEvents = useMemo(() => {
    return alertEvents.filter(e => {
      if (filterRule && e.rule_id !== parseInt(filterRule)) return false;
      if (filterAck === "unack" && e.acknowledged) return false;
      if (filterAck === "ack" && !e.acknowledged) return false;
      return true;
    });
  }, [alertEvents, filterRule, filterAck]);
  const pgEvents = usePagination(filteredEvents, 25);

  const ruleMap = useMemo(() => {
    const m = {};
    alertRules.forEach(r => { m[r.id] = r; });
    return m;
  }, [alertRules]);

  const unackCount = alertEvents.filter(e => !e.acknowledged).length;

  const labelStyle = {display:'block',fontSize:'10px',fontWeight:600,color:'var(--nv-text-dim)',marginBottom:'4px',textTransform:'uppercase',letterSpacing:'0.05em'};

  return (
    <>
      {/* Tab toggle */}
      <div style={{display:'flex',gap:'8px',marginBottom:'20px',alignItems:'center'}}>
        <div style={{display:'flex',gap:'2px',background:'var(--nv-bg)',borderRadius:'8px',padding:'2px',border:'1px solid var(--nv-border)'}}>
          {[{k:'events',l:'Alert History'},{k:'rules',l:'Alert Rules'}].map(t=>(
            <button key={t.k} className="btn btn-sm" onClick={()=>setTab(t.k)}
              style={{borderRadius:'6px',padding:'6px 18px',fontSize:'13px',
                background:tab===t.k?'var(--nv-green)':'transparent',
                color:tab===t.k?'#000':'var(--nv-text-muted)',
                fontWeight:tab===t.k?700:500,border:'none'}}>
              {t.l}
              {t.k==='events' && unackCount > 0 && (
                <span style={{marginLeft:'6px',background:tab===t.k?'rgba(0,0,0,0.2)':'#ff5050',color:tab===t.k?'#000':'#fff',
                  fontSize:'10px',fontWeight:700,padding:'1px 6px',borderRadius:'100px'}}>{unackCount}</span>
              )}
            </button>
          ))}
        </div>
        <div style={{flex:1}}></div>
        <button className="btn btn-secondary btn-icon" onClick={onRefresh} title="Refresh"><IconRefresh /></button>
      </div>

      {/* Events Tab */}
      {tab === "events" && (
        <>
          <div style={{display:'flex',gap:'12px',alignItems:'flex-end',marginBottom:'16px',flexWrap:'wrap'}}>
            <div>
              <div style={labelStyle}>Rule</div>
              <select className="select" value={filterRule} onChange={e=>setFilterRule(e.target.value)} style={{minWidth:'180px'}}>
                <option value="">All rules</option>
                {alertRules.map(r=><option key={r.id} value={r.id}>{r.name}</option>)}
              </select>
            </div>
            <div>
              <div style={labelStyle}>Status</div>
              <select className="select" value={filterAck} onChange={e=>setFilterAck(e.target.value)} style={{minWidth:'130px'}}>
                <option value="all">All</option>
                <option value="unack">Unacknowledged</option>
                <option value="ack">Acknowledged</option>
              </select>
            </div>
            <div style={{flex:1}}></div>
            {unackCount > 0 && (
              <button className="btn btn-secondary btn-sm" onClick={handleAcknowledgeAll}>
                <IconCheck /> Acknowledge All ({unackCount})
              </button>
            )}
          </div>

          <div className="card">
            <div style={{overflowX:'auto'}}>
              <table className="runs-table">
                <thead>
                  <tr>
                    <th style={{width:'40px'}}></th>
                    <th>Alert</th><th>Metric</th><th>Value</th>
                    <th>Threshold</th><th>Dataset</th><th>Commit</th>
                    <th>Host</th><th>Time</th><th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {alertEventsLoading ? (
                    <tr><td colSpan="10" style={{textAlign:'center',padding:'60px',color:'var(--nv-text-muted)'}}>
                      <div className="spinner spinner-lg" style={{margin:'0 auto 12px'}}></div><div>Loading alerts…</div>
                    </td></tr>
                  ) : filteredEvents.length === 0 ? (
                    <tr><td colSpan="10" style={{textAlign:'center',padding:'60px',color:'var(--nv-text-muted)'}}>
                      <div style={{marginBottom:'8px',fontSize:'15px'}}>{filterAck==='unack'?'No unacknowledged alerts':'No alerts found'}</div>
                      <div style={{fontSize:'12px',color:'var(--nv-text-dim)'}}>Alerts are generated when a run metric violates a rule threshold.</div>
                    </td></tr>
                  ) : pgEvents.pageData.map(ev => {
                    const rule = ruleMap[ev.rule_id];
                    return (
                      <tr key={ev.id} style={{background:!ev.acknowledged?'rgba(255,60,60,0.04)':'inherit',cursor:'default'}}>
                        <td>
                          {!ev.acknowledged
                            ? <span style={{display:'inline-block',width:'8px',height:'8px',borderRadius:'50%',background:'#ff5050',boxShadow:'0 0 6px rgba(255,80,80,0.5)'}}></span>
                            : <span style={{display:'inline-block',width:'8px',height:'8px',borderRadius:'50%',background:'#333'}}></span>}
                        </td>
                        <td>
                          <div style={{color:'#fff',fontWeight:500,fontSize:'13px'}}>{rule?.name || (ev.rule_id === 0 ? "System Alert" : `Rule #${ev.rule_id}`)}</div>
                          <div style={{fontSize:'11px',color:'var(--nv-text-dim)',marginTop:'2px',maxWidth:'220px',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}} title={ev.message}>{ev.message}</div>
                        </td>
                        <td><span className="badge badge-na">{METRIC_LABELS[ev.metric]||ev.metric}</span></td>
                        <td className="mono" style={{color:'#ff5050',fontWeight:600,fontSize:'13px'}}>{(ev.metric === "system" || ev.operator === "system") ? "\u2014" : (ev.metric_value != null ? ev.metric_value.toFixed(4) : "\u2014")}</td>
                        <td className="mono" style={{color:'var(--nv-text-muted)',fontSize:'13px'}}>{(ev.metric === "system" || ev.operator === "system") ? "\u2014" : `${ev.operator} ${ev.threshold}`}</td>
                        <td style={{color:'var(--nv-text-muted)',fontSize:'12px'}}>{ev.dataset || "\u2014"}</td>
                        <td>
                          <CommitLink sha={ev.git_commit} repoUrl={githubRepoUrl} />
                        </td>
                        <td style={{color:'var(--nv-text-muted)',fontSize:'12px',whiteSpace:'nowrap'}}>{ev.hostname||"\u2014"}</td>
                        <td style={{color:'var(--nv-text-muted)',fontSize:'12px',whiteSpace:'nowrap'}}>{fmtTs(ev.created_at)}</td>
                        <td>
                          <div style={{display:'flex',gap:'6px',flexWrap:'nowrap'}}>
                            {ev.run_id && ev.run_id !== 0 ? (
                              <button className="btn btn-secondary btn-sm" onClick={()=>onSelectRun(ev.run_id)} title="View Run">
                                <IconExternalLink /> Run #{ev.run_id}
                              </button>
                            ) : (
                              <span className="badge badge-na" style={{fontSize:'11px'}}>System</span>
                            )}
                            {!ev.acknowledged && (
                              <button className="btn btn-sm" onClick={()=>handleAcknowledge(ev.id)} title="Acknowledge"
                                style={{background:'rgba(118,185,0,0.1)',color:'var(--nv-green)',border:'1px solid rgba(118,185,0,0.2)'}}>
                                <IconCheck />
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <Pagination page={pgEvents.page} totalPages={pgEvents.totalPages} totalItems={pgEvents.totalItems}
              pageSize={pgEvents.pageSize} onPageChange={pgEvents.setPage} onPageSizeChange={pgEvents.setPageSize} />
          </div>
        </>
      )}

      {/* Rules Tab */}
      {tab === "rules" && (
        <>
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'16px'}}>
            <button className="btn btn-primary" onClick={handleCreateRule}><IconPlus /> Create Alert Rule</button>
          </div>

          <div className="card">
            <div style={{overflowX:'auto'}}>
              <table className="runs-table">
                <thead>
                  <tr>
                    <th style={{width:'60px'}}>Enabled</th>
                    <th>Name</th><th>Condition</th><th>Scope</th>
                    <th>Created</th><th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {alertRulesLoading ? (
                    <tr><td colSpan="6" style={{textAlign:'center',padding:'60px',color:'var(--nv-text-muted)'}}>
                      <div className="spinner spinner-lg" style={{margin:'0 auto 12px'}}></div><div>Loading rules…</div>
                    </td></tr>
                  ) : alertRules.length === 0 ? (
                    <tr><td colSpan="6" style={{textAlign:'center',padding:'60px',color:'var(--nv-text-muted)'}}>
                      <div style={{marginBottom:'8px',fontSize:'15px'}}>No alert rules defined</div>
                      <div style={{fontSize:'12px',color:'var(--nv-text-dim)'}}>Create a rule to get notified when a metric goes out of range.</div>
                    </td></tr>
                  ) : pgRules.pageData.map(r => (
                    <tr key={r.id} style={{cursor:'default'}}>
                      <td>
                        <label className="toggle">
                          <input type="checkbox" checked={r.enabled} onChange={()=>handleToggleRule(r)} />
                          <span className="toggle-slider"></span>
                        </label>
                      </td>
                      <td>
                        <div style={{display:'flex',alignItems:'center',gap:'6px'}}>
                          <span style={{color:'#fff',fontWeight:500}}>{r.name}</span>
                          {r.slack_notify && (
                            <span title="Slack notifications enabled" style={{
                              padding:'1px 6px',borderRadius:'4px',fontSize:'9px',fontWeight:700,
                              textTransform:'uppercase',letterSpacing:'0.05em',
                              background:'rgba(74,21,75,0.2)',color:'#e0a0e0',
                              border:'1px solid rgba(74,21,75,0.3)',
                            }}>Slack</span>
                          )}
                        </div>
                        {r.description && <div style={{fontSize:'11px',color:'var(--nv-text-dim)',marginTop:'2px'}}>{r.description}</div>}
                      </td>
                      <td>
                        <span style={{display:'inline-flex',alignItems:'center',gap:'6px',fontSize:'13px'}}>
                          <span className="badge badge-na">{METRIC_LABELS[r.metric]||r.metric}</span>
                          <span style={{color:'var(--nv-text-muted)',fontWeight:600}}>{r.operator}</span>
                          <span className="mono" style={{color:'var(--nv-green)',fontWeight:600}}>{r.threshold}</span>
                        </span>
                      </td>
                      <td style={{fontSize:'12px',color:'var(--nv-text-muted)'}}>
                        {r.dataset_filter || r.preset_filter
                          ? [r.dataset_filter && `Dataset: ${r.dataset_filter}`, r.preset_filter && `Preset: ${r.preset_filter}`].filter(Boolean).join(", ")
                          : "All runs"}
                      </td>
                      <td style={{color:'var(--nv-text-muted)',fontSize:'12px',whiteSpace:'nowrap'}}>{fmtTs(r.created_at)}</td>
                      <td>
                        <div style={{display:'flex',gap:'6px',flexWrap:'nowrap'}}>
                          <button className="btn btn-secondary btn-sm" onClick={()=>handleEditRule(r)} title="Edit"><IconEdit /> Edit</button>
                          <button className="btn btn-sm" onClick={()=>handleDeleteRule(r.id,r.name)} title="Delete"
                            style={{background:'rgba(255,80,80,0.1)',color:'#ff5050',border:'1px solid rgba(255,80,80,0.2)'}}><IconTrash /> Delete</button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Pagination page={pgRules.page} totalPages={pgRules.totalPages} totalItems={pgRules.totalItems}
              pageSize={pgRules.pageSize} onPageChange={pgRules.setPage} onPageSizeChange={pgRules.setPageSize} />
          </div>
        </>
      )}

      {showForm && (
        <AlertRuleFormModal
          rule={editRule}
          onClose={()=>setShowForm(false)}
          onSaved={()=>{setShowForm(false);onRefresh();}}
        />
      )}
    </>
  );
}

function AlertRuleFormModal({ rule, onClose, onSaved }) {
  const isEdit = !!rule;
  const [metrics, setMetrics] = useState([]);
  const [operators, setOperators] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [presets, setPresets] = useState([]);
  const [form, setForm] = useState({
    name: rule?.name || "",
    description: rule?.description || "",
    metric: rule?.metric || "pages_per_sec",
    operator: rule?.operator || "<",
    threshold: rule?.threshold ?? "",
    dataset_filter: rule?.dataset_filter || "",
    preset_filter: rule?.preset_filter || "",
    enabled: rule?.enabled ?? true,
    slack_notify: rule?.slack_notify ?? false,
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/alert-metrics").then(r=>r.json()).then(d=>{
      setMetrics(d.metrics||[]);
      setOperators(d.operators||[]);
    }).catch(()=>{});
    fetch("/api/config").then(r=>r.json()).then(cfg=>{
      setDatasets(cfg.datasets||[]);
      setPresets(cfg.presets||[]);
    }).catch(()=>{});
  }, []);

  function set(k,v) { setForm(f=>({...f,[k]:v})); }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!form.name.trim() || form.threshold === "") return;
    setSaving(true);
    setError("");
    const payload = {
      name: form.name.trim(),
      description: form.description.trim() || null,
      metric: form.metric,
      operator: form.operator,
      threshold: parseFloat(form.threshold),
      dataset_filter: form.dataset_filter || null,
      preset_filter: form.preset_filter || null,
      enabled: form.enabled,
      slack_notify: form.slack_notify,
    };
    try {
      const url = isEdit ? `/api/alert-rules/${rule.id}` : "/api/alert-rules";
      const method = isEdit ? "PUT" : "POST";
      const res = await fetch(url, { method, headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload) });
      if (!res.ok) { const d = await res.json().catch(()=>({})); throw new Error(d.detail||`HTTP ${res.status}`); }
      onSaved();
    } catch (err) { setError(err.message); } finally { setSaving(false); }
  }

  const labelStyle = {display:'block',fontSize:'12px',fontWeight:500,color:'var(--nv-text-muted)',marginBottom:'6px',textTransform:'uppercase',letterSpacing:'0.04em'};

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{maxWidth:'540px'}} onClick={e=>e.stopPropagation()}>
        <div className="modal-head">
          <h2 style={{fontSize:'16px',fontWeight:700,color:'#fff'}}>{isEdit?"Edit Alert Rule":"Create Alert Rule"}</h2>
          <button className="btn btn-ghost btn-icon" onClick={onClose} style={{borderRadius:'50%'}}><IconX /></button>
        </div>
        <form onSubmit={handleSubmit}>
          <div style={{padding:'24px',display:'flex',flexDirection:'column',gap:'16px',maxHeight:'65vh',overflowY:'auto'}}>
            {error && (
              <div style={{padding:'10px 14px',borderRadius:'8px',background:'rgba(255,50,50,0.08)',border:'1px solid rgba(255,50,50,0.2)',color:'#ff5050',fontSize:'13px'}}>{error}</div>
            )}
            <div>
              <label style={labelStyle}>Rule Name *</label>
              <input className="input" style={{width:'100%'}} value={form.name} onChange={e=>set('name',e.target.value)} placeholder="e.g. PPS regression threshold" required />
            </div>
            <div>
              <label style={labelStyle}>Description</label>
              <input className="input" style={{width:'100%'}} value={form.description} onChange={e=>set('description',e.target.value)} placeholder="Optional description" />
            </div>

            {/* Condition */}
            <div style={{borderTop:'1px solid var(--nv-border)',paddingTop:'16px'}}>
              <div style={{fontSize:'13px',fontWeight:600,color:'var(--nv-green)',marginBottom:'12px',textTransform:'uppercase',letterSpacing:'0.05em'}}>Condition</div>
              <div style={{display:'flex',alignItems:'center',gap:'8px',fontSize:'14px',color:'#fff',padding:'12px 16px',background:'var(--nv-bg)',borderRadius:'8px',border:'1px solid var(--nv-border)'}}>
                <span style={{color:'var(--nv-text-muted)'}}>Alert when</span>
                <select className="select" value={form.metric} onChange={e=>set('metric',e.target.value)} style={{minWidth:'130px'}}>
                  {metrics.filter(m=>m!=='system').map(m=><option key={m} value={m}>{METRIC_LABELS[m]||m}</option>)}
                </select>
                <span style={{color:'var(--nv-text-muted)'}}>is</span>
                <select className="select" value={form.operator} onChange={e=>set('operator',e.target.value)} style={{minWidth:'80px'}}>
                  {operators.map(op=><option key={op} value={op}>{op} ({OPERATOR_LABELS[op]||op})</option>)}
                </select>
                <input className="input mono" type="number" step="any" style={{width:'100px',textAlign:'center'}} value={form.threshold} onChange={e=>set('threshold',e.target.value)} placeholder="0" required />
              </div>
              <div style={{fontSize:'11px',color:'var(--nv-text-dim)',marginTop:'6px',paddingLeft:'4px'}}>
                Example: Alert when Pages / sec is {"<"} 50 — fires if a run's pages/sec drops below 50.
              </div>
            </div>

            {/* Scope */}
            <div style={{borderTop:'1px solid var(--nv-border)',paddingTop:'16px'}}>
              <div style={{fontSize:'13px',fontWeight:600,color:'var(--nv-green)',marginBottom:'12px',textTransform:'uppercase',letterSpacing:'0.05em'}}>Scope (optional filters)</div>
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'12px'}}>
                <div>
                  <label style={labelStyle}>Dataset</label>
                  <select className="select" style={{width:'100%'}} value={form.dataset_filter} onChange={e=>set('dataset_filter',e.target.value)}>
                    <option value="">All datasets</option>
                    {datasets.map(d=><option key={d} value={d}>{d}</option>)}
                  </select>
                </div>
                <div>
                  <label style={labelStyle}>Preset</label>
                  <select className="select" style={{width:'100%'}} value={form.preset_filter} onChange={e=>set('preset_filter',e.target.value)}>
                    <option value="">All presets</option>
                    {presets.map(p=><option key={p} value={p}>{p}</option>)}
                  </select>
                </div>
              </div>
            </div>

            {/* Enabled & Notifications */}
            <div style={{borderTop:'1px solid var(--nv-border)',paddingTop:'16px',display:'flex',flexDirection:'column',gap:'12px'}}>
              <div style={{display:'flex',alignItems:'center',gap:'12px'}}>
                <label className="toggle">
                  <input type="checkbox" checked={form.enabled} onChange={e=>set('enabled',e.target.checked)} />
                  <span className="toggle-slider"></span>
                </label>
                <span style={{fontSize:'14px',color:'var(--nv-text)'}}>Rule is {form.enabled?'enabled':'disabled'}</span>
              </div>
              <div style={{display:'flex',alignItems:'center',gap:'12px'}}>
                <label className="toggle">
                  <input type="checkbox" checked={form.slack_notify} onChange={e=>set('slack_notify',e.target.checked)} />
                  <span className="toggle-slider"></span>
                </label>
                <div>
                  <span style={{fontSize:'14px',color:'var(--nv-text)'}}>Send Slack notification when this rule fires</span>
                  <div style={{fontSize:'11px',color:'var(--nv-text-dim)',marginTop:'2px'}}>Configure the webhook URL in Settings &gt; Slack Integration</div>
                </div>
              </div>
            </div>
          </div>
          <div className="modal-foot">
            <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button type="submit" disabled={saving||!form.name.trim()||form.threshold===""} className="btn btn-primary" style={{flex:1,justifyContent:'center'}}>
              {saving ? <><span className="spinner" style={{marginRight:'8px'}}></span>Saving…</> : isEdit?"Update Rule":"Create Rule"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
