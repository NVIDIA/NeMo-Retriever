/* ===== Scheduling View ===== */
function fmtRelative(isoStr) {
  if (!isoStr) return null;
  const target = new Date(isoStr);
  const now = new Date();
  let diff = Math.round((target - now) / 1000);
  if (diff < 0) return "now";
  if (diff < 60) return `${diff}s`;
  const mins = Math.floor(diff / 60); diff %= 60;
  if (mins < 60) return `${mins}m ${diff}s`;
  const hrs = Math.floor(mins / 60); const rm = mins % 60;
  if (hrs < 24) return `${hrs}h ${rm}m`;
  const days = Math.floor(hrs / 24); const rh = hrs % 24;
  return `${days}d ${rh}h`;
}

function fmtIsoShort(isoStr) {
  if (!isoStr) return "\u2014";
  try {
    const d = new Date(isoStr);
    return d.toLocaleString(undefined, {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',hour12:false,timeZoneName:'short'});
  } catch { return isoStr; }
}

function SchedulingView({ schedules, loading, onRefresh, runners }) {
  const [showForm, setShowForm] = useState(false);
  const [editSchedule, setEditSchedule] = useState(null);
  const [upcoming, setUpcoming] = useState([]);
  const pg = usePagination(schedules, 25);

  useEffect(() => {
    fetch("/api/schedules/upcoming?count=10").then(r=>r.json()).then(setUpcoming).catch(()=>{});
  }, [schedules]);

  function handleCreate() { setEditSchedule(null); setShowForm(true); }
  function handleEdit(sched) { setEditSchedule(sched); setShowForm(true); }

  async function handleDelete(id, name) {
    if (!confirm(`Delete schedule "${name}"?`)) return;
    try {
      await fetch(`/api/schedules/${id}`, { method: "DELETE" });
      onRefresh();
    } catch {}
  }

  async function handleToggle(sched) {
    try {
      await fetch(`/api/schedules/${sched.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !sched.enabled }),
      });
      onRefresh();
    } catch {}
  }

  async function handleTriggerNow(id) {
    try {
      const res = await fetch(`/api/schedules/${id}/trigger`, { method: "POST" });
      if (!res.ok) throw new Error(await res.text());
      onRefresh();
    } catch (err) {
      alert("Failed to trigger schedule: " + err.message);
    }
  }

  function requirementsText(s) {
    const parts = [];
    if (s.min_gpu_count) parts.push(`${s.min_gpu_count} GPU`);
    if (s.gpu_type_pattern) parts.push(s.gpu_type_pattern);
    if (s.min_cpu_count) parts.push(`${s.min_cpu_count} CPU`);
    if (s.min_memory_gb) parts.push(`${s.min_memory_gb} GB RAM`);
    const prIds = s.preferred_runner_ids || [];
    if (prIds.length > 0) parts.push(`${prIds.length} runner${prIds.length!==1?'s':''}`);
    return parts.length > 0 ? parts.join(", ") : "\u2014";
  }

  return (
    <>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'20px'}}>
        <button className="btn btn-primary" onClick={handleCreate}><IconPlus /> Create Schedule</button>
        <button className="btn btn-secondary btn-icon" onClick={onRefresh} title="Refresh"><IconRefresh /></button>
      </div>

      {upcoming.length > 0 && (
        <div style={{marginBottom:'20px'}}>
          <div className="section-title" style={{marginBottom:'8px'}}>Upcoming Queue</div>
          <div className="card" style={{padding:0}}>
            <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill, minmax(280px, 1fr))',gap:0}}>
              {upcoming.map((u, i) => (
                <div key={i} style={{
                  display:'flex',alignItems:'center',gap:'12px',
                  padding:'10px 16px',borderBottom:'1px solid var(--nv-border)',borderRight:'1px solid var(--nv-border)',
                }}>
                  <div style={{
                    minWidth:'28px',height:'28px',borderRadius:'50%',display:'flex',alignItems:'center',justifyContent:'center',
                    fontSize:'12px',fontWeight:700,
                    background: i === 0 ? 'var(--nv-green)' : 'rgba(255,255,255,0.06)',
                    color: i === 0 ? '#000' : 'var(--nv-text-muted)',
                  }}>{i + 1}</div>
                  <div style={{flex:1,minWidth:0}}>
                    <div style={{display:'flex',alignItems:'center',gap:'6px'}}>
                      <span style={{color:'#fff',fontWeight:500,fontSize:'13px',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>
                        {u.schedule_name || `Schedule #${u.schedule_id}`}
                      </span>
                      {u.pending_jobs > 0 && (
                        <span style={{
                          fontSize:'10px',fontWeight:600,padding:'1px 6px',borderRadius:'100px',lineHeight:'16px',
                          background: u.pending_jobs >= 3 ? 'rgba(255,80,80,0.15)' : 'rgba(255,200,50,0.15)',
                          color: u.pending_jobs >= 3 ? '#ff5050' : '#fcd34d',
                        }}>{u.pending_jobs} queued</span>
                      )}
                    </div>
                    <div style={{fontSize:'11px',color:'var(--nv-text-dim)',marginTop:'1px'}}>
                      {u.dataset}{u.preset ? ` / ${u.preset}` : ''}
                    </div>
                  </div>
                  <div style={{textAlign:'right',flexShrink:0}}>
                    <div style={{fontSize:'12px',color:'var(--nv-green)',fontWeight:600}}>{fmtRelative(u.fire_at)}</div>
                    <div style={{fontSize:'10px',color:'var(--nv-text-dim)',marginTop:'1px'}}>{fmtIsoShort(u.fire_at)}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <div style={{overflowX:'auto'}}>
          <table className="runs-table">
            <thead>
              <tr>
                <th style={{width:'60px'}}>Enabled</th>
                <th>Name</th><th>Type</th><th>Dataset</th>
                <th>Cron / Branch</th><th>Next Run</th><th style={{textAlign:'center'}}>Pending</th><th>Requirements</th>
                <th>Last Triggered</th><th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan="10" style={{textAlign:'center',padding:'60px',color:'var(--nv-text-muted)'}}>
                  <div className="spinner spinner-lg" style={{margin:'0 auto 12px'}}></div><div>Loading schedules…</div>
                </td></tr>
              ) : schedules.length === 0 ? (
                <tr><td colSpan="10" style={{textAlign:'center',padding:'60px',color:'var(--nv-text-muted)'}}>
                  <div style={{marginBottom:'8px',fontSize:'15px'}}>No schedules configured</div>
                  <div style={{fontSize:'12px',color:'var(--nv-text-dim)'}}>Create a schedule to automatically run benchmarks on a cron schedule or GitHub push event.</div>
                </td></tr>
              ) : pg.pageData.map(s => (
                <tr key={s.id} style={{cursor:'default'}}>
                  <td>
                    <label className="toggle">
                      <input type="checkbox" checked={s.enabled} onChange={()=>handleToggle(s)} />
                      <span className="toggle-slider"></span>
                    </label>
                  </td>
                  <td>
                    <div style={{color:'#fff',fontWeight:500}}>{s.name}</div>
                    {s.description && <div style={{fontSize:'11px',color:'var(--nv-text-dim)',marginTop:'2px'}}>{s.description}</div>}
                  </td>
                  <td><ScheduleTypeBadge type={s.trigger_type} /></td>
                  <td style={{color:'var(--nv-text-muted)'}}>{s.preset_matrix ? <span title="Preset Matrix"><span style={{color:'var(--nv-green)',fontWeight:600}}>Matrix:</span> {s.preset_matrix}</span> : <>{s.dataset}{s.preset ? ` / ${s.preset}` : ''}</>}</td>
                  <td className="mono" style={{fontSize:'12px',color:'var(--nv-text-muted)'}}>
                    {s.trigger_type === 'cron' ? (s.cron_expression || "\u2014") : `${s.github_repo || ''}:${s.github_branch || 'main'}`}
                  </td>
                  <td style={{fontSize:'12px',whiteSpace:'nowrap'}}>
                    {s.next_run_at ? (
                      <div>
                        <span style={{color:'var(--nv-green)',fontWeight:600}}>{fmtRelative(s.next_run_at)}</span>
                        <div style={{fontSize:'10px',color:'var(--nv-text-dim)',marginTop:'1px'}}>{fmtIsoShort(s.next_run_at)}</div>
                      </div>
                    ) : (
                      <span style={{color:'var(--nv-text-dim)'}}>{s.trigger_type === 'cron' ? (s.enabled ? "\u2014" : "disabled") : "on push"}</span>
                    )}
                  </td>
                  <td style={{textAlign:'center'}}>
                    {s.pending_jobs > 0 ? (
                      <span style={{
                        display:'inline-block',fontSize:'11px',fontWeight:700,padding:'2px 8px',borderRadius:'100px',
                        background: s.pending_jobs >= 3 ? 'rgba(255,80,80,0.15)' : 'rgba(255,200,50,0.15)',
                        color: s.pending_jobs >= 3 ? '#ff5050' : '#fcd34d',
                      }}>{s.pending_jobs}</span>
                    ) : (
                      <span style={{color:'var(--nv-text-dim)',fontSize:'12px'}}>0</span>
                    )}
                  </td>
                  <td style={{fontSize:'12px',color:'var(--nv-text-muted)'}}>{requirementsText(s)}</td>
                  <td style={{color:'var(--nv-text-muted)',fontSize:'12px',whiteSpace:'nowrap'}}>{fmtTs(s.last_triggered_at)}</td>
                  <td>
                    <div style={{display:'flex',gap:'6px',flexWrap:'nowrap'}}>
                      <button className="btn btn-secondary btn-sm" onClick={()=>handleTriggerNow(s.id)} title="Trigger Now"><IconPlay /> Run</button>
                      <button className="btn btn-secondary btn-sm" onClick={()=>handleEdit(s)} title="Edit"><IconEdit /> Edit</button>
                      <button className="btn btn-sm" onClick={()=>handleDelete(s.id,s.name)} title="Delete" style={{background:'rgba(255,80,80,0.1)',color:'#ff5050',border:'1px solid rgba(255,80,80,0.2)'}}><IconTrash /> Delete</button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Pagination page={pg.page} totalPages={pg.totalPages} totalItems={pg.totalItems}
          pageSize={pg.pageSize} onPageChange={pg.setPage} onPageSizeChange={pg.setPageSize} />
      </div>
      {showForm && (
        <ScheduleFormModal
          schedule={editSchedule}
          runners={runners}
          onClose={()=>setShowForm(false)}
          onSaved={()=>{setShowForm(false);onRefresh();}}
        />
      )}
    </>
  );
}

/* ===== Schedule Form Modal ===== */
function ScheduleFormModal({ schedule, runners, onClose, onSaved }) {
  const isEdit = !!schedule;
  const [datasets, setDatasets] = useState([]);
  const [presets, setPresets] = useState([]);
  const [presetMatricesList, setPresetMatricesList] = useState([]);
  const [gpuTypes, setGpuTypes] = useState([]);
  const [useMatrix, setUseMatrix] = useState(!!schedule?.preset_matrix);
  const [selectedRunnerIds, setSelectedRunnerIds] = useState(new Set((schedule?.preferred_runner_ids || []).map(Number)));
  const [form, setForm] = useState({
    name: schedule?.name || "",
    description: schedule?.description || "",
    dataset: schedule?.dataset || "",
    preset: schedule?.preset || "",
    preset_matrix: schedule?.preset_matrix || "",
    config: schedule?.config || "",
    trigger_type: schedule?.trigger_type || "cron",
    cron_expression: schedule?.cron_expression || "",
    github_repo: schedule?.github_repo || "",
    github_branch: schedule?.github_branch || "main",
    min_gpu_count: schedule?.min_gpu_count ?? "",
    gpu_type_pattern: schedule?.gpu_type_pattern || "",
    min_cpu_count: schedule?.min_cpu_count ?? "",
    min_memory_gb: schedule?.min_memory_gb ?? "",
    enabled: schedule?.enabled ?? true,
    tags: (schedule?.tags || []).join(", "),
  });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetch("/api/config").then(r=>r.json()).then(cfg => {
      setDatasets(cfg.datasets || []);
      setPresets(cfg.presets || []);
      setPresetMatricesList(cfg.preset_matrices || []);
      if (!useMatrix && !form.dataset && cfg.datasets?.length) setForm(f=>({...f, dataset: cfg.datasets[0]}));
    });
    fetch("/api/runners/gpu-types").then(r=>r.json()).then(t => setGpuTypes(t || [])).catch(()=>{});
  }, []);

  function set(field, val) { setForm(f=>({...f,[field]:val})); }

  const canSubmit = form.name.trim() && (useMatrix ? form.preset_matrix : form.dataset);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!canSubmit) return;
    setSaving(true);
    try {
      const payload = {
        name: form.name.trim(),
        description: form.description.trim() || null,
        dataset: useMatrix ? "" : form.dataset,
        preset: useMatrix ? null : (form.preset || null),
        preset_matrix: useMatrix ? form.preset_matrix : null,
        config: form.config.trim() || null,
        trigger_type: form.trigger_type,
        cron_expression: form.trigger_type === "cron" ? form.cron_expression.trim() || null : null,
        github_repo: form.trigger_type === "github_push" ? form.github_repo.trim() || null : null,
        github_branch: form.trigger_type === "github_push" ? form.github_branch.trim() || "main" : null,
        min_gpu_count: form.min_gpu_count !== "" ? parseInt(form.min_gpu_count, 10) : null,
        gpu_type_pattern: form.gpu_type_pattern || null,
        min_cpu_count: form.min_cpu_count !== "" ? parseInt(form.min_cpu_count, 10) : null,
        min_memory_gb: form.min_memory_gb !== "" ? parseFloat(form.min_memory_gb) : null,
        preferred_runner_id: null,
        preferred_runner_ids: selectedRunnerIds.size > 0 ? [...selectedRunnerIds] : null,
        enabled: form.enabled,
        tags: form.tags ? form.tags.split(",").map(t=>t.trim()).filter(Boolean) : [],
      };
      const url = isEdit ? `/api/schedules/${schedule.id}` : "/api/schedules";
      const res = await fetch(url, {
        method: isEdit ? "PUT" : "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      onSaved();
    } catch (err) {
      alert("Failed to save schedule: " + err.message);
    } finally { setSaving(false); }
  }

  const labelStyle = {display:'block',fontSize:'12px',fontWeight:500,color:'var(--nv-text-muted)',marginBottom:'6px',textTransform:'uppercase',letterSpacing:'0.04em'};
  const helpStyle = {fontSize:'11px',color:'var(--nv-text-dim)',marginTop:'4px'};
  const onlineRunners = (runners || []).filter(r => r.status === 'online' || r.status === 'paused');

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{maxWidth:'640px'}} onClick={e=>e.stopPropagation()}>
        <div className="modal-head">
          <h2 style={{fontSize:'16px',fontWeight:700,color:'#fff'}}>{isEdit ? "Edit Schedule" : "Create Schedule"}</h2>
          <button className="btn btn-ghost btn-icon" onClick={onClose} style={{borderRadius:'50%'}}><IconX /></button>
        </div>
        <form onSubmit={handleSubmit}>
          <div className="modal-body" style={{display:'flex',flexDirection:'column',gap:'16px'}}>
            {/* Basic Info */}
            <div>
              <label style={labelStyle}>Name *</label>
              <input className="input" style={{width:'100%'}} value={form.name} onChange={e=>set('name',e.target.value)} placeholder="e.g. Nightly A100 Benchmark" required />
            </div>
            <div>
              <label style={labelStyle}>Description</label>
              <input className="input" style={{width:'100%'}} value={form.description} onChange={e=>set('description',e.target.value)} placeholder="Optional description" />
            </div>

            {/* Run Mode Toggle */}
            <div>
              <label style={labelStyle}>Run Mode</label>
              <div style={{display:'flex',gap:'8px'}}>
                <button type="button" className={`btn ${!useMatrix?'btn-primary':'btn-secondary'}`} style={{flex:1,justifyContent:'center'}}
                  onClick={()=>setUseMatrix(false)}>Single Dataset + Preset</button>
                <button type="button" className={`btn ${useMatrix?'btn-primary':'btn-secondary'}`} style={{flex:1,justifyContent:'center'}}
                  onClick={()=>setUseMatrix(true)}>Preset Matrix</button>
              </div>
              {useMatrix && <div style={helpStyle}>All dataset &times; preset combinations from the selected matrix will be queued as separate jobs</div>}
            </div>

            {!useMatrix ? (
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'14px'}}>
                <div>
                  <label style={labelStyle}>Dataset *</label>
                  <select className="select" style={{width:'100%'}} value={form.dataset} onChange={e=>set('dataset',e.target.value)}>
                    <option value="">Select dataset…</option>
                    {datasets.map(d=><option key={d} value={d}>{d}</option>)}
                  </select>
                </div>
                <div>
                  <label style={labelStyle}>Preset</label>
                  <select className="select" style={{width:'100%'}} value={form.preset} onChange={e=>set('preset',e.target.value)}>
                    <option value="">Default</option>
                    {presets.map(p=><option key={p} value={p}>{p}</option>)}
                  </select>
                </div>
              </div>
            ) : (
              <div>
                <label style={labelStyle}>Preset Matrix *</label>
                <select className="select" style={{width:'100%'}} value={form.preset_matrix} onChange={e=>set('preset_matrix',e.target.value)}>
                  <option value="">Select matrix…</option>
                  {presetMatricesList.map(m=><option key={m.id} value={m.name}>{m.name}</option>)}
                </select>
                {presetMatricesList.length === 0 && (
                  <div style={helpStyle}>No matrices available. Create one in the Presets view first.</div>
                )}
              </div>
            )}

            {/* Trigger Type */}
            <div>
              <label style={labelStyle}>Trigger Type</label>
              <div style={{display:'flex',gap:'8px'}}>
                <button type="button" className={`btn ${form.trigger_type==='cron'?'btn-primary':'btn-secondary'}`} style={{flex:1,justifyContent:'center'}}
                  onClick={()=>set('trigger_type','cron')}><IconCalendar /> Cron Schedule</button>
                <button type="button" className={`btn ${form.trigger_type==='github_push'?'btn-primary':'btn-secondary'}`} style={{flex:1,justifyContent:'center'}}
                  onClick={()=>set('trigger_type','github_push')}><IconGithub /> GitHub Push</button>
              </div>
            </div>

            {form.trigger_type === 'cron' && (
              <div>
                <label style={labelStyle}>Cron Expression</label>
                <input className="input mono" style={{width:'100%'}} value={form.cron_expression} onChange={e=>set('cron_expression',e.target.value)} placeholder="0 2 * * *" />
                <div style={helpStyle}>Format: minute hour day month day_of_week — e.g. "0 2 * * *" = daily at 2:00 AM</div>
              </div>
            )}

            {form.trigger_type === 'github_push' && (
              <div style={{display:'grid',gridTemplateColumns:'2fr 1fr',gap:'14px'}}>
                <div>
                  <label style={labelStyle}>GitHub Repository</label>
                  <input className="input" style={{width:'100%'}} value={form.github_repo} onChange={e=>set('github_repo',e.target.value)} placeholder="NVIDIA/nemo-retriever" />
                </div>
                <div>
                  <label style={labelStyle}>Branch</label>
                  <input className="input" style={{width:'100%'}} value={form.github_branch} onChange={e=>set('github_branch',e.target.value)} placeholder="main" />
                </div>
              </div>
            )}

            {/* Runner Requirements */}
            <div style={{borderTop:'1px solid var(--nv-border)',paddingTop:'16px'}}>
              <div style={{...labelStyle,marginBottom:'12px',fontSize:'13px',fontWeight:600}}>Runner Requirements</div>
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr 1fr',gap:'14px'}}>
                <div>
                  <label style={labelStyle}>Min GPUs</label>
                  <input className="input" style={{width:'100%'}} type="number" min="0" value={form.min_gpu_count} onChange={e=>set('min_gpu_count',e.target.value)} placeholder="Any" />
                </div>
                <div>
                  <label style={labelStyle}>GPU Type</label>
                  <select className="select" style={{width:'100%'}} value={form.gpu_type_pattern} onChange={e=>set('gpu_type_pattern',e.target.value)}>
                    <option value="">Any GPU</option>
                    {gpuTypes.map(g=><option key={g} value={g}>{g}</option>)}
                  </select>
                </div>
                <div>
                  <label style={labelStyle}>Min CPUs</label>
                  <input className="input" style={{width:'100%'}} type="number" min="0" value={form.min_cpu_count} onChange={e=>set('min_cpu_count',e.target.value)} placeholder="Any" />
                </div>
                <div>
                  <label style={labelStyle}>Min Memory (GB)</label>
                  <input className="input" style={{width:'100%'}} type="number" min="0" step="0.1" value={form.min_memory_gb} onChange={e=>set('min_memory_gb',e.target.value)} placeholder="Any" />
                </div>
              </div>
            </div>

            {/* Preferred Runners */}
            <div style={{borderTop:'1px solid var(--nv-border)',paddingTop:'16px'}}>
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'14px'}}>
                <div>
                  <label style={labelStyle}>Preferred Runners <span style={{color:'var(--nv-green)',fontWeight:700}}>({selectedRunnerIds.size} selected)</span></label>
                  <div style={{border:'1px solid var(--nv-border)',borderRadius:'8px',maxHeight:'160px',overflowY:'auto',padding:'4px'}}>
                    {onlineRunners.length === 0 ? (
                      <div style={{padding:'12px',textAlign:'center',color:'var(--nv-text-dim)',fontSize:'12px'}}>No runners online</div>
                    ) : onlineRunners.map(r => (
                      <label key={r.id} style={{display:'flex',alignItems:'center',gap:'8px',padding:'5px 8px',cursor:'pointer',borderRadius:'4px',fontSize:'13px',color:selectedRunnerIds.has(r.id)?'#fff':'var(--nv-text-muted)',background:selectedRunnerIds.has(r.id)?'rgba(118,185,0,0.08)':'transparent'}}>
                        <input type="checkbox" checked={selectedRunnerIds.has(r.id)} onChange={()=>{
                          setSelectedRunnerIds(prev => { const s = new Set(prev); s.has(r.id) ? s.delete(r.id) : s.add(r.id); return s; });
                        }} style={{accentColor:'var(--nv-green)'}} />
                        #{r.id} — {r.name}{r.gpu_type ? ` [${r.gpu_type}]` : ''}{r.status==='paused'?' (paused)':''}
                      </label>
                    ))}
                  </div>
                  <div style={{fontSize:'11px',color:'var(--nv-text-dim)',marginTop:'4px'}}>Leave empty to auto-select based on requirements above</div>
                </div>
                <div>
                  <label style={labelStyle}>Tags (comma-separated)</label>
                  <input className="input" style={{width:'100%'}} value={form.tags} onChange={e=>set('tags',e.target.value)} placeholder="e.g. nightly, regression" />
                </div>
              </div>
            </div>

            {/* Enabled Toggle */}
            <div style={{display:'flex',alignItems:'center',gap:'12px'}}>
              <label className="toggle">
                <input type="checkbox" checked={form.enabled} onChange={e=>set('enabled',e.target.checked)} />
                <span className="toggle-slider"></span>
              </label>
              <span style={{fontSize:'14px',color:'var(--nv-text)'}}>Schedule is {form.enabled ? 'enabled' : 'disabled'}</span>
            </div>
          </div>

          <div className="modal-foot">
            <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button type="submit" disabled={saving||!canSubmit} className="btn btn-primary" style={{flex:1,justifyContent:'center'}}>
              {saving ? <><span className="spinner" style={{marginRight:'8px'}}></span>Saving…</> : isEdit ? "Update Schedule" : "Create Schedule"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
