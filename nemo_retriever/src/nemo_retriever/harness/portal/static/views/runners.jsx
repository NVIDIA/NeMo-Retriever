/* ===== Runners View ===== */
function RunnersView({ runners, loading, onRefresh, githubRepoUrl }) {
  const [showForm, setShowForm] = useState(false);
  const [editRunner, setEditRunner] = useState(null);
  const pg = usePagination(runners, 25);

  function handleCreate() { setEditRunner(null); setShowForm(true); }
  function handleEdit(runner) { setEditRunner(runner); setShowForm(true); }
  async function handleDelete(id, name) {
    if (!confirm(`Delete runner "${name}"?`)) return;
    try {
      await fetch(`/api/runners/${id}`, { method:"DELETE" });
      onRefresh();
    } catch {}
  }
  async function handleTogglePause(runner) {
    const action = runner.status === 'paused' ? 'resume' : 'pause';
    try {
      await fetch(`/api/runners/${runner.id}/${action}`, { method:"POST" });
      onRefresh();
    } catch {}
  }

  return (
    <>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'20px'}}>
        <button className="btn btn-primary" onClick={handleCreate}><IconPlus /> Register Runner</button>
        <button className="btn btn-secondary btn-icon" onClick={onRefresh} title="Refresh"><IconRefresh /></button>
      </div>
      <div className="card">
        <div style={{overflowX:'auto'}}>
          <table className="runs-table">
            <thead>
              <tr>
                <th>Status</th><th>Name</th><th>Hostname</th><th>Git Commit</th><th>Ray Cluster</th><th>GPU Type</th>
                <th style={{textAlign:'right'}}>GPUs</th><th style={{textAlign:'right'}}>CPUs</th>
                <th style={{textAlign:'right'}}>Memory (GB)</th><th>Last Heartbeat</th><th>Tags</th><th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan="12" style={{textAlign:'center',padding:'60px',color:'var(--nv-text-muted)'}}>
                  <div className="spinner spinner-lg" style={{margin:'0 auto 12px'}}></div><div>Loading runners…</div>
                </td></tr>
              ) : runners.length === 0 ? (
                <tr><td colSpan="12" style={{textAlign:'center',padding:'60px',color:'var(--nv-text-muted)'}}>
                  <div style={{marginBottom:'8px',fontSize:'15px'}}>No runners registered</div>
                  <div style={{fontSize:'12px',color:'var(--nv-text-dim)'}}>
                    Register a runner manually or use <code className="mono" style={{background:'rgba(255,255,255,0.06)',padding:'2px 6px',borderRadius:'4px'}}>retriever harness runner start --manager-url &lt;portal-url&gt;</code>
                  </div>
                </td></tr>
              ) : pg.pageData.map(r => {
                const isOffline = r.status === 'offline';
                const isPaused = r.status === 'paused';
                const ago = timeSince(r.last_heartbeat);
                const rowBg = isOffline ? 'rgba(255,80,80,0.04)' : isPaused ? 'rgba(255,140,0,0.04)' : 'transparent';
                return (
                <tr key={r.id} style={{cursor:'default',background:rowBg}}>
                  <td><RunnerStatusDot status={r.status} /></td>
                  <td style={{color:isOffline?'#ff8888':isPaused?'#ff8c00':'#fff',fontWeight:500}}>{r.name}</td>
                  <td style={{color:'var(--nv-text-muted)',fontSize:'12px'}}>{r.hostname || "\u2014"}</td>
                  <td style={{fontSize:'12px'}}>
                    {r.pending_update_commit ? (
                      <span style={{display:'inline-flex',alignItems:'center',gap:'6px'}}>
                        <CommitLink sha={r.git_commit} repoUrl={githubRepoUrl} />
                        <span title={`Updating to ${r.pending_update_commit.slice(0,10)}…`}
                          style={{color:'#ff8c00',fontSize:'11px',fontWeight:600,animation:'pulse-offline 1.5s infinite'}}>
                          ⟳ updating
                        </span>
                      </span>
                    ) : (
                      <CommitLink sha={r.git_commit} repoUrl={githubRepoUrl} />
                    )}
                  </td>
                  <td style={{fontSize:'12px'}}>
                    {r.ray_address ? (
                      <span className="mono" style={{color:'var(--nv-green)',fontSize:'11px',background:'rgba(118,185,0,0.08)',padding:'2px 6px',borderRadius:'4px'}}>
                        {r.ray_address}
                      </span>
                    ) : (
                      <span style={{color:'var(--nv-text-dim)',fontSize:'11px'}}>local</span>
                    )}
                  </td>
                  <td style={{color:'var(--nv-text-muted)',fontSize:'12px'}}>{r.gpu_type || "\u2014"}</td>
                  <td style={{textAlign:'right'}}>{r.gpu_count ?? "\u2014"}</td>
                  <td style={{textAlign:'right'}}>{r.cpu_count ?? "\u2014"}</td>
                  <td style={{textAlign:'right'}}>{r.memory_gb != null ? r.memory_gb.toFixed(1) : "\u2014"}</td>
                  <td style={{color:isOffline?'#ff5050':'var(--nv-text-muted)',fontSize:'12px',whiteSpace:'nowrap'}}>
                    {fmtTs(r.last_heartbeat)}
                    {ago && <span style={{marginLeft:'6px',fontSize:'11px',color:isOffline?'#ff8888':'var(--nv-text-dim)'}}>({ago})</span>}
                  </td>
                  <td>{(r.tags||[]).map((t,i)=><span key={i} className="tag" style={{marginRight:'4px'}}>{t}</span>)}</td>
                  <td>
                    <div style={{display:'flex',gap:'6px',flexWrap:'nowrap'}}>
                      {r.status !== 'offline' && (
                        r.status === 'paused' ? (
                          <button className="btn btn-sm" onClick={()=>handleTogglePause(r)} title="Resume runner"
                            style={{background:'rgba(118,185,0,0.1)',color:'var(--nv-green)',border:'1px solid rgba(118,185,0,0.2)'}}>
                            <IconPlay /> Resume
                          </button>
                        ) : (
                          <button className="btn btn-sm" onClick={()=>handleTogglePause(r)} title="Pause runner — stops new jobs from being dispatched"
                            style={{background:'rgba(255,140,0,0.1)',color:'#ff8c00',border:'1px solid rgba(255,140,0,0.2)'}}>
                            <IconPause /> Pause
                          </button>
                        )
                      )}
                      <button className="btn btn-secondary btn-sm" onClick={()=>handleEdit(r)} title="Edit"><IconEdit /> Edit</button>
                      <button className="btn btn-sm" onClick={()=>handleDelete(r.id,r.name)} title="Delete" style={{background:'rgba(255,80,80,0.1)',color:'#ff5050',border:'1px solid rgba(255,80,80,0.2)'}}><IconTrash /> Delete</button>
                    </div>
                  </td>
                </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <Pagination page={pg.page} totalPages={pg.totalPages} totalItems={pg.totalItems}
          pageSize={pg.pageSize} onPageChange={pg.setPage} onPageSizeChange={pg.setPageSize} />
      </div>
      {showForm && <RunnerFormModal runner={editRunner} onClose={()=>setShowForm(false)} onSaved={()=>{setShowForm(false);onRefresh();}} />}
    </>
  );
}

/* ===== Runner Form Modal (Create / Edit) ===== */
function RunnerFormModal({ runner, onClose, onSaved }) {
  const isEdit = !!runner;
  const [form, setForm] = useState({
    name: runner?.name || "",
    hostname: runner?.hostname || "",
    url: runner?.url || "",
    gpu_type: runner?.gpu_type || "",
    gpu_count: runner?.gpu_count ?? "",
    cpu_count: runner?.cpu_count ?? "",
    memory_gb: runner?.memory_gb ?? "",
    status: runner?.status || "online",
    tags: (runner?.tags || []).join(", "),
    ray_address: runner?.ray_address || "",
  });
  const [saving, setSaving] = useState(false);

  function set(field, val) { setForm(f=>({...f,[field]:val})); }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!form.name.trim()) return;
    setSaving(true);
    try {
      const payload = {
        name: form.name.trim(),
        hostname: form.hostname.trim() || null,
        url: form.url.trim() || null,
        gpu_type: form.gpu_type.trim() || null,
        gpu_count: form.gpu_count !== "" ? parseInt(form.gpu_count,10) : null,
        cpu_count: form.cpu_count !== "" ? parseInt(form.cpu_count,10) : null,
        memory_gb: form.memory_gb !== "" ? parseFloat(form.memory_gb) : null,
        status: form.status,
        tags: form.tags ? form.tags.split(",").map(t=>t.trim()).filter(Boolean) : [],
        ray_address: form.ray_address.trim() || null,
      };
      const url = isEdit ? `/api/runners/${runner.id}` : "/api/runners";
      const res = await fetch(url, {
        method: isEdit ? "PUT" : "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      onSaved();
    } catch (err) {
      alert("Failed to save runner: " + err.message);
    } finally { setSaving(false); }
  }

  const labelStyle = {display:'block',fontSize:'12px',fontWeight:500,color:'var(--nv-text-muted)',marginBottom:'6px',textTransform:'uppercase',letterSpacing:'0.04em'};

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{maxWidth:'520px'}} onClick={e=>e.stopPropagation()}>
        <div className="modal-head">
          <h2 style={{fontSize:'16px',fontWeight:700,color:'#fff'}}>{isEdit ? "Edit Runner" : "Register Runner"}</h2>
          <button className="btn btn-ghost btn-icon" onClick={onClose} style={{borderRadius:'50%'}}><IconX /></button>
        </div>
        <form onSubmit={handleSubmit}>
          <div style={{padding:'24px',display:'flex',flexDirection:'column',gap:'14px'}}>
            <div>
              <label style={labelStyle}>Name *</label>
              <input className="input" style={{width:'100%'}} value={form.name} onChange={e=>set('name',e.target.value)} placeholder="e.g. gpu-worker-01" required />
            </div>
            <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'14px'}}>
              <div>
                <label style={labelStyle}>Hostname</label>
                <input className="input" style={{width:'100%'}} value={form.hostname} onChange={e=>set('hostname',e.target.value)} placeholder="e.g. dgx-station-3" />
              </div>
              <div>
                <label style={labelStyle}>URL</label>
                <input className="input" style={{width:'100%'}} value={form.url} onChange={e=>set('url',e.target.value)} placeholder="http://host:port" />
              </div>
            </div>
            <div style={{display:'grid',gridTemplateColumns:'2fr 1fr 1fr',gap:'14px'}}>
              <div>
                <label style={labelStyle}>GPU Type</label>
                <input className="input" style={{width:'100%'}} value={form.gpu_type} onChange={e=>set('gpu_type',e.target.value)} placeholder="e.g. NVIDIA A100-SXM4-80GB" />
              </div>
              <div>
                <label style={labelStyle}>GPU Count</label>
                <input className="input" style={{width:'100%'}} type="number" min="0" value={form.gpu_count} onChange={e=>set('gpu_count',e.target.value)} placeholder="0" />
              </div>
              <div>
                <label style={labelStyle}>CPU Count</label>
                <input className="input" style={{width:'100%'}} type="number" min="0" value={form.cpu_count} onChange={e=>set('cpu_count',e.target.value)} placeholder="0" />
              </div>
            </div>
            <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'14px'}}>
              <div>
                <label style={labelStyle}>Memory (GB)</label>
                <input className="input" style={{width:'100%'}} type="number" min="0" step="0.1" value={form.memory_gb} onChange={e=>set('memory_gb',e.target.value)} placeholder="0" />
              </div>
              <div>
                <label style={labelStyle}>Status</label>
                <select className="select" style={{width:'100%'}} value={form.status} onChange={e=>set('status',e.target.value)}>
                  <option value="online">Online</option>
                  <option value="offline">Offline</option>
                  <option value="busy">Busy</option>
                </select>
              </div>
            </div>
            <div>
              <label style={labelStyle}>Ray Cluster Address</label>
              <div style={{display:'flex',gap:'8px',alignItems:'center'}}>
                <input className="input" style={{flex:1}} value={form.ray_address} onChange={e=>set('ray_address',e.target.value)}
                  placeholder="Leave empty for local Ray" />
              </div>
              <div style={{marginTop:'6px',display:'flex',gap:'6px',flexWrap:'wrap'}}>
                {["auto","ray://localhost:10001","ray://head-node:10001"].map(preset => (
                  <button key={preset} type="button" className="btn btn-sm"
                    onClick={()=>set('ray_address',preset)}
                    style={{fontSize:'11px',padding:'2px 8px',
                      background: form.ray_address===preset ? 'rgba(118,185,0,0.15)' : 'rgba(255,255,255,0.04)',
                      color: form.ray_address===preset ? 'var(--nv-green)' : 'var(--nv-text-muted)',
                      border: form.ray_address===preset ? '1px solid rgba(118,185,0,0.3)' : '1px solid rgba(255,255,255,0.08)',
                      borderRadius:'4px',cursor:'pointer'}}>
                    {preset}
                  </button>
                ))}
                {form.ray_address && (
                  <button type="button" className="btn btn-sm"
                    onClick={()=>set('ray_address','')}
                    style={{fontSize:'11px',padding:'2px 8px',background:'rgba(255,80,80,0.08)',color:'#ff5050',
                      border:'1px solid rgba(255,80,80,0.2)',borderRadius:'4px',cursor:'pointer'}}>
                    Clear (local)
                  </button>
                )}
              </div>
              <div style={{marginTop:'4px',fontSize:'11px',color:'var(--nv-text-dim)'}}>
                Use <code className="mono" style={{background:'rgba(255,255,255,0.06)',padding:'1px 4px',borderRadius:'3px'}}>auto</code> for
                an existing local cluster, or <code className="mono" style={{background:'rgba(255,255,255,0.06)',padding:'1px 4px',borderRadius:'3px'}}>ray://host:port</code> for
                a remote cluster. Leave empty for embedded local Ray.
              </div>
            </div>
            <div>
              <label style={labelStyle}>Tags (comma-separated)</label>
              <input className="input" style={{width:'100%'}} value={form.tags} onChange={e=>set('tags',e.target.value)} placeholder="e.g. production, gpu-cluster" />
            </div>
          </div>
          <div className="modal-foot">
            <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button type="submit" disabled={saving||!form.name.trim()} className="btn btn-primary" style={{flex:1,justifyContent:'center'}}>
              {saving ? <><span className="spinner" style={{marginRight:'8px'}}></span>Saving…</> : isEdit ? "Update Runner" : "Register Runner"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
