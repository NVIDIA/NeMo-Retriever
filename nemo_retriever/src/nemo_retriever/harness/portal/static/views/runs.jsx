/* ===== Matrix Jobs Modal ===== */
function MatrixJobsModal({ matrixRunId, matrixName, jobs, runnerMap, githubRepoUrl, onClose, onRefresh, onViewLogs, onDiagnose }) {
  const matrixJobs = jobs.filter(j => j.matrix_run_id === matrixRunId);

  const counts = { pending: 0, running: 0, cancelling: 0, completed: 0, failed: 0, cancelled: 0, error: 0 };
  matrixJobs.forEach(j => { counts[j.status] = (counts[j.status] || 0) + 1; });

  const handleCancelAll = async () => {
    if (!confirm(`Cancel all ${matrixJobs.filter(j=>j.status==="pending"||j.status==="running").length} active jobs in this matrix run?`)) return;
    try {
      await fetch(`/api/matrix-runs/${matrixRunId}/cancel`, { method: "POST" });
      onRefresh();
    } catch (e) { console.error(e); }
  };

  const handleCancelOne = async (jobId) => {
    try {
      await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
      onRefresh();
    } catch (e) { console.error(e); }
  };

  const handleForceDelete = async (jobId) => {
    if (!confirm("Force delete this stuck job?")) return;
    try {
      await fetch(`/api/jobs/${jobId}`, { method: "DELETE" });
      onRefresh();
    } catch (e) { console.error(e); }
  };

  const activeCt = (counts.pending || 0) + (counts.running || 0) + (counts.cancelling || 0);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{maxWidth:'900px',maxHeight:'85vh',overflow:'hidden',display:'flex',flexDirection:'column'}} onClick={e=>e.stopPropagation()}>
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'20px 24px',borderBottom:'1px solid var(--nv-border)'}}>
          <div>
            <h2 style={{fontSize:'16px',fontWeight:700,color:'#fff',margin:0}}>Matrix: {matrixName || matrixRunId.substring(0,8)}</h2>
            <div style={{fontSize:'12px',color:'var(--nv-text-dim)',marginTop:'4px',display:'flex',gap:'12px',flexWrap:'wrap'}}>
              <span>{matrixJobs.length} total jobs</span>
              {counts.running > 0 && <span style={{color:'var(--nv-green)'}}>● {counts.running} running</span>}
              {counts.pending > 0 && <span style={{color:'#fcd34d'}}>● {counts.pending} pending</span>}
              {counts.cancelling > 0 && <span style={{color:'#ff8844'}}>● {counts.cancelling} cancelling</span>}
              {counts.completed > 0 && <span style={{color:'var(--nv-green)'}}>✓ {counts.completed} completed</span>}
              {counts.failed > 0 && <span style={{color:'#ff5050'}}>✕ {counts.failed} failed</span>}
              {counts.cancelled > 0 && <span style={{color:'var(--nv-text-dim)'}}>⊘ {counts.cancelled} cancelled</span>}
            </div>
          </div>
          <div style={{display:'flex',gap:'8px',alignItems:'center'}}>
            {activeCt > 0 && (
              <button className="btn" style={{fontSize:'12px',padding:'6px 14px',background:'rgba(255,80,80,0.12)',color:'#ff5050',border:'1px solid rgba(255,80,80,0.25)',fontWeight:600}}
                onClick={handleCancelAll}>
                <IconStop /> Cancel All ({activeCt})
              </button>
            )}
            <button className="btn btn-secondary" style={{fontSize:'12px',padding:'6px 14px'}} onClick={onClose}>Close</button>
          </div>
        </div>

        <div style={{overflow:'auto',flex:1,padding:'0'}}>
          {matrixJobs.map(j => {
            const runner = runnerMap[j.assigned_runner_id];
            const runnerLabel = runner ? (runner.name || runner.hostname || `#${j.assigned_runner_id}`) : j.assigned_runner_id ? `Runner #${j.assigned_runner_id}` : null;
            return (
              <div key={j.id} style={{
                display:'flex',alignItems:'center',justifyContent:'space-between',
                padding:'10px 24px',borderBottom:'1px solid var(--nv-border)',
              }}>
                <div style={{display:'flex',alignItems:'center',gap:'10px',flexWrap:'wrap',flex:1,minWidth:0}}>
                  <JobStatusBadge status={j.status} />
                  {(j.status==="running" || j.status==="cancelling") && <span className="spinner"></span>}
                  <span style={{color:'#fff',fontWeight:500,fontSize:'13px'}}>{j.dataset}</span>
                  {j.preset && <span style={{color:'var(--nv-text-muted)',fontSize:'12px'}}>{j.preset}</span>}
                  {j.git_commit && <CommitLink sha={j.git_commit} repoUrl={githubRepoUrl} />}
                  {runnerLabel && <span style={{color:'var(--nv-text-dim)',fontSize:'11px',background:'var(--nv-bg)',padding:'2px 6px',borderRadius:'4px',border:'1px solid var(--nv-border)'}}>{runnerLabel}</span>}
                </div>
                <div style={{display:'flex',alignItems:'center',gap:'6px',flexShrink:0}}>
                  {j.status==="pending" && (
                    <button className="btn btn-secondary" style={{fontSize:'11px',padding:'3px 8px'}}
                      onClick={() => onDiagnose(j.id)} title="Diagnose">
                      <IconSearch />
                    </button>
                  )}
                  <button className="btn btn-secondary" style={{fontSize:'11px',padding:'3px 8px'}}
                    onClick={() => onViewLogs && onViewLogs(j.id)} title="View Logs">
                    <IconTerminal />
                  </button>
                  {(j.status==="pending" || j.status==="running") && (
                    <button className="btn" style={{fontSize:'11px',padding:'3px 8px',background:'rgba(255,80,80,0.12)',color:'#ff5050',border:'1px solid rgba(255,80,80,0.2)'}}
                      onClick={() => handleCancelOne(j.id)} title="Cancel">
                      <IconStop />
                    </button>
                  )}
                  {j.status==="cancelling" && (
                    <button className="btn" style={{fontSize:'11px',padding:'3px 8px',background:'rgba(255,80,80,0.2)',color:'#ff5050',border:'1px solid rgba(255,80,80,0.3)',fontWeight:700}}
                      onClick={() => handleForceDelete(j.id)} title="Force delete">
                      <IconTrash />
                    </button>
                  )}
                  <span className="mono" style={{fontSize:'11px',color:'var(--nv-text-dim)'}}>{j.id}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}


/* ===== Job Diagnose Modal ===== */
function JobDiagnoseModal({ jobId, onClose }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState({});

  useEffect(() => {
    if (!jobId) return;
    setLoading(true); setError(null); setData(null);
    fetch(`/api/jobs/${jobId}/diagnose`).then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    }).then(d => setData(d)).catch(e => setError(e.message)).finally(() => setLoading(false));
  }, [jobId]);

  if (!jobId) return null;

  const toggle = (rid) => setExpanded(prev => ({ ...prev, [rid]: !prev[rid] }));

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{maxWidth:'700px',maxHeight:'80vh',overflow:'auto'}} onClick={e => e.stopPropagation()}>
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'16px'}}>
          <div style={{fontSize:'16px',fontWeight:600,color:'#fff'}}>Job Scheduling Diagnosis</div>
          <button className="btn btn-secondary" style={{fontSize:'11px',padding:'4px 10px'}} onClick={onClose}>Close</button>
        </div>

        {loading && <div style={{textAlign:'center',padding:'30px'}}><span className="spinner spinner-lg"></span></div>}
        {error && <div style={{padding:'12px 16px',borderRadius:'6px',background:'rgba(255,80,80,0.08)',border:'1px solid rgba(255,80,80,0.2)',color:'#ff5050',fontSize:'13px'}}>{error}</div>}

        {data && (
          <div>
            <div style={{padding:'12px 16px',borderRadius:'8px',marginBottom:'16px',fontSize:'13px',lineHeight:'1.6',
              background: data.eligible_count > 0 ? 'rgba(118,185,0,0.06)' : 'rgba(255,80,80,0.06)',
              border: data.eligible_count > 0 ? '1px solid rgba(118,185,0,0.15)' : '1px solid rgba(255,80,80,0.15)',
              color: data.eligible_count > 0 ? '#76b900' : '#ff5050',
            }}>
              {data.summary}
            </div>

            <div style={{display:'flex',gap:'12px',flexWrap:'wrap',marginBottom:'16px'}}>
              {[
                { label: 'Job ID', value: data.job_id },
                { label: 'Dataset', value: data.dataset || '—' },
                { label: 'Assigned Runner', value: data.assigned_runner_id != null ? `#${data.assigned_runner_id}` : 'Any' },
                { label: 'Eligible', value: `${data.eligible_count} / ${data.runner_count}` },
              ].map(s => (
                <div key={s.label} style={{background:'var(--nv-bg)',border:'1px solid var(--nv-border)',borderRadius:'6px',padding:'8px 12px',minWidth:'80px'}}>
                  <div style={{fontSize:'10px',color:'var(--nv-text-dim)',textTransform:'uppercase',fontWeight:600,marginBottom:'2px'}}>{s.label}</div>
                  <div className="mono" style={{fontSize:'12px',color:'#fff',fontWeight:600}}>{s.value}</div>
                </div>
              ))}
            </div>

            {data.allowed_runner_ids && (
              <div style={{fontSize:'12px',color:'var(--nv-text-muted)',marginBottom:'12px',padding:'8px 12px',background:'rgba(255,200,50,0.06)',border:'1px solid rgba(255,200,50,0.12)',borderRadius:'6px'}}>
                Dataset <strong>{data.dataset}</strong> restricts runners to IDs: <span className="mono" style={{color:'#fcd34d'}}>[{data.allowed_runner_ids.join(', ')}]</span>
              </div>
            )}

            {data.rejected_runners && data.rejected_runners.length > 0 && (
              <div style={{fontSize:'12px',color:'var(--nv-text-muted)',marginBottom:'12px',padding:'8px 12px',background:'rgba(255,80,80,0.06)',border:'1px solid rgba(255,80,80,0.12)',borderRadius:'6px'}}>
                Previously rejected by runners: <span className="mono" style={{color:'#ff5050'}}>[{data.rejected_runners.join(', ')}]</span>
              </div>
            )}

            <div style={{fontSize:'13px',fontWeight:600,color:'#fff',marginBottom:'10px'}}>Runner Eligibility</div>
            {(!data.runners || data.runners.length === 0) ? (
              <div style={{padding:'16px',textAlign:'center',color:'var(--nv-text-muted)',fontSize:'13px',background:'var(--nv-bg)',borderRadius:'8px',border:'1px solid var(--nv-border)'}}>No runners registered.</div>
            ) : data.runners.map(r => (
              <div key={r.runner_id} style={{marginBottom:'6px',borderRadius:'8px',border:'1px solid var(--nv-border)',overflow:'hidden',background:'var(--nv-surface)'}}>
                <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',padding:'10px 14px',cursor:'pointer',
                  background: r.eligible ? 'rgba(118,185,0,0.03)' : 'rgba(255,255,255,0.01)',
                }} onClick={() => toggle(r.runner_id)}>
                  <div style={{display:'flex',alignItems:'center',gap:'10px'}}>
                    <span style={{width:'8px',height:'8px',borderRadius:'50%',flexShrink:0,
                      background: r.eligible ? '#76b900' : r.status==='offline' ? '#666' : '#ff5050',
                    }}></span>
                    <span style={{fontSize:'13px',fontWeight:600,color:'#fff'}}>{r.runner_name}</span>
                    <span style={{fontSize:'11px',color:'var(--nv-text-dim)'}}>#{r.runner_id}</span>
                    <span style={{fontSize:'10px',padding:'2px 8px',borderRadius:'4px',fontWeight:600,textTransform:'uppercase',
                      background: r.status==='online' ? 'rgba(118,185,0,0.1)' : r.status==='paused' ? 'rgba(255,200,50,0.1)' : 'rgba(255,255,255,0.05)',
                      color: r.status==='online' ? '#76b900' : r.status==='paused' ? '#fcd34d' : '#888',
                    }}>{r.status}</span>
                  </div>
                  <div style={{display:'flex',alignItems:'center',gap:'8px'}}>
                    {r.eligible ?
                      <span style={{fontSize:'11px',fontWeight:600,color:'#76b900'}}>ELIGIBLE</span> :
                      <span style={{fontSize:'11px',fontWeight:600,color:'#ff5050'}}>{r.blockers.length} blocker{r.blockers.length!==1?'s':''}</span>
                    }
                    <span style={{fontSize:'10px',color:'var(--nv-text-dim)',transform:expanded[r.runner_id]?'rotate(180deg)':'rotate(0)',transition:'transform 0.15s'}}>▼</span>
                  </div>
                </div>
                {expanded[r.runner_id] && (
                  <div style={{padding:'8px 14px 12px 32px',borderTop:'1px solid var(--nv-border)'}}>
                    {r.eligible ? (
                      <div style={{fontSize:'12px',color:'#76b900'}}>This runner can accept the job. It will pick it up on the next heartbeat.</div>
                    ) : (
                      <div>
                        {r.blockers.map((b, i) => (
                          <div key={i} style={{display:'flex',alignItems:'flex-start',gap:'8px',marginBottom:'4px',fontSize:'12px',color:'var(--nv-text-muted)',lineHeight:'1.5'}}>
                            <span style={{color:'#ff5050',fontWeight:700,flexShrink:0}}>✕</span>
                            <span>{b}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ===== Runs View ===== */
function RunsView({ runs, datasets, loading, filterDataset, setFilterDataset, filterCommit, setFilterCommit, onRefresh, onSelectRun, onDeleteRun, onTrigger, jobs, runners, githubRepoUrl, onViewLogs }) {
  const activeJobs = (jobs || []).filter(j => j.status==="running" || j.status==="pending" || j.status==="cancelling");
  const [diagnoseJobId, setDiagnoseJobId] = useState(null);
  const [matrixModalId, setMatrixModalId] = useState(null);
  const pg = usePagination(runs, 25);

  const runnerMap = useMemo(() => {
    const m = {};
    (runners || []).forEach(r => { m[r.id] = r; });
    return m;
  }, [runners]);

  const { matrixGroups, standaloneJobs } = useMemo(() => {
    const groups = {};
    const standalone = [];
    activeJobs.forEach(j => {
      if (j.matrix_run_id) {
        if (!groups[j.matrix_run_id]) groups[j.matrix_run_id] = { id: j.matrix_run_id, name: j.matrix_name || "Matrix", jobs: [], gitCommit: j.git_commit };
        groups[j.matrix_run_id].jobs.push(j);
      } else {
        standalone.push(j);
      }
    });
    return { matrixGroups: Object.values(groups), standaloneJobs: standalone };
  }, [activeJobs]);

  const handleCancel = async (jobId) => {
    if (!confirm("Cancel this job?")) return;
    try {
      await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
      onRefresh();
    } catch (e) { console.error(e); }
  };
  const handleForceDelete = async (jobId) => {
    if (!confirm("Force delete this job? This permanently removes it and cannot be undone.")) return;
    try {
      await fetch(`/api/jobs/${jobId}`, { method: "DELETE" });
      onRefresh();
    } catch (e) { console.error(e); }
  };
  const handleCancelMatrix = async (matrixRunId, activeCt) => {
    if (!confirm(`Cancel all ${activeCt} active jobs in this matrix run?`)) return;
    try {
      await fetch(`/api/matrix-runs/${matrixRunId}/cancel`, { method: "POST" });
      onRefresh();
    } catch (e) { console.error(e); }
  };

  return (
    <>
      {(matrixGroups.length > 0 || standaloneJobs.length > 0) && (
        <div style={{marginBottom:'20px'}}>
          <div className="section-title" style={{marginBottom:'8px'}}>Active Jobs</div>
          <div className="card" style={{padding:'0'}}>
            {/* Matrix group rows */}
            {matrixGroups.map(g => {
              const counts = { pending: 0, running: 0, cancelling: 0 };
              g.jobs.forEach(j => { counts[j.status] = (counts[j.status] || 0) + 1; });
              const activeCt = counts.pending + counts.running + counts.cancelling;
              const hasRunning = counts.running > 0 || counts.cancelling > 0;
              return (
                <div key={g.id} style={{borderBottom:'1px solid var(--nv-border)'}}>
                  <div style={{
                    display:'flex',alignItems:'center',justifyContent:'space-between',
                    padding:'10px 16px',cursor:'pointer',
                    background:'rgba(118,185,0,0.02)',
                  }} onClick={() => setMatrixModalId(g.id)}>
                    <div style={{display:'flex',alignItems:'center',gap:'10px',flexWrap:'wrap'}}>
                      {hasRunning && <span className="spinner"></span>}
                      <span style={{
                        fontSize:'10px',fontWeight:700,textTransform:'uppercase',letterSpacing:'0.05em',
                        padding:'3px 8px',borderRadius:'4px',
                        background:'rgba(118,185,0,0.1)',color:'var(--nv-green)',border:'1px solid rgba(118,185,0,0.2)',
                      }}>Matrix</span>
                      <span style={{color:'#fff',fontWeight:600,fontSize:'13px'}}>{g.name}</span>
                      <span style={{color:'var(--nv-text-dim)',fontSize:'12px'}}>{g.jobs.length} jobs</span>
                      <span style={{fontSize:'11px',display:'flex',gap:'8px',color:'var(--nv-text-muted)'}}>
                        {counts.running > 0 && <span style={{color:'var(--nv-green)'}}>● {counts.running} running</span>}
                        {counts.pending > 0 && <span style={{color:'#fcd34d'}}>● {counts.pending} pending</span>}
                        {counts.cancelling > 0 && <span style={{color:'#ff8844'}}>● {counts.cancelling} cancelling</span>}
                      </span>
                      {g.gitCommit && <CommitLink sha={g.gitCommit} repoUrl={githubRepoUrl} />}
                    </div>
                    <div style={{display:'flex',alignItems:'center',gap:'8px'}} onClick={e => e.stopPropagation()}>
                      <button className="btn btn-secondary" style={{fontSize:'11px',padding:'3px 10px'}}
                        onClick={() => setMatrixModalId(g.id)} title="View all jobs">
                        <IconSearch /> Details
                      </button>
                      <button className="btn" style={{fontSize:'11px',padding:'3px 10px',background:'rgba(255,80,80,0.12)',color:'#ff5050',border:'1px solid rgba(255,80,80,0.2)'}}
                        onClick={() => handleCancelMatrix(g.id, activeCt)} title="Cancel all matrix jobs">
                        <IconStop /> Cancel All
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}

            {/* Standalone (non-matrix) job rows */}
            {standaloneJobs.map(j => (
              <div key={j.id} style={{
                display:'flex',alignItems:'center',justifyContent:'space-between',
                padding:'10px 16px',borderBottom:'1px solid var(--nv-border)',
              }}>
                <div style={{display:'flex',alignItems:'center',gap:'10px',flexWrap:'wrap'}}>
                  <JobStatusBadge status={j.status} />
                  {(j.status==="running" || j.status==="cancelling") && <span className="spinner"></span>}
                  <span style={{color:'#fff',fontWeight:500,fontSize:'13px'}}>{j.dataset}</span>
                  {j.preset && <span style={{color:'var(--nv-text-muted)',fontSize:'12px'}}>{j.preset}</span>}
                  <TriggerSourceBadge source={j.trigger_source} />
                  {j.git_commit && <CommitLink sha={j.git_commit} repoUrl={githubRepoUrl} />}
                  {j.assigned_runner_id && (() => {
                    const r = runnerMap[j.assigned_runner_id];
                    const label = r ? (r.name || r.hostname || `#${j.assigned_runner_id}`) : `Runner #${j.assigned_runner_id}`;
                    return <span style={{color:'var(--nv-text-dim)',fontSize:'11px',background:'var(--nv-bg)',padding:'2px 6px',borderRadius:'4px',border:'1px solid var(--nv-border)'}}>{label}</span>;
                  })()}
                </div>
                <div style={{display:'flex',alignItems:'center',gap:'8px'}}>
                  {j.status==="pending" && (
                    <button className="btn btn-secondary" style={{fontSize:'11px',padding:'3px 8px'}}
                      onClick={() => setDiagnoseJobId(j.id)} title="Why isn't this job running?">
                      <IconSearch /> Diagnose
                    </button>
                  )}
                  <button className="btn btn-secondary" style={{fontSize:'11px',padding:'3px 8px'}}
                    onClick={() => onViewLogs && onViewLogs(j.id)} title="View Logs">
                    <IconTerminal /> Logs
                  </button>
                  {(j.status==="pending" || j.status==="running") && (
                    <button className="btn" style={{fontSize:'11px',padding:'3px 8px',background:'rgba(255,80,80,0.12)',color:'#ff5050',border:'1px solid rgba(255,80,80,0.2)'}}
                      onClick={() => handleCancel(j.id)} title="Cancel Job">
                      <IconStop /> Cancel
                    </button>
                  )}
                  {j.status==="cancelling" && (
                    <button className="btn" style={{fontSize:'11px',padding:'3px 8px',background:'rgba(255,80,80,0.2)',color:'#ff5050',border:'1px solid rgba(255,80,80,0.3)',fontWeight:700}}
                      onClick={() => handleForceDelete(j.id)} title="Force delete this stuck job">
                      <IconTrash /> Force Delete
                    </button>
                  )}
                  <span className="mono" style={{fontSize:'11px',color:'var(--nv-text-dim)'}}>{j.id}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'20px',flexWrap:'wrap',gap:'12px'}}>
        <div style={{display:'flex',gap:'8px',alignItems:'center'}}>
          <button className="btn btn-primary" onClick={onTrigger}><IconPlay /> Trigger Run</button>
          {runs.filter(r=>r.success===0).length > 0 && (
            <button className="btn btn-sm" onClick={async () => {
              const failedIds = runs.filter(r=>r.success===0).map(r=>r.id);
              if (!confirm(`Delete ${failedIds.length} failed run${failedIds.length!==1?'s':''}? This cannot be undone.`)) return;
              try {
                await fetch("/api/runs/delete-bulk", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({run_ids:failedIds})});
                onRefresh();
              } catch {}
            }} title="Delete all failed runs"
              style={{background:'rgba(255,80,80,0.08)',color:'#ff5050',border:'1px solid rgba(255,80,80,0.15)',fontSize:'12px'}}>
              <IconTrash /> Delete All Failed ({runs.filter(r=>r.success===0).length})
            </button>
          )}
        </div>
        <div style={{display:'flex',gap:'12px',alignItems:'center',flexWrap:'wrap'}}>
          <select value={filterDataset} onChange={e=>setFilterDataset(e.target.value)} className="select" style={{minWidth:'200px'}}>
            <option value="">All datasets</option>
            {datasets.map(d=><option key={d} value={d}>{d}</option>)}
          </select>
          <div style={{position:'relative'}}>
            <input type="text" placeholder="Filter by commit\u2026" value={filterCommit}
              onChange={e=>setFilterCommit(e.target.value)} className="input" style={{minWidth:'220px',paddingLeft:'34px'}} />
            <span style={{position:'absolute',left:'10px',top:'50%',transform:'translateY(-50%)',color:'var(--nv-text-dim)',width:'16px',height:'16px'}}><IconSearch /></span>
          </div>
          <button className="btn btn-secondary btn-icon" onClick={onRefresh} title="Refresh"><IconRefresh /></button>
        </div>
      </div>
      <div className="card">
        <div style={{overflowX:'auto'}}>
          <table className="runs-table">
            <thead>
              <tr>
                <th>Status</th><th>Source</th><th>Timestamp</th><th>Host</th><th style={{textAlign:'right'}}>GPUs</th><th>Commit</th><th>Exec Commit</th><th>Dataset</th>
                <th>Preset</th><th style={{textAlign:'right'}}>Pages</th>
                <th style={{textAlign:'right'}}>PPS</th><th style={{textAlign:'right'}}>Recall@5</th>
                <th style={{textAlign:'right'}}>Ingest (s)</th>
                <th style={{width:'40px'}}></th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan="14" style={{textAlign:'center',padding:'60px',color:'var(--nv-text-muted)'}}>
                  <div className="spinner spinner-lg" style={{margin:'0 auto 12px'}}></div><div>Loading runs…</div>
                </td></tr>
              ) : runs.length === 0 ? (
                <tr><td colSpan="14" style={{textAlign:'center',padding:'60px',color:'var(--nv-text-muted)'}}>
                  <div style={{marginBottom:'8px',fontSize:'15px'}}>No runs found</div>
                  <div style={{fontSize:'12px',color:'var(--nv-text-dim)'}}>
                    Trigger a run or use <code className="mono" style={{background:'rgba(255,255,255,0.06)',padding:'2px 6px',borderRadius:'4px'}}>retriever harness backfill</code> to import existing results.
                  </div>
                </td></tr>
              ) : pg.pageData.map(run => (
                <tr key={run.id} onClick={()=>onSelectRun(run.id)} style={run.success===0 ? {background:'rgba(255,60,60,0.04)'} : undefined}>
                  <td>
                    <StatusBadge success={run.success} />
                    {run.failure_reason && <div style={{fontSize:'11px',color:'#ff5050',marginTop:'2px',maxWidth:'140px',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}} title={run.failure_reason}>{run.failure_reason}</div>}
                  </td>
                  <td><TriggerSourceBadge source={run.trigger_source} /></td>
                  <td style={{color:'var(--nv-text-muted)',whiteSpace:'nowrap'}}>{fmtTs(run.timestamp)}</td>
                  <td style={{color:'var(--nv-text-muted)',whiteSpace:'nowrap',fontSize:'12px'}}>
                    {run.hostname || "\u2014"}
                    {run.ray_cluster_mode === "existing" && (
                      <span title="Connected to existing Ray cluster" style={{marginLeft:'6px',fontSize:'10px',color:'#64b4ff',background:'rgba(100,180,255,0.1)',padding:'1px 5px',borderRadius:'4px',border:'1px solid rgba(100,180,255,0.2)',whiteSpace:'nowrap'}}>Ray</span>
                    )}
                  </td>
                  <td style={{textAlign:'right',color:'var(--nv-text-muted)',fontSize:'12px'}}>{run.num_gpus != null ? run.num_gpus : "\u2014"}</td>
                  <td><CommitLink sha={run.git_commit} repoUrl={githubRepoUrl} /></td>
                  <td><CommitLink sha={run.execution_commit} repoUrl={githubRepoUrl} /></td>
                  <td style={{color:'#fff',fontWeight:500}}>{run.dataset}</td>
                  <td style={{color:'var(--nv-text-muted)'}}>{run.preset || "\u2014"}</td>
                  <td style={{textAlign:'right'}}>{fmt(run.pages,0)}</td>
                  <td style={{textAlign:'right',color:'#fff',fontWeight:500}}>{fmt(run.pages_per_sec)}</td>
                  <td style={{textAlign:'right',color:'var(--nv-green)',fontWeight:600}}>{fmt(run.recall_5,3)}</td>
                  <td style={{textAlign:'right'}}>{fmt(run.ingest_secs)}</td>
                  <td onClick={e=>e.stopPropagation()}>
                    <button className="btn btn-sm btn-icon" onClick={() => {
                      if (confirm(`Delete run #${run.id}? This cannot be undone.`)) { onDeleteRun && onDeleteRun(run.id); }
                    }} title="Delete run"
                      style={{background:'transparent',color:'var(--nv-text-dim)',border:'none',padding:'4px',opacity:0.5,transition:'opacity 0.15s'}}
                      onMouseEnter={e=>e.currentTarget.style.opacity='1'} onMouseLeave={e=>e.currentTarget.style.opacity='0.5'}>
                      <IconTrash />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Pagination page={pg.page} totalPages={pg.totalPages} totalItems={pg.totalItems}
          pageSize={pg.pageSize} onPageChange={pg.setPage} onPageSizeChange={pg.setPageSize} />
      </div>
      {diagnoseJobId && <JobDiagnoseModal jobId={diagnoseJobId} onClose={() => setDiagnoseJobId(null)} />}
      {matrixModalId && (
        <MatrixJobsModal
          matrixRunId={matrixModalId}
          matrixName={(matrixGroups.find(g => g.id === matrixModalId) || {}).name}
          jobs={jobs || []}
          runnerMap={runnerMap}
          githubRepoUrl={githubRepoUrl}
          onClose={() => setMatrixModalId(null)}
          onRefresh={onRefresh}
          onViewLogs={onViewLogs}
          onDiagnose={setDiagnoseJobId}
        />
      )}
    </>
  );
}
