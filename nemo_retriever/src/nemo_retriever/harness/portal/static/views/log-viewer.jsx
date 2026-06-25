/* ===== Log Viewer Modal ===== */
function LogViewerModal({ jobId, onClose }) {
  const [logData, setLogData] = useState({ log_tail: [], status: null });
  const [jobDetail, setJobDetail] = useState(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [showPipList, setShowPipList] = useState(false);
  const [showPlan, setShowPlan] = useState(false);
  const [showRayStats, setShowRayStats] = useState(false);
  const logRef = useRef(null);

  const fetchLogs = useCallback(async () => {
    if (!jobId) return;
    try {
      const resp = await fetch(`/api/jobs/${jobId}/logs`);
      if (resp.ok) {
        const data = await resp.json();
        setLogData(data);
      }
    } catch (e) { console.error(e); }
  }, [jobId]);

  const fetchJobDetail = useCallback(async () => {
    if (!jobId) return;
    try {
      const resp = await fetch(`/api/jobs/${jobId}`);
      if (resp.ok) setJobDetail(await resp.json());
    } catch {}
  }, [jobId]);

  useEffect(() => { fetchLogs(); fetchJobDetail(); }, [fetchLogs, fetchJobDetail]);

  useEffect(() => {
    const isActive = logData.status === "running" || logData.status === "cancelling";
    if (!isActive) return;
    const iv = setInterval(() => { fetchLogs(); fetchJobDetail(); }, 3000);
    return () => clearInterval(iv);
  }, [fetchLogs, fetchJobDetail, logData.status]);

  useEffect(() => {
    if (autoScroll && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logData.log_tail, autoScroll]);

  const handleCancel = async () => {
    if (!confirm("Cancel this job?")) return;
    try {
      await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
      fetchLogs(); fetchJobDetail();
    } catch (e) { console.error(e); }
  };

  const isActive = logData.status === "running" || logData.status === "cancelling";
  const lines = logData.log_tail || [];
  const jd = jobDetail || {};
  const isFailed = jd.status === "failed" || jd.status === "error";
  const resultData = jd.result || {};

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{maxWidth:'900px'}} onClick={e => e.stopPropagation()}>
        <div className="modal-head">
          <div style={{display:'flex',alignItems:'center',gap:'12px',flexWrap:'wrap'}}>
            <h2 style={{margin:0,fontSize:'16px',color:'#fff'}}>Job Logs</h2>
            <span className="mono" style={{fontSize:'12px',color:'var(--nv-text-dim)'}}>{jobId}</span>
            {logData.status && <JobStatusBadge status={logData.status} />}
            {isActive && <span className="spinner"></span>}
            {jd.dataset && <span style={{fontSize:'12px',color:'var(--nv-text-muted)'}}>{jd.dataset}</span>}
            {jd.preset && <span style={{fontSize:'12px',color:'var(--nv-text-dim)'}}>{jd.preset}</span>}
          </div>
          <button className="btn btn-ghost btn-icon" onClick={onClose} style={{borderRadius:'50%'}}><IconX /></button>
        </div>
        <div className="modal-body" style={{padding:'16px'}}>

          {isFailed && (jd.error || resultData.failure_reason) && (
            <div style={{
              marginBottom:'12px',padding:'12px 16px',borderRadius:'8px',
              background:'rgba(255,60,60,0.08)',border:'1px solid rgba(255,60,60,0.2)',
            }}>
              <div style={{display:'flex',alignItems:'center',gap:'8px',marginBottom:'6px'}}>
                <span style={{fontSize:'14px'}}>&#x26A0;</span>
                <span style={{fontSize:'13px',fontWeight:700,color:'#ff5050'}}>
                  {resultData.failure_reason || 'Job Failed'}
                </span>
                {resultData.return_code != null && (
                  <span className="mono" style={{fontSize:'11px',color:'var(--nv-text-muted)'}}>
                    Exit code: {resultData.return_code}
                  </span>
                )}
              </div>
              {resultData.error_detail && (
                <pre className="mono" style={{
                  fontSize:'11px',color:'#ff8888',margin:'6px 0 0',
                  whiteSpace:'pre-wrap',wordBreak:'break-all',lineHeight:'1.5',
                  maxHeight:'200px',overflow:'auto',
                  background:'rgba(0,0,0,0.2)',padding:'8px',borderRadius:'4px',
                }}>{resultData.error_detail}</pre>
              )}
              {jd.error && jd.error !== resultData.failure_reason && !resultData.error_detail && (
                <pre className="mono" style={{
                  fontSize:'11px',color:'#ff8888',margin:0,
                  whiteSpace:'pre-wrap',wordBreak:'break-all',lineHeight:'1.5',
                  maxHeight:'120px',overflow:'auto',
                }}>{jd.error}</pre>
              )}
            </div>
          )}

          {resultData.nsys_status && resultData.nsys_status.requested && (
            <div style={{
              marginBottom:'12px',padding:'10px 14px',borderRadius:'8px',
              background: resultData.nsys_status.found ? 'rgba(118,185,0,0.08)' : resultData.nsys_status.enabled ? 'rgba(255,165,0,0.08)' : 'rgba(255,60,60,0.08)',
              border: `1px solid ${resultData.nsys_status.found ? 'rgba(118,185,0,0.2)' : resultData.nsys_status.enabled ? 'rgba(255,165,0,0.2)' : 'rgba(255,60,60,0.2)'}`,
            }}>
              <div style={{display:'flex',alignItems:'center',gap:'8px',flexWrap:'wrap'}}>
                <span style={{fontSize:'13px'}}>{resultData.nsys_status.found ? '\u2705' : resultData.nsys_status.enabled ? '\u26A0\uFE0F' : '\u274C'}</span>
                <span style={{fontSize:'12px',fontWeight:600,color: resultData.nsys_status.found ? '#76b900' : resultData.nsys_status.enabled ? '#ffa500' : '#ff5050'}}>
                  Nsight Systems Profile: {resultData.nsys_status.found ? 'Captured' : resultData.nsys_status.enabled ? 'No Report Generated' : 'Not Available'}
                </span>
                {resultData.nsys_status.files && resultData.nsys_status.files.length > 0 && (
                  <span style={{fontSize:'11px',color:'var(--nv-text-muted)'}}>
                    ({resultData.nsys_status.files.map(f => `${f.name}: ${f.size_mb} MB`).join(', ')})
                  </span>
                )}
              </div>
              {resultData.nsys_status.error && (
                <div style={{fontSize:'11px',color:'var(--nv-text-muted)',marginTop:'4px'}}>{resultData.nsys_status.error}</div>
              )}
            </div>
          )}

          <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'10px'}}>
            <span style={{fontSize:'12px',color:'var(--nv-text-muted)'}}>{lines.length} line{lines.length!==1?'s':''}</span>
            <div style={{display:'flex',gap:'8px',alignItems:'center'}}>
              {lines.length > 0 && (
                <button className="btn btn-secondary" style={{fontSize:'11px',padding:'3px 8px'}}
                  onClick={() => navigator.clipboard.writeText(lines.join('\n'))}>
                  Copy All
                </button>
              )}
              {isActive && (
                <label style={{display:'flex',alignItems:'center',gap:'4px',fontSize:'12px',color:'var(--nv-text-muted)',cursor:'pointer'}}>
                  <input type="checkbox" checked={autoScroll} onChange={e => setAutoScroll(e.target.checked)} /> Auto-scroll
                </label>
              )}
              <button className="btn btn-secondary" style={{fontSize:'11px',padding:'3px 8px'}} onClick={fetchLogs}>
                <IconRefresh /> Refresh
              </button>
            </div>
          </div>
          <div className="log-viewer" ref={logRef}>
            {lines.length === 0 ? (
              <div style={{color:'var(--nv-text-dim)',fontStyle:'italic'}}>
                {isActive ? "Waiting for log output..." : "No log output available."}
              </div>
            ) : (
              lines.map((line, i) => <div key={i} className="log-line">{line}</div>)
            )}
          </div>

          {resultData.requested_plan && Array.isArray(resultData.requested_plan) && (
            <div style={{marginTop:'12px'}}>
              <button className="btn btn-secondary" style={{fontSize:'11px',padding:'4px 10px',display:'flex',alignItems:'center',gap:'6px'}}
                onClick={() => setShowPlan(v => !v)}>
                <span style={{transform: showPlan ? 'rotate(90deg)' : 'rotate(0deg)', transition:'transform 0.15s', display:'inline-block'}}>&#9654;</span>
                Requested Plan ({resultData.requested_plan.length} stage{resultData.requested_plan.length!==1?'s':''})
              </button>
              {showPlan && (
                <div style={{marginTop:'8px',position:'relative'}}>
                  <button className="btn btn-secondary" style={{position:'absolute',top:'6px',right:'6px',fontSize:'10px',padding:'2px 6px',zIndex:1}}
                    onClick={() => navigator.clipboard.writeText(JSON.stringify(resultData.requested_plan, null, 2))}>Copy</button>
                  <div style={{
                    background:'rgba(0,0,0,0.25)',padding:'10px',borderRadius:'6px',
                    border:'1px solid rgba(255,255,255,0.06)',maxHeight:'300px',overflow:'auto',
                  }}>
                    <table style={{width:'100%',borderCollapse:'collapse',fontSize:'11px'}}>
                      <thead>
                        <tr style={{borderBottom:'1px solid rgba(255,255,255,0.1)'}}>
                          <th style={{textAlign:'left',padding:'4px 8px',color:'var(--nv-text-muted)',fontWeight:600}}>Stage</th>
                          <th style={{textAlign:'left',padding:'4px 8px',color:'var(--nv-text-muted)',fontWeight:600}}>Type</th>
                          <th style={{textAlign:'center',padding:'4px 8px',color:'var(--nv-text-muted)',fontWeight:600}}>GPUs</th>
                          <th style={{textAlign:'center',padding:'4px 8px',color:'var(--nv-text-muted)',fontWeight:600}}>CPUs</th>
                          <th style={{textAlign:'center',padding:'4px 8px',color:'var(--nv-text-muted)',fontWeight:600}}>Batch</th>
                          <th style={{textAlign:'center',padding:'4px 8px',color:'var(--nv-text-muted)',fontWeight:600}}>Concurrency</th>
                        </tr>
                      </thead>
                      <tbody>
                        {resultData.requested_plan.map((s, i) => (
                          <tr key={i} style={{borderBottom:'1px solid rgba(255,255,255,0.04)'}}>
                            <td style={{padding:'4px 8px',color:'#fff'}}>{s.display_name || s.stage}</td>
                            <td style={{padding:'4px 8px'}}>
                              <span style={{
                                fontSize:'10px',padding:'2px 6px',borderRadius:'4px',
                                background: s.type==='gpu' ? 'rgba(118,185,0,0.15)' : s.type==='source' ? 'rgba(0,150,255,0.15)' : s.type==='sink' ? 'rgba(255,165,0,0.15)' : 'rgba(150,150,150,0.15)',
                                color: s.type==='gpu' ? '#76b900' : s.type==='source' ? '#4da6ff' : s.type==='sink' ? '#ffa500' : '#aaa',
                              }}>{s.type}</span>
                            </td>
                            <td style={{padding:'4px 8px',textAlign:'center',color:'var(--nv-text-muted)'}}>{s.num_gpus != null ? s.num_gpus : '-'}</td>
                            <td style={{padding:'4px 8px',textAlign:'center',color:'var(--nv-text-muted)'}}>{s.num_cpus != null ? s.num_cpus : '-'}</td>
                            <td style={{padding:'4px 8px',textAlign:'center',color:'var(--nv-text-muted)'}}>{s.batch_size != null ? s.batch_size : '-'}</td>
                            <td style={{padding:'4px 8px',textAlign:'center',color:'var(--nv-text-muted)'}}>{s.concurrency != null ? s.concurrency : '-'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {resultData.ray_stats && (
            <div style={{marginTop:'12px'}}>
              <button className="btn btn-secondary" style={{fontSize:'11px',padding:'4px 10px',display:'flex',alignItems:'center',gap:'6px'}}
                onClick={() => setShowRayStats(v => !v)}>
                <span style={{transform: showRayStats ? 'rotate(90deg)' : 'rotate(0deg)', transition:'transform 0.15s', display:'inline-block'}}>&#9654;</span>
                Ray Execution Stats
              </button>
              {showRayStats && (
                <div style={{marginTop:'8px',position:'relative'}}>
                  <button className="btn btn-secondary" style={{position:'absolute',top:'6px',right:'6px',fontSize:'10px',padding:'2px 6px',zIndex:1}}
                    onClick={() => navigator.clipboard.writeText(resultData.ray_stats)}>Copy</button>
                  <pre className="mono" style={{
                    fontSize:'11px',color:'var(--nv-text-muted)',margin:0,
                    whiteSpace:'pre',lineHeight:'1.4',
                    maxHeight:'300px',overflow:'auto',
                    background:'rgba(0,0,0,0.25)',padding:'10px',borderRadius:'6px',
                    border:'1px solid rgba(255,255,255,0.06)',
                  }}>{resultData.ray_stats}</pre>
                </div>
              )}
            </div>
          )}

          {jd.pip_list && (
            <div style={{marginTop:'12px'}}>
              <button className="btn btn-secondary" style={{fontSize:'11px',padding:'4px 10px',display:'flex',alignItems:'center',gap:'6px'}}
                onClick={() => setShowPipList(v => !v)}>
                <span style={{transform: showPipList ? 'rotate(90deg)' : 'rotate(0deg)', transition:'transform 0.15s', display:'inline-block'}}>&#9654;</span>
                Installed Packages ({jd.pip_list.split('\n').length} lines)
              </button>
              {showPipList && (
                <div style={{marginTop:'8px',position:'relative'}}>
                  <button className="btn btn-secondary" style={{position:'absolute',top:'6px',right:'6px',fontSize:'10px',padding:'2px 6px',zIndex:1}}
                    onClick={() => navigator.clipboard.writeText(jd.pip_list)}>Copy</button>
                  <pre className="mono" style={{
                    fontSize:'11px',color:'var(--nv-text-muted)',margin:0,
                    whiteSpace:'pre',lineHeight:'1.4',
                    maxHeight:'250px',overflow:'auto',
                    background:'rgba(0,0,0,0.25)',padding:'10px',borderRadius:'6px',
                    border:'1px solid rgba(255,255,255,0.06)',
                  }}>{jd.pip_list}</pre>
                </div>
              )}
            </div>
          )}
        </div>
        <div className="modal-foot">
          {(logData.status === "running" || logData.status === "pending") && (
            <button className="btn" style={{background:'rgba(255,80,80,0.12)',color:'#ff5050',border:'1px solid rgba(255,80,80,0.2)'}}
              onClick={handleCancel}><IconStop /> Cancel Job</button>
          )}
          <button className="btn btn-secondary" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}
