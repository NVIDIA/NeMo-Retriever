/* ===== Log Viewer Modal ===== */
function LogViewerModal({ jobId, onClose }) {
  const [logData, setLogData] = useState({ log_tail: [], status: null });
  const [autoScroll, setAutoScroll] = useState(true);
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

  useEffect(() => { fetchLogs(); }, [fetchLogs]);

  useEffect(() => {
    const isActive = logData.status === "running" || logData.status === "cancelling";
    if (!isActive) return;
    const iv = setInterval(fetchLogs, 3000);
    return () => clearInterval(iv);
  }, [fetchLogs, logData.status]);

  useEffect(() => {
    if (autoScroll && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logData.log_tail, autoScroll]);

  const handleCancel = async () => {
    if (!confirm("Cancel this job?")) return;
    try {
      await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
      fetchLogs();
    } catch (e) { console.error(e); }
  };

  const isActive = logData.status === "running" || logData.status === "cancelling";
  const lines = logData.log_tail || [];

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{maxWidth:'900px'}} onClick={e => e.stopPropagation()}>
        <div className="modal-head">
          <div style={{display:'flex',alignItems:'center',gap:'12px'}}>
            <h2 style={{margin:0,fontSize:'16px',color:'#fff'}}>Job Logs</h2>
            <span className="mono" style={{fontSize:'12px',color:'var(--nv-text-dim)'}}>{jobId}</span>
            {logData.status && <JobStatusBadge status={logData.status} />}
            {isActive && <span className="spinner"></span>}
          </div>
          <button className="btn btn-ghost btn-icon" onClick={onClose} style={{borderRadius:'50%'}}><IconX /></button>
        </div>
        <div className="modal-body" style={{padding:'16px'}}>
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'10px'}}>
            <span style={{fontSize:'12px',color:'var(--nv-text-muted)'}}>{lines.length} line{lines.length!==1?'s':''}</span>
            <div style={{display:'flex',gap:'8px',alignItems:'center'}}>
              <label style={{display:'flex',alignItems:'center',gap:'4px',fontSize:'12px',color:'var(--nv-text-muted)',cursor:'pointer'}}>
                <input type="checkbox" checked={autoScroll} onChange={e => setAutoScroll(e.target.checked)} /> Auto-scroll
              </label>
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
