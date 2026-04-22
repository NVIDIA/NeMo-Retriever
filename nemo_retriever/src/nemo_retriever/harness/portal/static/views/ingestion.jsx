/* ===== Ingestion Playground View ===== */
function IngestionView({ jobs, onViewLogs }) {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [presets, setPresets] = useState([]);
  const [runners, setRunners] = useState([]);
  const [preset, setPreset] = useState("");
  const [runnerId, setRunnerId] = useState("");
  const [inputType, setInputType] = useState("pdf");
  const [triggering, setTriggering] = useState(false);
  const [triggered, setTriggered] = useState(null);
  const [error, setError] = useState("");
  const [sessions, setSessions] = useState([]);
  const [sessionsLoading, setSessionsLoading] = useState(true);

  useEffect(() => {
    fetch("/api/config").then(r => r.json()).then(cfg => {
      setPresets(cfg.presets || []);
      if (cfg.presets?.length) setPreset(cfg.presets[0]);
    }).catch(() => {});
    fetch("/api/runners").then(r => r.json()).then(setRunners).catch(() => {});
    fetchSessions();
  }, []);

  async function fetchSessions() {
    setSessionsLoading(true);
    try {
      const res = await fetch("/api/playground/sessions");
      setSessions(await res.json());
    } catch {} finally { setSessionsLoading(false); }
  }

  function handleFileChange(e) {
    setFiles(Array.from(e.target.files || []));
    setUploadResult(null);
    setTriggered(null);
    setError("");
  }

  function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    const dropped = Array.from(e.dataTransfer.files || []);
    if (dropped.length) {
      setFiles(dropped);
      setUploadResult(null);
      setTriggered(null);
      setError("");
    }
  }

  async function handleUpload() {
    if (files.length === 0) return;
    setUploading(true);
    setError("");
    setTriggered(null);
    try {
      const fd = new FormData();
      files.forEach(f => fd.append("files", f));
      const res = await fetch("/api/playground/upload", { method: "POST", body: fd });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      const result = await res.json();
      setUploadResult(result);
      fetchSessions();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function handleTrigger() {
    if (!uploadResult) return;
    setTriggering(true);
    setError("");
    try {
      const payload = {
        session_id: uploadResult.session_id,
        preset: preset || null,
        runner_id: runnerId ? parseInt(runnerId, 10) : null,
        input_type: inputType,
      };
      const res = await fetch("/api/playground/ingest", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      setTriggered(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setTriggering(false);
    }
  }

  async function handleDeleteSession(sessionId) {
    if (!confirm("Delete this upload session and its files?")) return;
    try {
      await fetch(`/api/playground/sessions/${sessionId}`, { method: "DELETE" });
      fetchSessions();
      if (uploadResult && uploadResult.session_id === sessionId) {
        setUploadResult(null);
        setTriggered(null);
      }
    } catch {}
  }

  function formatBytes(b) {
    if (b < 1024) return b + " B";
    if (b < 1024 * 1024) return (b / 1024).toFixed(1) + " KB";
    return (b / (1024 * 1024)).toFixed(1) + " MB";
  }

  const onlineRunners = runners.filter(r => r.status === "online" || r.status === "paused");
  const playgroundJobs = (jobs || []).filter(j => j.trigger_source === "playground" && (j.status === "running" || j.status === "pending" || j.status === "cancelling"));
  const labelStyle = {display:'block',fontSize:'11px',fontWeight:600,color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.04em',marginBottom:'6px'};

  return (
    <>
      {/* Active playground jobs */}
      {playgroundJobs.length > 0 && (
        <div style={{marginBottom:'20px'}}>
          <div className="section-title" style={{marginBottom:'8px'}}>Active Ingestion Jobs</div>
          <div className="card" style={{padding:'0'}}>
            {playgroundJobs.map(j => (
              <div key={j.id} style={{display:'flex',alignItems:'center',justifyContent:'space-between',padding:'10px 16px',borderBottom:'1px solid var(--nv-border)'}}>
                <div style={{display:'flex',alignItems:'center',gap:'10px'}}>
                  <JobStatusBadge status={j.status} />
                  {(j.status==="running" || j.status==="cancelling") && <span className="spinner"></span>}
                  <span style={{color:'#fff',fontWeight:500,fontSize:'13px'}}>{j.dataset}</span>
                  {j.preset && <span style={{color:'var(--nv-text-muted)',fontSize:'12px'}}>{j.preset}</span>}
                </div>
                <div style={{display:'flex',alignItems:'center',gap:'8px'}}>
                  {(j.status==="running" || j.status==="cancelling") && onViewLogs && (
                    <button className="btn btn-secondary" style={{fontSize:'11px',padding:'3px 8px'}}
                      onClick={() => onViewLogs(j.id)} title="View Logs">
                      <IconTerminal /> Logs
                    </button>
                  )}
                  <span className="mono" style={{fontSize:'11px',color:'var(--nv-text-dim)'}}>{j.id}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'20px',marginBottom:'20px'}}>
        {/* Upload Card */}
        <div className="card" style={{padding:'24px'}}>
          <div className="section-title" style={{marginBottom:'14px'}}>Upload Documents</div>
          <div
            onDragOver={e => { e.preventDefault(); e.stopPropagation(); }}
            onDrop={handleDrop}
            style={{
              border:'2px dashed var(--nv-border)',borderRadius:'12px',padding:'32px 20px',
              textAlign:'center',cursor:'pointer',transition:'border-color 0.2s',
              background:'rgba(255,255,255,0.02)',marginBottom:'16px',
            }}
            onClick={() => document.getElementById('pg-file-input')?.click()}
            onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--nv-green)'}
            onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--nv-border)'}
          >
            <input type="file" id="pg-file-input" multiple style={{display:'none'}} onChange={handleFileChange}
              accept=".pdf,.txt,.html,.doc,.docx,.pptx,.png,.jpg,.jpeg,.bmp,.tiff,.svg" />
            <div style={{fontSize:'36px',marginBottom:'10px',opacity:0.3}}><IconUpload /></div>
            <div style={{fontSize:'14px',color:'var(--nv-text-muted)',fontWeight:500}}>
              {files.length > 0
                ? `${files.length} file${files.length !== 1 ? 's' : ''} selected (${formatBytes(files.reduce((s,f) => s+f.size, 0))})`
                : "Drop files here or click to browse"}
            </div>
            <div style={{fontSize:'11px',color:'var(--nv-text-dim)',marginTop:'6px'}}>
              PDF, TXT, HTML, DOC, DOCX, PPTX, Images
            </div>
          </div>

          {files.length > 0 && (
            <div style={{marginBottom:'16px',maxHeight:'120px',overflowY:'auto',fontSize:'12px'}}>
              {files.map((f, i) => (
                <div key={i} style={{display:'flex',justifyContent:'space-between',padding:'4px 0',borderBottom:'1px solid var(--nv-border)',color:'var(--nv-text-muted)'}}>
                  <span className="mono" style={{overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap',maxWidth:'250px'}}>{f.name}</span>
                  <span style={{color:'var(--nv-text-dim)',whiteSpace:'nowrap',marginLeft:'12px'}}>{formatBytes(f.size)}</span>
                </div>
              ))}
            </div>
          )}

          <button className="btn btn-primary" onClick={handleUpload}
            disabled={uploading || files.length === 0}
            style={{width:'100%',justifyContent:'center'}}>
            {uploading ? <><span className="spinner" style={{marginRight:'8px'}}></span>Uploading…</> : <><IconUpload /> Upload Files</>}
          </button>

          {uploadResult && (
            <div style={{marginTop:'14px',padding:'12px',borderRadius:'8px',background:'rgba(118,185,0,0.08)',border:'1px solid rgba(118,185,0,0.2)'}}>
              <div style={{display:'flex',alignItems:'center',gap:'8px',marginBottom:'6px'}}>
                <IconCheck />
                <span style={{fontSize:'13px',fontWeight:600,color:'var(--nv-green)'}}>
                  {uploadResult.file_count} file{uploadResult.file_count !== 1 ? 's' : ''} uploaded
                </span>
              </div>
              <div className="mono" style={{fontSize:'11px',color:'var(--nv-text-dim)',wordBreak:'break-all'}}>
                Session: {uploadResult.session_id}
              </div>
            </div>
          )}
        </div>

        {/* Run Configuration Card */}
        <div className="card" style={{padding:'24px'}}>
          <div className="section-title" style={{marginBottom:'14px'}}>Run Configuration</div>
          <div style={{display:'flex',flexDirection:'column',gap:'14px'}}>
            <div>
              <label style={labelStyle}>Input Type</label>
              <select className="select" style={{width:'100%'}} value={inputType} onChange={e => setInputType(e.target.value)}>
                <option value="pdf">PDF</option>
                <option value="image">Image</option>
                <option value="text">Text</option>
              </select>
            </div>
            <div>
              <label style={labelStyle}>Preset</label>
              <select className="select" style={{width:'100%'}} value={preset} onChange={e => setPreset(e.target.value)}>
                <option value="">Default</option>
                {presets.map(p => <option key={p} value={p}>{p}</option>)}
              </select>
            </div>
            <div>
              <label style={labelStyle}>Runner</label>
              <select className="select" style={{width:'100%'}} value={runnerId} onChange={e => setRunnerId(e.target.value)}>
                <option value="">Any available runner</option>
                {onlineRunners.map(r => <option key={r.id} value={r.id}>{r.name || r.hostname} — {r.gpu_type || 'no GPU'} x{r.gpu_count || 0}{r.status==='paused'?' [PAUSED]':''}</option>)}
              </select>
              <div style={{fontSize:'11px',color:'var(--nv-text-dim)',marginTop:'4px'}}>
                {onlineRunners.length === 0
                  ? "No runners online — job will queue until one is available."
                  : `${onlineRunners.length} runner${onlineRunners.length !== 1 ? 's' : ''} online`}
              </div>
            </div>

            <button className="btn btn-primary" onClick={handleTrigger}
              disabled={triggering || !uploadResult}
              style={{width:'100%',justifyContent:'center',marginTop:'8px'}}>
              {triggering ? <><span className="spinner" style={{marginRight:'8px'}}></span>Starting…</> : <><IconPlay /> Start Ingestion</>}
            </button>

            {!uploadResult && (
              <div style={{fontSize:'12px',color:'var(--nv-text-dim)',textAlign:'center',fontStyle:'italic'}}>
                Upload documents first to enable ingestion.
              </div>
            )}

            {triggered && (
              <div style={{padding:'12px',borderRadius:'8px',background:'rgba(118,185,0,0.08)',border:'1px solid rgba(118,185,0,0.2)'}}>
                <div style={{display:'flex',alignItems:'center',gap:'8px',marginBottom:'6px'}}>
                  <IconCheck />
                  <span style={{fontSize:'13px',fontWeight:600,color:'var(--nv-green)'}}>Ingestion job queued</span>
                </div>
                <div style={{fontSize:'12px',color:'var(--nv-text-muted)'}}>
                  Job <span className="mono" style={{color:'#fff'}}>{triggered.job_id}</span> — {triggered.file_count} file{triggered.file_count !== 1 ? 's' : ''}
                </div>
                <div style={{fontSize:'11px',color:'var(--nv-text-dim)',marginTop:'4px'}}>
                  The job will be picked up by a runner on its next heartbeat. Check the Runs view for results.
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{marginBottom:'16px',padding:'12px 16px',borderRadius:'8px',background:'rgba(255,50,50,0.08)',border:'1px solid rgba(255,50,50,0.2)',color:'#ff5050',fontSize:'13px'}}>
          {error}
        </div>
      )}

      {/* Upload History */}
      <div className="card">
        <div style={{padding:'16px 20px',borderBottom:'1px solid var(--nv-border)',display:'flex',justifyContent:'space-between',alignItems:'center'}}>
          <div className="section-title" style={{margin:0}}>Upload Sessions</div>
          <button className="btn btn-secondary btn-icon btn-sm" onClick={fetchSessions} title="Refresh"><IconRefresh /></button>
        </div>
        {sessionsLoading ? (
          <div style={{padding:'40px',textAlign:'center',color:'var(--nv-text-muted)'}}>
            <span className="spinner spinner-lg" style={{display:'block',margin:'0 auto 12px'}}></span>Loading…
          </div>
        ) : sessions.length === 0 ? (
          <div style={{padding:'40px',textAlign:'center',color:'var(--nv-text-muted)',fontSize:'13px'}}>
            No upload sessions yet. Upload documents above to get started.
          </div>
        ) : (
          <div style={{overflowX:'auto'}}>
            <table className="runs-table">
              <thead>
                <tr><th>Session ID</th><th>Files</th><th>Size</th><th>Path</th><th>Actions</th></tr>
              </thead>
              <tbody>
                {sessions.map(s => (
                  <tr key={s.session_id}>
                    <td className="mono" style={{fontSize:'12px',color:'#fff',fontWeight:500}}>{s.session_id}</td>
                    <td style={{fontSize:'12px',color:'var(--nv-text-muted)'}}>
                      {s.file_count} file{s.file_count !== 1 ? 's' : ''}
                      <span style={{color:'var(--nv-text-dim)',fontSize:'11px',marginLeft:'6px'}}>
                        ({s.files.slice(0, 3).join(", ")}{s.files.length > 3 ? `, +${s.files.length - 3} more` : ""})
                      </span>
                    </td>
                    <td style={{fontSize:'12px',color:'var(--nv-text-muted)',whiteSpace:'nowrap'}}>{formatBytes(s.total_bytes)}</td>
                    <td className="mono" style={{fontSize:'11px',color:'var(--nv-text-dim)',maxWidth:'250px',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}} title={s.path}>{s.path}</td>
                    <td>
                      <div style={{display:'flex',gap:'6px'}}>
                        <button className="btn btn-primary btn-sm" onClick={() => {
                          setUploadResult({ session_id: s.session_id, file_count: s.file_count, files: s.files, total_bytes: s.total_bytes });
                          setTriggered(null);
                          setError("");
                        }} title="Use this session for a new ingestion run">
                          <IconPlay /> Use
                        </button>
                        <button className="btn btn-sm" onClick={() => handleDeleteSession(s.session_id)}
                          style={{background:'rgba(255,80,80,0.1)',color:'#ff5050',border:'1px solid rgba(255,80,80,0.2)'}}
                          title="Delete session"><IconTrash /></button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}
