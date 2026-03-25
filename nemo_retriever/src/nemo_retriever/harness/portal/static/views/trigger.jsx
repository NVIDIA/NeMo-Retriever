/* ===== Trigger Modal ===== */
function TriggerModal({ onClose, onTriggered }) {
  const [datasets, setDatasets] = useState([]);
  const [presets, setPresets] = useState([]);
  const [runners, setRunners] = useState([]);
  const [dataset, setDataset] = useState("");
  const [preset, setPreset] = useState("");
  const [runnerId, setRunnerId] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const [showGit, setShowGit] = useState(false);
  const [gitMode, setGitMode] = useState("default");
  const [gitRef, setGitRef] = useState("");
  const [gitCommit, setGitCommit] = useState("");
  const [remoteBranches, setRemoteBranches] = useState([]);
  const [defaultRef, setDefaultRef] = useState("");

  useEffect(() => {
    fetch("/api/config").then(r=>r.json()).then(cfg => {
      setDatasets(cfg.datasets || []);
      setPresets(cfg.presets || []);
      if (cfg.datasets?.length) setDataset(cfg.datasets[0]);
      if (cfg.presets?.length) setPreset(cfg.presets[0]);
    });
    fetch("/api/runners").then(r=>r.json()).then(setRunners).catch(()=>{});
    fetch("/api/portal-settings").then(r=>r.json()).then(s => {
      setDefaultRef(s.run_code_ref || "");
    }).catch(()=>{});
    fetch("/api/settings/git-info").then(r=>r.json()).then(info => {
      if (info.available) setRemoteBranches(info.remote_branches || []);
    }).catch(()=>{});
  }, []);

  const onlineRunners = runners.filter(r => r.status === "online" || r.status === "paused");

  async function handleSubmit(e) {
    e.preventDefault();
    if (!dataset) return;
    setSubmitting(true);
    try {
      const payload = {
        dataset,
        preset: preset || null,
        runner_id: runnerId ? parseInt(runnerId, 10) : null,
      };
      if (gitMode === "branch" && gitRef.trim()) {
        payload.git_ref = gitRef.trim();
      } else if (gitMode === "commit" && gitRef.trim()) {
        payload.git_ref = gitRef.trim();
        if (gitCommit.trim()) payload.git_commit = gitCommit.trim();
      }
      const res = await fetch("/api/runs/trigger", {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      onTriggered(data);
      onClose();
    } catch (err) {
      alert("Failed to trigger run: " + err.message);
    } finally {
      setSubmitting(false);
    }
  }

  const labelStyle = {display:'block',fontSize:'12px',fontWeight:500,color:'var(--nv-text-muted)',marginBottom:'6px',textTransform:'uppercase',letterSpacing:'0.04em'};
  const hintStyle = {fontSize:'11px',color:'var(--nv-text-dim)',marginTop:'4px',lineHeight:'1.5'};

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{maxWidth:'520px'}} onClick={e=>e.stopPropagation()}>
        <div className="modal-head">
          <h2 style={{fontSize:'16px',fontWeight:700,color:'#fff'}}>Trigger New Run</h2>
          <button className="btn btn-ghost btn-icon" onClick={onClose} style={{borderRadius:'50%'}}><IconX /></button>
        </div>
        <form onSubmit={handleSubmit}>
          <div style={{padding:'24px',display:'flex',flexDirection:'column',gap:'16px'}}>
            <div>
              <label style={labelStyle}>Dataset</label>
              <select value={dataset} onChange={e=>setDataset(e.target.value)} className="select" style={{width:'100%'}}>
                {datasets.map(d=><option key={d} value={d}>{d}</option>)}
              </select>
            </div>
            <div>
              <label style={labelStyle}>Preset</label>
              <select value={preset} onChange={e=>setPreset(e.target.value)} className="select" style={{width:'100%'}}>
                {presets.map(p=><option key={p} value={p}>{p}</option>)}
              </select>
            </div>
            <div>
              <label style={labelStyle}>Runner</label>
              <select value={runnerId} onChange={e=>setRunnerId(e.target.value)} className="select" style={{width:'100%'}}>
                <option value="">Any available runner</option>
                {onlineRunners.map(r=><option key={r.id} value={r.id}>{r.name} ({r.hostname || 'unknown'}) — {r.gpu_type || 'no GPU'} x{r.gpu_count||0}{r.status==='paused'?' [PAUSED]':''}</option>)}
              </select>
              <div style={hintStyle}>
                {onlineRunners.length === 0
                  ? "No runners online. The job will wait until a runner becomes available."
                  : `${onlineRunners.length} runner${onlineRunners.length!==1?'s':''} online`}
              </div>
            </div>

            {/* Git Override Section */}
            <div style={{borderTop:'1px solid var(--nv-border)',paddingTop:'16px'}}>
              <button type="button" onClick={()=>setShowGit(!showGit)}
                style={{
                  background:'none',border:'none',padding:0,cursor:'pointer',
                  display:'flex',alignItems:'center',gap:'8px',width:'100%',
                }}>
                <span style={{fontSize:'12px',fontWeight:500,color:'var(--nv-text-muted)',textTransform:'uppercase',letterSpacing:'0.04em'}}>
                  Git Checkout Override
                </span>
                <span style={{fontSize:'10px',color:'var(--nv-text-dim)',transform:showGit?'rotate(180deg)':'rotate(0)',transition:'transform 0.15s'}}>&#9660;</span>
                {gitMode !== "default" && (
                  <span style={{fontSize:'10px',padding:'1px 6px',borderRadius:'4px',background:'rgba(118,185,0,0.12)',color:'var(--nv-green)',fontWeight:600,marginLeft:'auto'}}>
                    Override Active
                  </span>
                )}
              </button>

              {showGit && (
                <div style={{marginTop:'14px',display:'flex',flexDirection:'column',gap:'14px'}}>
                  <div style={{display:'flex',gap:'6px'}}>
                    {[
                      {id:'default', label:'Use Settings Default'},
                      {id:'branch', label:'Latest from Branch'},
                      {id:'commit', label:'Specific Commit'},
                    ].map(opt => (
                      <button key={opt.id} type="button" onClick={()=>setGitMode(opt.id)}
                        className="btn btn-sm"
                        style={{
                          fontSize:'11px',padding:'4px 10px',flex:1,justifyContent:'center',
                          background: gitMode===opt.id ? 'rgba(118,185,0,0.12)' : 'transparent',
                          color: gitMode===opt.id ? 'var(--nv-green)' : 'var(--nv-text-dim)',
                          border: `1px solid ${gitMode===opt.id ? 'rgba(118,185,0,0.3)' : 'var(--nv-border)'}`,
                        }}>
                        {opt.label}
                      </button>
                    ))}
                  </div>

                  {gitMode === "default" && (
                    <div style={hintStyle}>
                      Uses the Runner Execution Branch from Settings{defaultRef ? `: ${defaultRef}` : '.'}
                    </div>
                  )}

                  {(gitMode === "branch" || gitMode === "commit") && (
                    <>
                      <div>
                        <label style={labelStyle}>Remote / Branch</label>
                        <input className="input" style={{width:'100%'}} value={gitRef}
                          onChange={e=>setGitRef(e.target.value)}
                          placeholder="e.g. nvidia/main or origin/feat/my-branch" />
                        {remoteBranches.length > 0 && (
                          <div style={{marginTop:'8px',display:'flex',gap:'4px',flexWrap:'wrap',maxHeight:'80px',overflow:'auto'}}>
                            {remoteBranches.slice(0, 20).map(b => (
                              <button key={b} type="button" className="btn btn-sm"
                                onClick={()=>setGitRef(b)}
                                style={{
                                  fontSize:'10px',padding:'1px 6px',
                                  background: b===gitRef ? 'rgba(118,185,0,0.12)' : 'transparent',
                                  color: b===gitRef ? 'var(--nv-green)' : 'var(--nv-text-dim)',
                                  border: `1px solid ${b===gitRef ? 'rgba(118,185,0,0.3)' : 'var(--nv-border)'}`,
                                }}>
                                {b}
                              </button>
                            ))}
                          </div>
                        )}
                        <div style={hintStyle}>
                          {gitMode === "branch"
                            ? "The runner will fetch and checkout the latest commit from this branch."
                            : "The branch that contains the commit below."}
                        </div>
                      </div>
                      {gitMode === "commit" && (
                        <div>
                          <label style={labelStyle}>Commit SHA</label>
                          <input className="input mono" style={{width:'100%',fontSize:'12px'}} value={gitCommit}
                            onChange={e=>setGitCommit(e.target.value)}
                            placeholder="e.g. a1b2c3d4e5f6 or full 40-char SHA" />
                          <div style={hintStyle}>
                            The runner will checkout this exact commit. Leave empty to use the latest from the branch above.
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
          <div className="modal-foot">
            <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button type="submit" disabled={submitting||!dataset} className="btn btn-primary" style={{flex:1,justifyContent:'center'}}>
              {submitting ? <><span className="spinner" style={{marginRight:'8px'}}></span>Triggering…</> : "Start Run"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
