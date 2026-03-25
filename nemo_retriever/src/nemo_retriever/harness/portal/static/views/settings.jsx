/* ===== Settings View ===== */
function SettingsView() {
  const [gitInfo, setGitInfo] = useState(null);
  const [gitLoading, setGitLoading] = useState(true);
  const [deployBranch, setDeployBranch] = useState("main");
  const [deployRemote, setDeployRemote] = useState("origin");
  const [deploying, setDeploying] = useState(false);
  const [deployResult, setDeployResult] = useState(null);
  const [deployError, setDeployError] = useState("");
  const [showConfirm, setShowConfirm] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);
  const [portalSettings, setPortalSettings] = useState({});
  const [runCodeRefInput, setRunCodeRefInput] = useState("");
  const [savingRunCodeRef, setSavingRunCodeRef] = useState(false);
  const [runCodeRefSaved, setRunCodeRefSaved] = useState(false);

  async function fetchPortalSettings() {
    try {
      const res = await fetch("/api/portal-settings");
      const data = await res.json();
      setPortalSettings(data);
      setRunCodeRefInput(data.run_code_ref || "upstream/main");
    } catch (err) {
      console.error("Failed to fetch portal settings:", err);
    }
  }

  async function saveRunCodeRef() {
    setSavingRunCodeRef(true);
    setRunCodeRefSaved(false);
    try {
      const res = await fetch("/api/portal-settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_code_ref: runCodeRefInput.trim() }),
      });
      if (!res.ok) throw new Error("Failed to save");
      const data = await res.json();
      setPortalSettings(data);
      setRunCodeRefSaved(true);
      setTimeout(() => setRunCodeRefSaved(false), 3000);
    } catch (err) {
      console.error("Failed to save run code ref:", err);
    } finally {
      setSavingRunCodeRef(false);
    }
  }

  async function fetchGitInfo() {
    setGitLoading(true);
    try {
      const res = await fetch("/api/settings/git-info");
      const data = await res.json();
      setGitInfo(data);
      if (data.available && data.remotes?.length) {
        const nvidiaR = data.remotes.find(r => r.url && (r.url.includes("NVIDIA/") || r.url.includes("nvidia/")));
        setDeployRemote(nvidiaR ? nvidiaR.name : data.remotes[0].name);
      }
      if (data.available && data.current_branch) {
        setDeployBranch(data.current_branch);
      }
    } catch (err) {
      setGitInfo({ available: false, error: err.message });
    } finally {
      setGitLoading(false);
    }
  }

  useEffect(() => { fetchGitInfo(); fetchPortalSettings(); }, []);

  async function handleDeploy() {
    setShowConfirm(false);
    setDeploying(true);
    setDeployResult(null);
    setDeployError("");
    try {
      const res = await fetch("/api/settings/deploy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ branch: deployBranch, remote: deployRemote }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setDeployResult(data);
      setReconnecting(true);
      pollForRestart();
    } catch (err) {
      setDeployError(err.message);
    } finally {
      setDeploying(false);
    }
  }

  function pollForRestart() {
    let attempts = 0;
    const maxAttempts = 30;
    const interval = setInterval(async () => {
      attempts++;
      try {
        const res = await fetch("/api/version", { signal: AbortSignal.timeout(3000) });
        if (res.ok) {
          clearInterval(interval);
          setReconnecting(false);
          fetchGitInfo();
        }
      } catch {
        if (attempts >= maxAttempts) {
          clearInterval(interval);
          setReconnecting(false);
          setDeployError("Portal did not come back online within 60 seconds. Check the server manually.");
        }
      }
    }, 2000);
  }

  const labelStyle = {display:'block',fontSize:'11px',fontWeight:600,color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.04em',marginBottom:'6px'};

  if (gitLoading) {
    return (
      <div className="card" style={{padding:'60px',textAlign:'center'}}>
        <span className="spinner spinner-lg" style={{display:'block',margin:'0 auto 16px'}}></span>
        <div style={{color:'var(--nv-text-muted)',fontSize:'14px'}}>Loading git information…</div>
      </div>
    );
  }

  if (!gitInfo || !gitInfo.available) {
    return (
      <div className="card" style={{padding:'40px',textAlign:'center'}}>
        <div style={{fontSize:'32px',marginBottom:'12px',opacity:0.3}}>&#x26A0;</div>
        <div style={{fontSize:'15px',fontWeight:600,color:'#ff5050',marginBottom:'8px'}}>Git Not Available</div>
        <div style={{fontSize:'13px',color:'var(--nv-text-muted)'}}>{gitInfo?.error || "Unable to detect a git repository."}</div>
      </div>
    );
  }

  return (
    <>
      {/* Reconnecting overlay */}
      {reconnecting && (
        <div style={{
          position:'fixed',top:0,left:0,right:0,bottom:0,zIndex:9999,
          background:'rgba(0,0,0,0.85)',display:'flex',alignItems:'center',justifyContent:'center',flexDirection:'column',gap:'20px',
        }}>
          <span className="spinner spinner-lg"></span>
          <div style={{fontSize:'18px',fontWeight:600,color:'#fff'}}>Portal is restarting…</div>
          <div style={{fontSize:'13px',color:'var(--nv-text-muted)',maxWidth:'400px',textAlign:'center',lineHeight:'1.6'}}>
            The portal is pulling the latest code and restarting. This page will automatically reconnect when the server is back online.
          </div>
        </div>
      )}

      {/* Current State */}
      <div className="card" style={{padding:'24px',marginBottom:'20px'}}>
        <div className="section-title" style={{marginBottom:'16px'}}>Current Portal State</div>
        <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill, minmax(200px, 1fr))',gap:'16px'}}>
          <div>
            <div style={labelStyle}>Branch</div>
            <div style={{fontSize:'14px',color:'#fff',fontWeight:600}}>{gitInfo.current_branch}</div>
          </div>
          <div>
            <div style={labelStyle}>Commit</div>
            <div className="mono" style={{fontSize:'13px',color:'var(--nv-green)'}}>{gitInfo.current_short_sha}</div>
            <div className="mono" style={{fontSize:'11px',color:'var(--nv-text-dim)',marginTop:'2px',wordBreak:'break-all'}}>{gitInfo.current_sha}</div>
          </div>
          <div>
            <div style={labelStyle}>Working Directory</div>
            <div style={{fontSize:'13px',color:gitInfo.is_dirty ? '#ffb84d' : 'var(--nv-green)',fontWeight:500}}>
              {gitInfo.is_dirty ? "Uncommitted changes" : "Clean"}
            </div>
          </div>
          <div>
            <div style={labelStyle}>Repository</div>
            <div className="mono" style={{fontSize:'11px',color:'var(--nv-text-muted)',wordBreak:'break-all'}}>{gitInfo.repo_root}</div>
          </div>
        </div>
      </div>

      {/* Runner Execution Branch */}
      <div className="card" style={{padding:'24px',marginBottom:'20px'}}>
        <div className="section-title" style={{marginBottom:'6px'}}>Runner Execution Branch</div>
        <div style={{fontSize:'12px',color:'var(--nv-text-dim)',lineHeight:'1.6',marginBottom:'16px'}}>
          The git ref that runners will <code className="mono" style={{background:'rgba(255,255,255,0.06)',padding:'1px 5px',borderRadius:'4px'}}>git checkout</code> before executing each job.
          This separates the runner infrastructure code (synced via Deploy) from the harness pipeline code used for actual runs.
          Runners receive this value on every heartbeat.
        </div>
        {portalSettings._nvidia_remote_name && (
          <div style={{marginBottom:'14px',padding:'10px 14px',borderRadius:'8px',background:'rgba(118,185,0,0.06)',border:'1px solid rgba(118,185,0,0.15)'}}>
            <div style={{fontSize:'12px',color:'var(--nv-green)',fontWeight:600,marginBottom:'4px'}}>Detected NVIDIA Remote</div>
            <div style={{fontSize:'12px',color:'var(--nv-text-muted)'}}>
              <span className="mono" style={{color:'#fff',fontWeight:600}}>{portalSettings._nvidia_remote_name}</span>
              {portalSettings._nvidia_remote_url && (
                <span style={{marginLeft:'8px',color:'var(--nv-text-dim)'}}>→ {portalSettings._nvidia_remote_url}</span>
              )}
            </div>
          </div>
        )}
        <div style={{display:'flex',gap:'10px',alignItems:'flex-end'}}>
          <div style={{flex:1}}>
            <label style={labelStyle}>Git Ref (remote/branch)</label>
            <input className="input" style={{width:'100%'}} value={runCodeRefInput}
              onChange={e => setRunCodeRefInput(e.target.value)}
              placeholder={portalSettings._nvidia_remote_name ? `e.g. ${portalSettings._nvidia_remote_name}/main` : "e.g. nvidia/main"} />
          </div>
          <button className="btn btn-primary" onClick={saveRunCodeRef}
            disabled={savingRunCodeRef || !runCodeRefInput.trim() || runCodeRefInput.trim() === portalSettings.run_code_ref}
            style={{whiteSpace:'nowrap'}}>
            {savingRunCodeRef ? <><span className="spinner" style={{marginRight:'6px'}}></span>Saving…</> : "Save"}
          </button>
        </div>
        {runCodeRefSaved && (
          <div style={{marginTop:'10px',fontSize:'12px',color:'var(--nv-green)',display:'flex',alignItems:'center',gap:'6px'}}>
            <IconCheck /> Saved. Runners will pick up the new ref on their next heartbeat.
          </div>
        )}
        {(() => {
          const nvidiaRemote = portalSettings._nvidia_remote_name;
          const nvBranches = (gitInfo?.remote_branches || []).filter(b => nvidiaRemote ? b.startsWith(nvidiaRemote + "/") : true);
          return nvBranches.length > 0 && (
            <div style={{marginTop:'12px'}}>
              <div style={{...labelStyle,marginBottom:'8px'}}>
                Quick Select{nvidiaRemote ? ` (${nvidiaRemote} branches)` : ""}
              </div>
              <div style={{display:'flex',gap:'6px',flexWrap:'wrap'}}>
                {nvBranches.slice(0, 15).map(b => (
                  <button key={b} className="btn btn-sm"
                    onClick={() => setRunCodeRefInput(b)}
                    style={{
                      fontSize:'10px',padding:'2px 8px',
                      background: b === runCodeRefInput ? 'rgba(118,185,0,0.15)' : 'transparent',
                      color: b === runCodeRefInput ? 'var(--nv-green)' : 'var(--nv-text-dim)',
                      border: `1px solid ${b === runCodeRefInput ? 'rgba(118,185,0,0.3)' : 'var(--nv-border)'}`,
                    }}>
                    {b}
                  </button>
                ))}
              </div>
            </div>
          );
        })()}
      </div>

      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'20px',marginBottom:'20px'}}>
        {/* Deploy Card */}
        <div className="card" style={{padding:'24px'}}>
          <div className="section-title" style={{marginBottom:'16px'}}>Deploy Latest Code</div>
          <div style={{display:'flex',flexDirection:'column',gap:'14px'}}>
            <div>
              <label style={labelStyle}>Remote</label>
              <select className="select" style={{width:'100%'}} value={deployRemote} onChange={e => setDeployRemote(e.target.value)}>
                {(gitInfo.remotes || []).map(r => (
                  <option key={r.name} value={r.name}>{r.name} ({r.url})</option>
                ))}
              </select>
            </div>
            <div>
              <label style={labelStyle}>Branch</label>
              <div style={{display:'flex',gap:'8px'}}>
                <input className="input" style={{flex:1}} value={deployBranch}
                  onChange={e => setDeployBranch(e.target.value)}
                  placeholder="e.g. main" />
              </div>
              {gitInfo.remote_branches?.length > 0 && (
                <div style={{marginTop:'8px',display:'flex',gap:'6px',flexWrap:'wrap'}}>
                  {gitInfo.remote_branches.slice(0, 10).map(b => {
                    const short = b.replace(/^[^/]+\//, "");
                    return (
                      <button key={b} className="btn btn-sm"
                        onClick={() => setDeployBranch(short)}
                        style={{
                          fontSize:'10px',padding:'2px 8px',
                          background: short === deployBranch ? 'rgba(118,185,0,0.15)' : 'transparent',
                          color: short === deployBranch ? 'var(--nv-green)' : 'var(--nv-text-dim)',
                          border: `1px solid ${short === deployBranch ? 'rgba(118,185,0,0.3)' : 'var(--nv-border)'}`,
                        }}>
                        {short}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
            <button className="btn btn-primary" onClick={() => setShowConfirm(true)}
              disabled={deploying || !deployBranch.trim()}
              style={{width:'100%',justifyContent:'center',marginTop:'4px'}}>
              {deploying ? <><span className="spinner" style={{marginRight:'8px'}}></span>Deploying…</> : <><IconDownload /> Deploy & Restart</>}
            </button>
            <div style={{fontSize:'11px',color:'var(--nv-text-dim)',lineHeight:'1.5'}}>
              This will pull the latest code from <span className="mono" style={{color:'var(--nv-text-muted)'}}>{deployRemote}/{deployBranch}</span>, then restart the portal web server.
              The page will automatically reconnect.
            </div>
          </div>
        </div>

        {/* Recent Commits Card */}
        <div className="card" style={{padding:'24px'}}>
          <div className="section-title" style={{marginBottom:'16px'}}>Recent Commits</div>
          <div style={{display:'flex',flexDirection:'column',gap:'0'}}>
            {(gitInfo.recent_commits || []).map((c, i) => (
              <div key={c.sha} style={{
                display:'flex',gap:'10px',alignItems:'flex-start',padding:'8px 0',
                borderBottom: i < gitInfo.recent_commits.length - 1 ? '1px solid var(--nv-border)' : 'none',
              }}>
                <span className="mono" style={{
                  fontSize:'11px',fontWeight:600,whiteSpace:'nowrap',
                  color: i === 0 ? 'var(--nv-green)' : 'var(--nv-text-dim)',
                }}>{c.short_sha}</span>
                <div style={{flex:1,minWidth:0}}>
                  <div style={{fontSize:'12px',color:'#fff',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{c.message}</div>
                  <div style={{fontSize:'10px',color:'var(--nv-text-dim)',marginTop:'2px'}}>{c.date}</div>
                </div>
              </div>
            ))}
            {(!gitInfo.recent_commits || gitInfo.recent_commits.length === 0) && (
              <div style={{padding:'20px',textAlign:'center',color:'var(--nv-text-dim)',fontSize:'13px'}}>No commit history available</div>
            )}
          </div>
        </div>
      </div>

      {/* Remotes */}
      <div className="card" style={{padding:'24px',marginBottom:'20px'}}>
        <div className="section-title" style={{marginBottom:'12px'}}>Remotes</div>
        <div style={{display:'flex',flexDirection:'column',gap:'8px'}}>
          {(gitInfo.remotes || []).map(r => (
            <div key={r.name} style={{display:'flex',alignItems:'center',gap:'12px',padding:'6px 0',borderBottom:'1px solid var(--nv-border)'}}>
              <span style={{fontSize:'13px',fontWeight:600,color:'#fff',minWidth:'80px'}}>{r.name}</span>
              <span className="mono" style={{fontSize:'12px',color:'var(--nv-text-muted)',wordBreak:'break-all'}}>{r.url}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Deploy Result */}
      {deployResult && (
        <div className="card" style={{padding:'20px',marginBottom:'20px',background:'rgba(118,185,0,0.05)',border:'1px solid rgba(118,185,0,0.2)'}}>
          <div style={{display:'flex',alignItems:'center',gap:'10px',marginBottom:'12px'}}>
            <IconCheck />
            <span style={{fontSize:'14px',fontWeight:600,color:'var(--nv-green)'}}>{deployResult.message}</span>
          </div>
          {deployResult.log && deployResult.log.length > 0 && (
            <pre className="mono" style={{
              fontSize:'11px',padding:'12px',borderRadius:'8px',
              background:'var(--nv-bg)',border:'1px solid var(--nv-border)',
              color:'var(--nv-text-muted)',maxHeight:'200px',overflow:'auto',
              whiteSpace:'pre-wrap',wordBreak:'break-all',lineHeight:'1.6',
            }}>{deployResult.log.join("\n")}</pre>
          )}
        </div>
      )}

      {/* Deploy Error */}
      {deployError && (
        <div style={{marginBottom:'16px',padding:'14px 18px',borderRadius:'8px',background:'rgba(255,50,50,0.08)',border:'1px solid rgba(255,50,50,0.2)'}}>
          <div style={{fontSize:'13px',fontWeight:600,color:'#ff5050',marginBottom:'6px'}}>Deploy Failed</div>
          <pre className="mono" style={{fontSize:'12px',color:'#ff5050',whiteSpace:'pre-wrap',wordBreak:'break-all',margin:0}}>{deployError}</pre>
        </div>
      )}

      {/* Confirmation Modal */}
      {showConfirm && (
        <div className="modal-overlay" onClick={() => setShowConfirm(false)}>
          <div className="modal-content" style={{maxWidth:'480px'}} onClick={e => e.stopPropagation()}>
            <div className="modal-head">
              <h2 style={{fontSize:'16px',fontWeight:700,color:'#fff'}}>Confirm Deploy</h2>
              <button className="btn btn-ghost btn-icon" onClick={() => setShowConfirm(false)} style={{borderRadius:'50%'}}><IconX /></button>
            </div>
            <div style={{padding:'24px'}}>
              <div style={{fontSize:'14px',color:'var(--nv-text-muted)',lineHeight:'1.7',marginBottom:'16px'}}>
                This will:
              </div>
              <ol style={{fontSize:'13px',color:'#fff',lineHeight:'2',paddingLeft:'20px',marginBottom:'20px'}}>
                <li>Stash any uncommitted local changes</li>
                <li>Fetch the latest code from <span className="mono" style={{color:'var(--nv-green)'}}>{deployRemote}</span></li>
                <li>Checkout branch <span className="mono" style={{color:'var(--nv-green)'}}>{deployBranch}</span></li>
                <li>Restart the portal web server</li>
              </ol>
              {gitInfo.is_dirty && (
                <div style={{padding:'10px 14px',borderRadius:'8px',background:'rgba(255,184,77,0.08)',border:'1px solid rgba(255,184,77,0.25)',color:'#ffb84d',fontSize:'12px',marginBottom:'16px'}}>
                  <strong>Warning:</strong> You have uncommitted changes. They will be stashed before the deploy.
                </div>
              )}
              <div style={{fontSize:'12px',color:'var(--nv-text-dim)',marginBottom:'4px'}}>
                The portal will be briefly unavailable during restart. This page will reconnect automatically.
              </div>
            </div>
            <div className="modal-foot">
              <button className="btn btn-secondary" onClick={() => setShowConfirm(false)}>Cancel</button>
              <button className="btn btn-primary" onClick={handleDeploy} style={{flex:1,justifyContent:'center'}}>
                <IconDownload /> Deploy {deployBranch}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
