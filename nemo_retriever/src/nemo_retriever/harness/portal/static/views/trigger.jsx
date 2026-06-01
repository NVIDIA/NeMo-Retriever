/* ===== Trigger Modal ===== */
function TriggerModal({ onClose, onTriggered }) {
  const [datasets, setDatasets] = useState([]);
  const [presets, setPresets] = useState([]);
  const [graphs, setGraphs] = useState([]);
  const [runners, setRunners] = useState([]);
  const [clusters, setClusters] = useState([]);
  const [dataset, setDataset] = useState("");
  const [preset, setPreset] = useState("");
  const [pipelineSource, setPipelineSource] = useState("preset");
  const [graphId, setGraphId] = useState("");
  const [runnerId, setRunnerId] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const [runMode, setRunMode] = useState("batch");
  const [executionTarget, setExecutionTarget] = useState("local");
  const [clusterId, setClusterId] = useState("");

  const [showGit, setShowGit] = useState(false);
  const [gitMode, setGitMode] = useState("default");
  const [gitRef, setGitRef] = useState("");
  const [gitCommit, setGitCommit] = useState("");
  const [remoteBranches, setRemoteBranches] = useState([]);
  const [defaultRef, setDefaultRef] = useState("");
  const [nsysProfile, setNsysProfile] = useState(false);

  const [serviceUrl, setServiceUrl] = useState("");
  const [serviceMaxConcurrency, setServiceMaxConcurrency] = useState(8);
  const [defaultServiceUrl, setDefaultServiceUrl] = useState("");

  useEffect(() => {
    fetch("/api/config").then(r=>r.json()).then(cfg => {
      setDatasets(cfg.datasets || []);
      setPresets(cfg.presets || []);
      if (cfg.datasets?.length) setDataset(cfg.datasets[0]);
      if (cfg.presets?.length) setPreset(cfg.presets[0]);
    });
    fetch("/api/runners").then(r=>r.json()).then(setRunners).catch(()=>{});
    fetch("/api/clusters").then(r=>r.json()).then(list => {
      setClusters(Array.isArray(list) ? list : []);
    }).catch(()=>{});
    fetch("/api/graphs").then(r=>r.json()).then(list => {
      const arr = Array.isArray(list) ? list : [];
      setGraphs(arr);
      if (arr.length > 0) setGraphId(String(arr[0].id));
    }).catch(()=>{});
    fetch("/api/portal-settings").then(r=>r.json()).then(s => {
      setDefaultRef(s.run_code_ref || "");
      if (s.service_url) {
        setDefaultServiceUrl(s.service_url);
        setServiceUrl(s.service_url);
      }
    }).catch(()=>{});
    fetch("/api/settings/git-info").then(r=>r.json()).then(info => {
      if (info.available) setRemoteBranches(info.remote_branches || []);
    }).catch(()=>{});
  }, []);

  useEffect(() => {
    if (executionTarget === "cluster" && clusterId) {
      const cl = clusters.find(c => String(c.id) === clusterId);
      if (cl && cl.default_run_mode) setRunMode(cl.default_run_mode);
    }
  }, [clusterId, executionTarget, clusters]);

  const onlineRunners = runners.filter(r => r.status === "online" || r.status === "paused");
  const onlineClusters = clusters.filter(c => c.status !== "error");

  async function handleSubmit(e) {
    e.preventDefault();
    if (!dataset) return;
    if (pipelineSource === "graph" && !graphId) return;
    setSubmitting(true);
    try {
      const payload = {
        dataset,
        preset: pipelineSource === "preset" ? (preset || null) : null,
        run_mode: runMode,
        execution_target: executionTarget,
        cluster_id: executionTarget === "cluster" ? parseInt(clusterId, 10) : null,
        runner_id: executionTarget === "local" && runnerId ? parseInt(runnerId, 10) : null,
        nsys_profile: nsysProfile,
      };
      if (pipelineSource === "graph") {
        payload.graph_id = parseInt(graphId, 10);
      }
      if (runMode === "service") {
        payload.service_url = serviceUrl.trim();
        payload.service_max_concurrency = serviceMaxConcurrency;
      }
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
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
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
  const modeBtn = (current, id, label) => ({
    fontSize:'11px',padding:'5px 12px',flex:1,justifyContent:'center',textAlign:'center',
    background: current===id ? 'rgba(118,185,0,0.12)' : 'transparent',
    color: current===id ? 'var(--nv-green)' : 'var(--nv-text-dim)',
    border: `1px solid ${current===id ? 'rgba(118,185,0,0.3)' : 'var(--nv-border)'}`,
    cursor:'pointer', borderRadius:'6px', fontWeight: current===id ? 600 : 400,
  });

  const selectedGraph = graphs.find(g => String(g.id) === graphId);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{maxWidth:'560px',maxHeight:'85vh',overflow:'auto'}} onClick={e=>e.stopPropagation()}>
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

            {/* Pipeline Source Toggle */}
            <div>
              <label style={labelStyle}>Pipeline Source</label>
              <div style={{display:'flex',gap:'6px'}}>
                <button type="button" onClick={()=>setPipelineSource("preset")} className="btn btn-sm" style={modeBtn(pipelineSource,"preset")}>
                  Preset
                </button>
                <button type="button" onClick={()=>setPipelineSource("graph")} className="btn btn-sm"
                  style={modeBtn(pipelineSource,"graph")} disabled={graphs.length===0}>
                  Graph Pipeline
                </button>
              </div>
              {graphs.length === 0 && pipelineSource === "preset" && (
                <div style={hintStyle}>No saved graphs available. Create one in the Designer view to enable graph pipeline runs.</div>
              )}
            </div>

            {pipelineSource === "preset" && (
              <div>
                <label style={labelStyle}>Preset</label>
                <select value={preset} onChange={e=>setPreset(e.target.value)} className="select" style={{width:'100%'}}>
                  {presets.map(p=><option key={p} value={p}>{p}</option>)}
                </select>
              </div>
            )}

            {pipelineSource === "graph" && (
              <div>
                <label style={labelStyle}>Graph</label>
                <select value={graphId} onChange={e=>setGraphId(e.target.value)} className="select" style={{width:'100%'}}>
                  {graphs.map(g=><option key={g.id} value={String(g.id)}>{g.name || `Graph #${g.id}`}</option>)}
                </select>
                {selectedGraph && (
                  <div style={{...hintStyle,display:'flex',gap:'12px',alignItems:'center',marginTop:'6px'}}>
                    <span style={{padding:'1px 6px',fontSize:'10px',borderRadius:'4px',background:'rgba(118,185,0,0.12)',color:'var(--nv-green)',fontWeight:600}}>
                      GRAPH
                    </span>
                    <span>The graph pipeline replaces the batch_pipeline preset. Recall/BEIR evaluation runs against the dataset's query CSV after graph execution.</span>
                  </div>
                )}
              </div>
            )}

            {/* Run Mode Selector */}
            <div style={{borderTop:'1px solid var(--nv-border)',paddingTop:'16px'}}>
              <label style={labelStyle}>Run Mode</label>
              <div style={{display:'flex',gap:'6px'}}>
                <button type="button" onClick={()=>setRunMode("batch")} className="btn btn-sm" style={modeBtn(runMode,"batch")}>
                  Batch (Ray)
                </button>
                <button type="button" onClick={()=>setRunMode("inprocess")} className="btn btn-sm" style={modeBtn(runMode,"inprocess")}>
                  In-Process
                </button>
                <button type="button" onClick={()=>setRunMode("service")} className="btn btn-sm" style={modeBtn(runMode,"service")}>
                  Service
                </button>
              </div>
              <div style={hintStyle}>
                {runMode === "batch" && "Distributed execution using Ray Data. Requires a Ray cluster or local Ray."}
                {runMode === "inprocess" && "Single-process execution using pandas. No Ray cluster needed."}
                {runMode === "service" && "Uploads documents to a running retriever service for ingestion."}
              </div>
            </div>

            {runMode === "service" && (
              <div style={{display:'flex',flexDirection:'column',gap:'12px'}}>
                <div>
                  <label style={labelStyle}>Service URL</label>
                  <input className="input" style={{width:'100%'}} value={serviceUrl}
                    onChange={e=>setServiceUrl(e.target.value)}
                    placeholder="http://localhost:7670" />
                  {defaultServiceUrl && serviceUrl === defaultServiceUrl && (
                    <div style={hintStyle}>Using default from portal settings.</div>
                  )}
                </div>
                <div>
                  <label style={labelStyle}>Max Concurrency</label>
                  <input className="input" type="number" min="1" max="64" style={{width:'120px'}}
                    value={serviceMaxConcurrency}
                    onChange={e=>setServiceMaxConcurrency(parseInt(e.target.value,10)||8)} />
                  <div style={hintStyle}>Maximum concurrent page uploads to the service.</div>
                </div>
              </div>
            )}

            {/* Execution Target */}
            <div style={{borderTop:'1px solid var(--nv-border)',paddingTop:'16px'}}>
              <label style={labelStyle}>Execution Target</label>
              <div style={{display:'flex',gap:'6px'}}>
                <button type="button" onClick={()=>setExecutionTarget("local")} className="btn btn-sm" style={modeBtn(executionTarget,"local")}>
                  Local Runner
                </button>
                <button type="button" onClick={()=>setExecutionTarget("cluster")} className="btn btn-sm"
                  style={modeBtn(executionTarget,"cluster")} disabled={clusters.length===0}>
                  K8s Cluster
                </button>
              </div>
              <div style={hintStyle}>
                {executionTarget === "local" && "Job will be claimed and executed by a registered runner agent."}
                {executionTarget === "cluster" && "Job will be dispatched as a Kubernetes Job on the selected cluster."}
              </div>
            </div>

            {executionTarget === "local" && (
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
            )}

            {executionTarget === "cluster" && (
              <div>
                <label style={labelStyle}>Cluster</label>
                <select value={clusterId} onChange={e=>setClusterId(e.target.value)} className="select" style={{width:'100%'}}>
                  <option value="">Select a cluster...</option>
                  {onlineClusters.map(c=><option key={c.id} value={c.id}>{c.name} ({c.namespace}) — {c.gpu_type || 'no GPU'} x{c.gpu_count||0}</option>)}
                </select>
                {clusters.length === 0 && (
                  <div style={hintStyle}>No clusters registered. Add clusters in the Clusters view.</div>
                )}
              </div>
            )}

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

            {/* Profiling */}
            <div style={{borderTop:'1px solid var(--nv-border)',paddingTop:'16px'}}>
              <label style={{display:'flex',alignItems:'center',gap:'8px',cursor:'pointer',fontSize:'13px',color:'var(--nv-text-muted)'}}>
                <input type="checkbox" checked={nsysProfile} onChange={e => setNsysProfile(e.target.checked)}
                  style={{width:'16px',height:'16px',accentColor:'var(--nv-green)'}} />
                Enable Nsight Systems Profile
              </label>
              <div style={hintStyle}>
                Wraps the job subprocess with <code style={{fontSize:'11px'}}>nsys profile</code>. The <code style={{fontSize:'11px'}}>.nsys-rep</code> file will be included in the artifacts ZIP.
              </div>
            </div>
          </div>
          <div className="modal-foot">
            <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button type="submit" disabled={submitting||!dataset||(pipelineSource==='graph'&&!graphId)||(runMode==='service'&&!serviceUrl.trim())||(executionTarget==='cluster'&&!clusterId)} className="btn btn-primary" style={{flex:1,justifyContent:'center'}}>
              {submitting ? <><span className="spinner" style={{marginRight:'8px'}}></span>Triggering…</> : (
                executionTarget==='cluster' ? 'Run on Cluster' :
                pipelineSource==='graph' ? 'Run Graph Pipeline' :
                runMode==='service' ? 'Run Service Ingest' : 'Start Run'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
