/* ===== Presets View ===== */
const TUNING_FIELDS = [
  { key:"pdf_extract_workers", label:"PDF Extract Workers", type:"int" },
  { key:"pdf_extract_num_cpus", label:"PDF Extract CPUs", type:"float" },
  { key:"pdf_extract_batch_size", label:"PDF Extract Batch", type:"int" },
  { key:"pdf_split_batch_size", label:"PDF Split Batch", type:"int" },
  { key:"page_elements_workers", label:"Page Elements Workers", type:"int" },
  { key:"page_elements_batch_size", label:"Page Elements Batch", type:"int" },
  { key:"page_elements_cpus_per_actor", label:"Page Elements CPUs/Actor", type:"float" },
  { key:"gpu_page_elements", label:"GPU Page Elements", type:"float" },
  { key:"ocr_workers", label:"OCR Workers", type:"int" },
  { key:"ocr_batch_size", label:"OCR Batch", type:"int" },
  { key:"ocr_cpus_per_actor", label:"OCR CPUs/Actor", type:"float" },
  { key:"gpu_ocr", label:"GPU OCR", type:"float" },
  { key:"embed_workers", label:"Embed Workers", type:"int" },
  { key:"embed_batch_size", label:"Embed Batch", type:"int" },
  { key:"embed_cpus_per_actor", label:"Embed CPUs/Actor", type:"float" },
  { key:"gpu_embed", label:"GPU Embed", type:"float" },
];

const TUNING_GROUPS = [
  { label:"PDF Extract", fields: TUNING_FIELDS.filter(f=>f.key.startsWith("pdf_")) },
  { label:"Page Elements", fields: TUNING_FIELDS.filter(f=>f.key.includes("page_elements")) },
  { label:"OCR", fields: TUNING_FIELDS.filter(f=>f.key.includes("ocr")) },
  { label:"Embed", fields: TUNING_FIELDS.filter(f=>f.key.includes("embed")) },
];

function PresetsView({ managedPresets, yamlPresets, loading, onRefresh, presetMatrices, presetMatricesLoading }) {
  const [showForm, setShowForm] = useState(false);
  const [editPreset, setEditPreset] = useState(null);
  const [expandedId, setExpandedId] = useState(null);
  const [showMatrixForm, setShowMatrixForm] = useState(false);
  const [editMatrix, setEditMatrix] = useState(null);
  const [expandedMatrixId, setExpandedMatrixId] = useState(null);
  const [importing, setImporting] = useState(false);
  const allPresets = managedPresets;
  const pg = usePagination(allPresets, 25);
  const matrices = presetMatrices || [];
  const mxPg = usePagination(matrices, 25);

  function handleCreate() { setEditPreset(null); setShowForm(true); }
  function handleEdit(p) { setEditPreset(p); setShowForm(true); }
  async function handleDelete(id, name) {
    if (!confirm(`Delete preset "${name}"? This cannot be undone.`)) return;
    try {
      await fetch(`/api/managed-presets/${id}`, { method: "DELETE" });
      onRefresh();
    } catch {}
  }

  function handleExport() {
    window.location.href = "/api/managed-presets/export.yaml";
  }

  function handleImport() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".yaml,.yml";
    input.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      setImporting(true);
      try {
        const formData = new FormData();
        formData.append("file", file);
        const res = await fetch("/api/managed-presets/import", { method: "POST", body: formData });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          alert("Import failed: " + (data.detail || `HTTP ${res.status}`));
          return;
        }
        const data = await res.json();
        alert(`Import complete: ${data.created} created, ${data.updated} updated`);
        onRefresh();
      } catch (err) {
        alert("Import failed: " + err.message);
      } finally {
        setImporting(false);
      }
    };
    input.click();
  }

  function handleCreateMatrix() { setEditMatrix(null); setShowMatrixForm(true); }
  function handleEditMatrix(m) { setEditMatrix(m); setShowMatrixForm(true); }
  async function handleDeleteMatrix(id, name) {
    if (!confirm(`Delete preset matrix "${name}"? This cannot be undone.`)) return;
    try {
      await fetch(`/api/preset-matrices/${id}`, { method: "DELETE" });
      onRefresh();
    } catch {}
  }
  const [triggerMatrixTarget, setTriggerMatrixTarget] = useState(null);

  function handleTriggerMatrix(m) {
    setTriggerMatrixTarget(m);
  }

  return (
    <>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'20px',flexWrap:'wrap',gap:'12px'}}>
        <div style={{display:'flex',gap:'8px'}}>
          <button className="btn btn-primary" onClick={handleCreate}><IconPlus /> Add Preset</button>
          <button className="btn btn-secondary" onClick={handleCreateMatrix}><IconPlus /> Add Matrix</button>
          <button className="btn btn-secondary" onClick={handleExport} disabled={allPresets.length===0} title="Export all presets to YAML"><IconDownload /> Export YAML</button>
          <button className="btn btn-secondary" onClick={handleImport} disabled={importing} title="Import presets from a YAML file">
            {importing ? <><span className="spinner" style={{marginRight:'6px'}}></span>Importing…</> : <><IconUpload /> Import YAML</>}
          </button>
        </div>
        <button className="btn btn-secondary btn-icon" onClick={onRefresh} title="Refresh"><IconRefresh /></button>
      </div>

      <div className="card">
        <div style={{overflowX:'auto'}}>
          <table className="runs-table">
            <thead>
              <tr>
                <th>Name</th><th>Description</th><th style={{textAlign:'right'}}>Fields</th><th style={{textAlign:'right'}}>Overrides</th><th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan="5" style={{textAlign:'center',padding:'60px',color:'var(--nv-text-muted)'}}>
                  <div className="spinner spinner-lg" style={{margin:'0 auto 12px'}}></div><div>Loading presets…</div>
                </td></tr>
              ) : allPresets.length === 0 ? (
                <tr><td colSpan="5" style={{textAlign:'center',padding:'40px',color:'var(--nv-text-muted)'}}>
                  <div style={{marginBottom:'8px',fontSize:'15px'}}>No presets configured</div>
                  <div style={{fontSize:'12px',color:'var(--nv-text-dim)'}}>Click "Add Preset" to create one, or presets from test_configs.yaml will be imported on portal startup.</div>
                </td></tr>
              ) : pg.pageData.map(p => {
                const cfg = typeof p.config === 'object' ? p.config : {};
                const ovr = typeof p.overrides === 'object' ? p.overrides : {};
                const tuningKeySet_ = new Set(TUNING_FIELDS.map(f=>f.key));
                const tuningCount = Object.keys(cfg).filter(k=>tuningKeySet_.has(k)).length;
                const extraCfgCount = Object.keys(cfg).filter(k=>!tuningKeySet_.has(k)).length;
                const overrideCount = Object.keys(ovr).length;
                const totalOverrides = extraCfgCount + overrideCount;
                const isExpanded = expandedId === p.id;
                return (
                  <React.Fragment key={p.id}>
                    <tr>
                      <td>
                        <span style={{color:'#fff',fontWeight:600,cursor:'pointer'}} onClick={()=>setExpandedId(isExpanded?null:p.id)}>
                          <span style={{display:'inline-block',transform:isExpanded?'rotate(90deg)':'rotate(0deg)',transition:'transform 0.15s',marginRight:'6px',fontSize:'10px'}}>{"\u25B6"}</span>
                          {p.name}
                        </span>
                      </td>
                      <td style={{color:'var(--nv-text-muted)',fontSize:'13px'}}>{p.description || "\u2014"}</td>
                      <td style={{textAlign:'right'}}>{tuningCount > 0 ? <span className="badge badge-na">{tuningCount} param{tuningCount!==1?'s':''}</span> : <span style={{color:'var(--nv-text-dim)',fontSize:'12px'}}>{"\u2014"}</span>}</td>
                      <td style={{textAlign:'right'}}>
                        {totalOverrides > 0 ? (
                          <span className="badge" style={{background:'rgba(118,185,0,0.1)',color:'var(--nv-green)',border:'1px solid rgba(118,185,0,0.25)'}}>
                            {totalOverrides} override{totalOverrides!==1?'s':''}
                          </span>
                        ) : <span style={{color:'var(--nv-text-dim)',fontSize:'12px'}}>{"\u2014"}</span>}
                      </td>
                      <td>
                        <div style={{display:'flex',gap:'6px',flexWrap:'nowrap'}}>
                          <button className="btn btn-secondary btn-sm" onClick={()=>handleEdit(p)} title="Edit"><IconEdit /> Edit</button>
                          <button className="btn btn-sm" onClick={()=>handleDelete(p.id,p.name)} title="Delete" style={{background:'rgba(255,80,80,0.1)',color:'#ff5050',border:'1px solid rgba(255,80,80,0.2)'}}><IconTrash /> Delete</button>
                        </div>
                      </td>
                    </tr>
                    {isExpanded && (() => {
                      const tuningKeys = new Set(TUNING_FIELDS.map(f=>f.key));
                      const extraCfgEntries = Object.entries(cfg).filter(([k])=>!tuningKeys.has(k));
                      const setTuningFields = TUNING_FIELDS.filter(f=>cfg[f.key]!=null);
                      return (
                      <tr><td colSpan="5" style={{padding:'0 16px 16px',background:'rgba(255,255,255,0.01)'}}>
                        {setTuningFields.length > 0 && (
                        <div style={{display:'grid',gridTemplateColumns:'repeat(4, 1fr)',gap:'8px 16px',fontSize:'12px',padding:'12px',borderRadius:'8px',background:'var(--nv-bg)',border:'1px solid var(--nv-border)'}}>
                          {setTuningFields.map(f => (
                            <div key={f.key} style={{display:'flex',justifyContent:'space-between',gap:'8px'}}>
                              <span style={{color:'var(--nv-text-dim)'}}>{f.label}</span>
                              <span className="mono" style={{color:'var(--nv-green)',fontWeight:600}}>{cfg[f.key]}</span>
                            </div>
                          ))}
                        </div>
                        )}
                        {extraCfgEntries.length > 0 && (
                          <div style={{marginTop:setTuningFields.length>0?'10px':0,padding:'12px',borderRadius:'8px',background:'var(--nv-bg)',border:'1px solid var(--nv-border)'}}>
                            <div style={{fontSize:'11px',fontWeight:600,color:'#64b4ff',textTransform:'uppercase',letterSpacing:'0.05em',marginBottom:'8px'}}>Additional Config</div>
                            <div style={{display:'grid',gridTemplateColumns:'repeat(3, 1fr)',gap:'6px 16px',fontSize:'12px'}}>
                              {extraCfgEntries.map(([k, v]) => (
                                <div key={k} style={{display:'flex',justifyContent:'space-between',gap:'8px'}}>
                                  <span style={{color:'var(--nv-text-dim)'}}>{k}</span>
                                  <span className="mono" style={{color:'#64b4ff',fontWeight:600}}>{String(v)}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                        {overrideCount > 0 && (
                          <div style={{marginTop:'10px',padding:'12px',borderRadius:'8px',background:'var(--nv-bg)',border:'1px solid var(--nv-border)'}}>
                            <div style={{fontSize:'11px',fontWeight:600,color:'var(--nv-green)',textTransform:'uppercase',letterSpacing:'0.05em',marginBottom:'8px'}}>Custom Overrides</div>
                            <div style={{display:'grid',gridTemplateColumns:'repeat(3, 1fr)',gap:'6px 16px',fontSize:'12px'}}>
                              {Object.entries(ovr).map(([k, v]) => (
                                <div key={k} style={{display:'flex',justifyContent:'space-between',gap:'8px'}}>
                                  <span style={{color:'var(--nv-text-dim)'}}>{k}</span>
                                  <span className="mono" style={{color:'var(--nv-green)',fontWeight:600}}>{String(v)}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                        {setTuningFields.length===0 && extraCfgEntries.length===0 && overrideCount===0 && (
                          <div style={{padding:'12px',textAlign:'center',color:'var(--nv-text-dim)',fontSize:'12px'}}>No configuration set</div>
                        )}
                      </td></tr>
                      );
                    })()}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
        <Pagination page={pg.page} totalPages={pg.totalPages} totalItems={pg.totalItems}
          pageSize={pg.pageSize} onPageChange={pg.setPage} onPageSizeChange={pg.setPageSize} />
      </div>

      {/* ===== Preset Matrices Section ===== */}
      <div style={{marginTop:'32px'}}>
        <div style={{fontSize:'16px',fontWeight:700,color:'#fff',marginBottom:'16px'}}>Preset Matrices</div>
        <div className="card">
          <div style={{overflowX:'auto'}}>
            <table className="runs-table">
              <thead>
                <tr>
                  <th>Name</th><th>Description</th><th style={{textAlign:'right'}}>Datasets</th><th style={{textAlign:'right'}}>Presets</th><th style={{textAlign:'right'}}>Jobs</th><th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {presetMatricesLoading ? (
                  <tr><td colSpan="6" style={{textAlign:'center',padding:'60px',color:'var(--nv-text-muted)'}}>
                    <div className="spinner spinner-lg" style={{margin:'0 auto 12px'}}></div><div>Loading matrices…</div>
                  </td></tr>
                ) : matrices.length === 0 ? (
                  <tr><td colSpan="6" style={{textAlign:'center',padding:'40px',color:'var(--nv-text-muted)'}}>
                    <div style={{marginBottom:'8px',fontSize:'15px'}}>No preset matrices configured</div>
                    <div style={{fontSize:'12px',color:'var(--nv-text-dim)'}}>Click "Add Matrix" to group datasets and presets for batch scheduling.</div>
                  </td></tr>
                ) : mxPg.pageData.map(m => {
                  const dsCount = (m.dataset_names || []).length;
                  const prCount = (m.preset_names || []).length;
                  const jobCount = dsCount * prCount;
                  const isExpanded = expandedMatrixId === m.id;
                  return (
                    <React.Fragment key={m.id}>
                      <tr>
                        <td>
                          <span style={{color:'#fff',fontWeight:600,cursor:'pointer'}} onClick={()=>setExpandedMatrixId(isExpanded?null:m.id)}>
                            <span style={{display:'inline-block',transform:isExpanded?'rotate(90deg)':'rotate(0deg)',transition:'transform 0.15s',marginRight:'6px',fontSize:'10px'}}>{"\u25B6"}</span>
                            {m.name}
                          </span>
                        </td>
                        <td style={{color:'var(--nv-text-muted)',fontSize:'13px'}}>{m.description || "\u2014"}</td>
                        <td style={{textAlign:'right'}}><span className="badge badge-na">{dsCount} dataset{dsCount!==1?'s':''}</span></td>
                        <td style={{textAlign:'right'}}><span className="badge badge-na">{prCount} preset{prCount!==1?'s':''}</span></td>
                        <td style={{textAlign:'right'}}><span className="badge" style={{background:'rgba(118,185,0,0.1)',color:'var(--nv-green)',border:'1px solid rgba(118,185,0,0.25)'}}>{jobCount} job{jobCount!==1?'s':''}</span></td>
                        <td>
                          <div style={{display:'flex',gap:'6px',flexWrap:'nowrap'}}>
                            <button className="btn btn-primary btn-sm" onClick={()=>handleTriggerMatrix(m)} title="Run all jobs in this matrix"><IconPlay /> Run</button>
                            <button className="btn btn-secondary btn-sm" onClick={()=>handleEditMatrix(m)} title="Edit"><IconEdit /> Edit</button>
                            <button className="btn btn-sm" onClick={()=>handleDeleteMatrix(m.id,m.name)} title="Delete" style={{background:'rgba(255,80,80,0.1)',color:'#ff5050',border:'1px solid rgba(255,80,80,0.2)'}}><IconTrash /> Delete</button>
                          </div>
                        </td>
                      </tr>
                      {isExpanded && (
                        <tr><td colSpan="6" style={{padding:'0 16px 16px',background:'rgba(255,255,255,0.01)'}}>
                          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'16px',padding:'12px',borderRadius:'8px',background:'var(--nv-bg)',border:'1px solid var(--nv-border)'}}>
                            <div>
                              <div style={{fontSize:'11px',fontWeight:600,color:'var(--nv-green)',textTransform:'uppercase',letterSpacing:'0.05em',marginBottom:'8px'}}>Datasets ({dsCount})</div>
                              <div style={{display:'flex',flexWrap:'wrap',gap:'6px'}}>
                                {(m.dataset_names || []).map(n => (
                                  <span key={n} className="badge badge-na" style={{fontSize:'12px'}}>{n}</span>
                                ))}
                              </div>
                            </div>
                            <div>
                              <div style={{fontSize:'11px',fontWeight:600,color:'var(--nv-green)',textTransform:'uppercase',letterSpacing:'0.05em',marginBottom:'8px'}}>Presets ({prCount})</div>
                              <div style={{display:'flex',flexWrap:'wrap',gap:'6px'}}>
                                {(m.preset_names || []).map(n => (
                                  <span key={n} className="badge badge-na" style={{fontSize:'12px'}}>{n}</span>
                                ))}
                              </div>
                            </div>
                          </div>
                          {(m.preferred_runner_id || m.gpu_type_filter) && (
                            <div style={{marginTop:'8px',padding:'8px 12px',borderRadius:'8px',background:'rgba(118,185,0,0.04)',border:'1px solid rgba(118,185,0,0.15)',fontSize:'12px',display:'flex',gap:'20px',flexWrap:'wrap'}}>
                              <span style={{color:'var(--nv-text-dim)'}}>Runner Targeting:</span>
                              {m.preferred_runner_id && <span style={{color:'#fff'}}><span style={{color:'var(--nv-green)',fontWeight:600}}>Preferred Runner:</span> #{m.preferred_runner_id}</span>}
                              {m.gpu_type_filter && <span style={{color:'#fff'}}><span style={{color:'var(--nv-green)',fontWeight:600}}>GPU Type:</span> {m.gpu_type_filter}</span>}
                            </div>
                          )}
                          {(m.git_ref || m.git_commit) && (
                            <div style={{marginTop:'8px',padding:'8px 12px',borderRadius:'8px',background:'rgba(118,185,0,0.04)',border:'1px solid rgba(118,185,0,0.15)',fontSize:'12px',display:'flex',gap:'20px',flexWrap:'wrap'}}>
                              <span style={{color:'var(--nv-text-dim)'}}>Git Override:</span>
                              {m.git_ref && <span style={{color:'#fff'}}><span style={{color:'var(--nv-green)',fontWeight:600}}>Branch:</span> <span className="mono">{m.git_ref}</span></span>}
                              {m.git_commit && <span style={{color:'#fff'}}><span style={{color:'var(--nv-green)',fontWeight:600}}>Commit:</span> <span className="mono">{m.git_commit.substring(0,12)}</span></span>}
                            </div>
                          )}
                          {(m.tags || []).length > 0 && (
                            <div style={{marginTop:'8px',padding:'8px 12px',borderRadius:'8px',background:'var(--nv-bg)',border:'1px solid var(--nv-border)',fontSize:'12px'}}>
                              <span style={{color:'var(--nv-text-dim)',marginRight:'8px'}}>Tags:</span>
                              {m.tags.map(t => <span key={t} className="badge badge-na" style={{fontSize:'11px',marginRight:'4px'}}>{t}</span>)}
                            </div>
                          )}
                        </td></tr>
                      )}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
          <Pagination page={mxPg.page} totalPages={mxPg.totalPages} totalItems={mxPg.totalItems}
            pageSize={mxPg.pageSize} onPageChange={mxPg.setPage} onPageSizeChange={mxPg.setPageSize} />
        </div>
      </div>

      {showForm && (
        <PresetFormModal
          preset={editPreset}
          onClose={()=>setShowForm(false)}
          onSaved={()=>{setShowForm(false);onRefresh();}}
        />
      )}
      {showMatrixForm && (
        <PresetMatrixFormModal
          matrix={editMatrix}
          onClose={()=>setShowMatrixForm(false)}
          onSaved={()=>{setShowMatrixForm(false);onRefresh();}}
        />
      )}
      {triggerMatrixTarget && (
        <MatrixTriggerModal
          matrix={triggerMatrixTarget}
          onClose={()=>setTriggerMatrixTarget(null)}
        />
      )}
    </>
  );
}

function MatrixTriggerModal({ matrix, onClose }) {
  const matrixId = matrix.id;
  const matrixName = matrix.name;
  const savedRef = matrix.git_ref || "";
  const savedCommit = matrix.git_commit || "";

  const [gitMode, setGitMode] = useState(savedCommit ? "commit" : savedRef ? "branch" : "default");
  const [gitRef, setGitRef] = useState(savedRef);
  const [gitCommit, setGitCommit] = useState(savedCommit);
  const [remoteBranches, setRemoteBranches] = useState([]);
  const [defaultRef, setDefaultRef] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [nsysProfile, setNsysProfile] = useState(!!matrix.nsys_profile);

  useEffect(() => {
    fetch("/api/portal-settings").then(r=>r.json()).then(s => {
      setDefaultRef(s.run_code_ref || "");
    }).catch(()=>{});
    fetch("/api/settings/git-info").then(r=>r.json()).then(info => {
      if (info.available) setRemoteBranches(info.remote_branches || []);
    }).catch(()=>{});
  }, []);

  async function handleTrigger() {
    setSubmitting(true);
    try {
      const body = { nsys_profile: nsysProfile };
      if (gitMode === "branch" && gitRef.trim()) {
        body.git_ref = gitRef.trim();
      } else if (gitMode === "commit" && gitRef.trim()) {
        body.git_ref = gitRef.trim();
        if (gitCommit.trim()) body.git_commit = gitCommit.trim();
      }
      const res = await fetch(`/api/preset-matrices/${matrixId}/trigger`, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(body),
      });
      if (!res.ok) { const d = await res.json().catch(()=>({})); alert(d.detail || "Trigger failed"); return; }
      const data = await res.json();
      alert(`Queued ${data.job_count} job${data.job_count!==1?'s':''} from matrix "${data.matrix_name}"`);
      onClose();
    } catch (err) {
      alert("Trigger failed: " + err.message);
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
          <h2 style={{fontSize:'16px',fontWeight:700,color:'#fff'}}>Run Matrix: {matrixName}</h2>
          <button className="btn btn-ghost btn-icon" onClick={onClose} style={{borderRadius:'50%'}}><IconX /></button>
        </div>
        <div style={{padding:'24px',display:'flex',flexDirection:'column',gap:'16px'}}>
          <div style={{fontSize:'13px',color:'var(--nv-text-muted)',lineHeight:'1.6'}}>
            All jobs in this matrix will be queued. Choose which code version runners should checkout.
          </div>

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
              {(savedRef || savedCommit)
                ? <>Uses the matrix's saved git config: <span className="mono" style={{color:'var(--nv-green)'}}>{savedRef}{savedCommit ? ` @ ${savedCommit.substring(0,12)}` : ''}</span></>
                : <>Uses the Runner Execution Branch from Settings{defaultRef ? `: ${defaultRef}` : '.'}</>
              }
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
                    ? "Runners will fetch and checkout the latest commit from this branch."
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
                    Runners will checkout this exact commit. Leave empty to use the latest from the branch above.
                  </div>
                </div>
              )}
            </>
          )}

          {/* Profiling */}
          <div style={{borderTop:'1px solid var(--nv-border)',paddingTop:'16px'}}>
            <label style={{display:'flex',alignItems:'center',gap:'8px',cursor:'pointer',fontSize:'13px',color:'var(--nv-text-muted)'}}>
              <input type="checkbox" checked={nsysProfile} onChange={e => setNsysProfile(e.target.checked)}
                style={{width:'16px',height:'16px',accentColor:'var(--nv-green)'}} />
              Enable Nsight Systems Profile
            </label>
            <div style={hintStyle}>
              All jobs in this matrix will be profiled with <code style={{fontSize:'11px'}}>nsys profile</code>.
            </div>
          </div>
        </div>
        <div className="modal-foot">
          <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
          <button type="button" className="btn btn-primary" onClick={handleTrigger}
            disabled={submitting}
            style={{flex:1,justifyContent:'center'}}>
            {submitting ? <><span className="spinner" style={{marginRight:'8px'}}></span>Triggering…</> : <><IconPlay /> Run Matrix</>}
          </button>
        </div>
      </div>
    </div>
  );
}


function PresetFormModal({ preset, onClose, onSaved }) {
  const isEdit = !!preset;
  const existingConfig = (isEdit && typeof preset.config === 'object') ? preset.config : {};
  const existingOverrides = (isEdit && typeof preset.overrides === 'object') ? preset.overrides : {};
  const tuningKeySet = new Set(TUNING_FIELDS.map(f => f.key));
  const [name, setName] = useState(preset?.name || "");
  const [description, setDescription] = useState(preset?.description || "");
  const [tags, setTags] = useState((preset?.tags || []).join(", "));
  const [config, setConfig] = useState(() => {
    const c = {};
    TUNING_FIELDS.forEach(f => { c[f.key] = existingConfig[f.key] != null ? String(existingConfig[f.key]) : ""; });
    return c;
  });
  const [extraConfig, setExtraConfig] = useState(() => {
    return Object.entries(existingConfig)
      .filter(([k]) => !tuningKeySet.has(k) && k !== "use_heuristics")
      .map(([k, v]) => ({ key: k, value: String(v) }));
  });
  const [overrides, setOverrides] = useState(() => {
    return Object.entries(existingOverrides).map(([k, v]) => ({ key: k, value: String(v) }));
  });
  const [useDefaults, setUseDefaults] = useState(() => {
    return isEdit && existingConfig.use_heuristics === true;
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  function setField(key, val) { setConfig(c=>({...c,[key]:val})); }

  function handleToggleDefaults(checked) {
    setUseDefaults(checked);
    if (checked) {
      const cleared = {};
      TUNING_FIELDS.forEach(f => { cleared[f.key] = ""; });
      setConfig(cleared);
    }
  }

  function addExtraConfig() { setExtraConfig(prev => [...prev, { key: "", value: "" }]); }
  function removeExtraConfig(idx) { setExtraConfig(prev => prev.filter((_, i) => i !== idx)); }
  function updateExtraConfig(idx, field, val) {
    setExtraConfig(prev => prev.map((o, i) => i === idx ? { ...o, [field]: val } : o));
  }

  function addOverride() { setOverrides(prev => [...prev, { key: "", value: "" }]); }
  function removeOverride(idx) { setOverrides(prev => prev.filter((_, i) => i !== idx)); }
  function updateOverride(idx, field, val) {
    setOverrides(prev => prev.map((o, i) => i === idx ? { ...o, [field]: val } : o));
  }

  function _parseValue(v) {
    const s = v.trim();
    if (s === "true") return true;
    if (s === "false") return false;
    if (s !== "" && !isNaN(Number(s))) return Number(s);
    return s;
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!name.trim()) return;
    setSaving(true);
    setError("");
    const parsedConfig = {};
    if (useDefaults) {
      parsedConfig["use_heuristics"] = true;
    } else {
      TUNING_FIELDS.forEach(f => {
        const raw = config[f.key];
        if (raw !== "" && raw != null) {
          parsedConfig[f.key] = f.type === "int" ? parseInt(raw, 10) : parseFloat(raw);
        }
      });
    }
    extraConfig.forEach(o => {
      const k = o.key.trim();
      if (k) parsedConfig[k] = _parseValue(o.value);
    });
    const parsedOverrides = {};
    overrides.forEach(o => {
      const k = o.key.trim();
      if (k) parsedOverrides[k] = _parseValue(o.value);
    });
    const payload = {
      name: name.trim(),
      description: description || null,
      config: parsedConfig,
      tags: tags ? tags.split(",").map(t=>t.trim()).filter(Boolean) : [],
      overrides: parsedOverrides,
    };
    try {
      const url = isEdit ? `/api/managed-presets/${preset.id}` : "/api/managed-presets";
      const method = isEdit ? "PUT" : "POST";
      const res = await fetch(url, {
        method, headers: {"Content-Type":"application/json"},
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const data = await res.json().catch(()=>({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      onSaved();
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  }

  const labelStyle = {display:'block',fontSize:'11px',fontWeight:500,color:'var(--nv-text-muted)',marginBottom:'4px',textTransform:'uppercase',letterSpacing:'0.04em'};

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{maxWidth:'700px'}} onClick={e=>e.stopPropagation()}>
        <div className="modal-head">
          <h2 style={{fontSize:'16px',fontWeight:700,color:'#fff'}}>{isEdit ? "Edit Preset" : "Create Preset"}</h2>
          <button className="btn btn-ghost btn-icon" onClick={onClose} style={{borderRadius:'50%'}}><IconX /></button>
        </div>
        <form onSubmit={handleSubmit}>
          <div style={{padding:'24px',display:'flex',flexDirection:'column',gap:'16px',maxHeight:'65vh',overflowY:'auto'}}>
            {error && (
              <div style={{padding:'10px 14px',borderRadius:'8px',background:'rgba(255,50,50,0.08)',border:'1px solid rgba(255,50,50,0.2)',color:'#ff5050',fontSize:'13px'}}>{error}</div>
            )}
            <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'12px'}}>
              <div>
                <label style={labelStyle}>Name *</label>
                <input className="input" style={{width:'100%'}} value={name} onChange={e=>setName(e.target.value)} placeholder="e.g. single_gpu" required />
              </div>
              <div>
                <label style={labelStyle}>Description</label>
                <input className="input" style={{width:'100%'}} value={description} onChange={e=>setDescription(e.target.value)} placeholder="Optional description" />
              </div>
            </div>

            <div style={{padding:'12px 16px',borderRadius:'8px',background:useDefaults?'rgba(118,185,0,0.06)':'rgba(255,255,255,0.02)',border:'1px solid '+(useDefaults?'rgba(118,185,0,0.2)':'var(--nv-border)'),transition:'all 0.15s'}}>
              <label style={{display:'flex',alignItems:'center',gap:'8px',cursor:'pointer',fontSize:'13px',color:useDefaults?'var(--nv-green)':'var(--nv-text-muted)'}}>
                <input type="checkbox" checked={useDefaults} onChange={e=>handleToggleDefaults(e.target.checked)}
                  style={{width:'16px',height:'16px',accentColor:'var(--nv-green)'}} />
                Use Default Heuristics
              </label>
              <div style={{fontSize:'11px',color:'var(--nv-text-dim)',marginTop:'4px',lineHeight:'1.5'}}>
                When enabled, no tuning parameters are stored — the system will compute optimal worker counts, batch sizes, and GPU allocations automatically at runtime.
              </div>
            </div>

            {!useDefaults && TUNING_GROUPS.map(g => (
              <div key={g.label}>
                <div style={{fontSize:'12px',fontWeight:600,color:'var(--nv-green)',marginBottom:'8px',textTransform:'uppercase',letterSpacing:'0.05em'}}>{g.label}</div>
                <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill, minmax(140px, 1fr))',gap:'8px'}}>
                  {g.fields.map(f => (
                    <div key={f.key}>
                      <label style={{...labelStyle,fontSize:'10px'}}>{f.label}</label>
                      <input className="input" style={{width:'100%',fontSize:'13px'}} type="number" step={f.type==='float'?'any':'1'} value={config[f.key]} onChange={e=>setField(f.key,e.target.value)} placeholder="\u2014" />
                    </div>
                  ))}
                </div>
              </div>
            ))}

            <div>
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'8px'}}>
                <div style={{fontSize:'12px',fontWeight:600,color:'#64b4ff',textTransform:'uppercase',letterSpacing:'0.05em'}}>Additional Config</div>
                <button type="button" className="btn btn-ghost btn-sm" onClick={addExtraConfig} style={{fontSize:'11px'}}>
                  <IconPlus /> Add Config Field
                </button>
              </div>
              <div style={{fontSize:'11px',color:'var(--nv-text-dim)',marginBottom:'8px'}}>
                Arbitrary key=value pairs stored in the preset config (e.g. ray_address, hybrid)
              </div>
              {extraConfig.length === 0 ? (
                <div style={{padding:'12px',textAlign:'center',color:'var(--nv-text-dim)',fontSize:'12px',border:'1px dashed var(--nv-border)',borderRadius:'8px'}}>
                  No additional config fields
                </div>
              ) : (
                <div style={{display:'flex',flexDirection:'column',gap:'6px'}}>
                  {extraConfig.map((o, idx) => (
                    <div key={idx} style={{display:'flex',gap:'8px',alignItems:'center'}}>
                      <input className="input" style={{flex:1,fontSize:'13px'}} value={o.key} onChange={e=>updateExtraConfig(idx,'key',e.target.value)} placeholder="key (e.g. ray_address)" />
                      <span style={{color:'var(--nv-text-dim)',fontSize:'13px'}}>=</span>
                      <input className="input" style={{flex:1,fontSize:'13px'}} value={o.value} onChange={e=>updateExtraConfig(idx,'value',e.target.value)} placeholder="value" />
                      <button type="button" className="btn btn-ghost btn-sm btn-icon" onClick={()=>removeExtraConfig(idx)}
                        style={{color:'#ff5050',flexShrink:0}} title="Remove">
                        <IconTrash />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div>
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'8px'}}>
                <div style={{fontSize:'12px',fontWeight:600,color:'var(--nv-green)',textTransform:'uppercase',letterSpacing:'0.05em'}}>Custom Overrides</div>
                <button type="button" className="btn btn-ghost btn-sm" onClick={addOverride} style={{fontSize:'11px'}}>
                  <IconPlus /> Add Override
                </button>
              </div>
              <div style={{fontSize:'11px',color:'var(--nv-text-dim)',marginBottom:'8px'}}>
                Arbitrary key=value pairs passed as --override arguments at runtime
              </div>
              {overrides.length === 0 ? (
                <div style={{padding:'12px',textAlign:'center',color:'var(--nv-text-dim)',fontSize:'12px',border:'1px dashed var(--nv-border)',borderRadius:'8px'}}>
                  No custom overrides configured
                </div>
              ) : (
                <div style={{display:'flex',flexDirection:'column',gap:'6px'}}>
                  {overrides.map((o, idx) => (
                    <div key={idx} style={{display:'flex',gap:'8px',alignItems:'center'}}>
                      <input className="input" style={{flex:1,fontSize:'13px'}} value={o.key} onChange={e=>updateOverride(idx,'key',e.target.value)} placeholder="key (e.g. embed_modality)" />
                      <span style={{color:'var(--nv-text-dim)',fontSize:'13px'}}>=</span>
                      <input className="input" style={{flex:1,fontSize:'13px'}} value={o.value} onChange={e=>updateOverride(idx,'value',e.target.value)} placeholder="value (e.g. text_image)" />
                      <button type="button" className="btn btn-ghost btn-sm btn-icon" onClick={()=>removeOverride(idx)}
                        style={{color:'#ff5050',flexShrink:0}} title="Remove">
                        <IconTrash />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div>
              <label style={labelStyle}>Tags</label>
              <input className="input" style={{width:'100%'}} value={tags} onChange={e=>setTags(e.target.value)} placeholder="Comma-separated tags" />
            </div>
          </div>
          <div className="modal-foot">
            <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button type="submit" disabled={saving||!name.trim()} className="btn btn-primary" style={{flex:1,justifyContent:'center'}}>
              {saving ? <><span className="spinner" style={{marginRight:'8px'}}></span>Saving…</> : isEdit ? "Update Preset" : "Create Preset"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

/* ===== Preset Matrix Form Modal ===== */
function PresetMatrixFormModal({ matrix, onClose, onSaved }) {
  const isEdit = !!matrix;
  const [name, setName] = useState(matrix?.name || "");
  const [description, setDescription] = useState(matrix?.description || "");
  const [tags, setTags] = useState((matrix?.tags || []).join(", "));
  const [selectedDatasets, setSelectedDatasets] = useState(new Set(matrix?.dataset_names || []));
  const [selectedPresets, setSelectedPresets] = useState(new Set(matrix?.preset_names || []));
  const [availableDatasets, setAvailableDatasets] = useState([]);
  const [availablePresets, setAvailablePresets] = useState([]);
  const [runnersList, setRunnersList] = useState([]);
  const [gpuTypes, setGpuTypes] = useState([]);
  const [preferredRunnerId, setPreferredRunnerId] = useState(matrix?.preferred_runner_id || "");
  const [gpuTypeFilter, setGpuTypeFilter] = useState(matrix?.gpu_type_filter || "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const existingGitRef = matrix?.git_ref || "";
  const existingGitCommit = matrix?.git_commit || "";
  const [gitMode, setGitMode] = useState(existingGitCommit ? "commit" : existingGitRef ? "branch" : "default");
  const [gitRef, setGitRef] = useState(existingGitRef);
  const [gitCommit, setGitCommit] = useState(existingGitCommit);
  const [remoteBranches, setRemoteBranches] = useState([]);
  const [defaultRef, setDefaultRef] = useState("");
  const [nsysProfile, setNsysProfile] = useState(!!matrix?.nsys_profile);

  useEffect(() => {
    fetch("/api/config").then(r=>r.json()).then(cfg => {
      setAvailableDatasets(cfg.datasets || []);
      setAvailablePresets(cfg.presets || []);
    }).catch(()=>{});
    fetch("/api/runners").then(r=>r.json()).then(runners => {
      setRunnersList(runners || []);
    }).catch(()=>{});
    fetch("/api/runners/gpu-types").then(r=>r.json()).then(types => {
      setGpuTypes(types || []);
    }).catch(()=>{});
    fetch("/api/portal-settings").then(r=>r.json()).then(s => {
      setDefaultRef(s.run_code_ref || "");
    }).catch(()=>{});
    fetch("/api/settings/git-info").then(r=>r.json()).then(info => {
      if (info.available) setRemoteBranches(info.remote_branches || []);
    }).catch(()=>{});
  }, []);

  function toggleDataset(d) {
    setSelectedDatasets(prev => { const s = new Set(prev); s.has(d) ? s.delete(d) : s.add(d); return s; });
  }
  function togglePreset(p) {
    setSelectedPresets(prev => { const s = new Set(prev); s.has(p) ? s.delete(p) : s.add(p); return s; });
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!name.trim() || selectedDatasets.size === 0 || selectedPresets.size === 0) return;
    setSaving(true);
    setError("");
    const payload = {
      name: name.trim(),
      description: description || null,
      dataset_names: [...selectedDatasets],
      preset_names: [...selectedPresets],
      tags: tags ? tags.split(",").map(t=>t.trim()).filter(Boolean) : [],
      preferred_runner_id: preferredRunnerId ? Number(preferredRunnerId) : null,
      gpu_type_filter: gpuTypeFilter || null,
      git_ref: null,
      git_commit: null,
      nsys_profile: nsysProfile,
    };
    if (gitMode === "branch" && gitRef.trim()) {
      payload.git_ref = gitRef.trim();
    } else if (gitMode === "commit" && gitRef.trim()) {
      payload.git_ref = gitRef.trim();
      if (gitCommit.trim()) payload.git_commit = gitCommit.trim();
    }
    try {
      const url = isEdit ? `/api/preset-matrices/${matrix.id}` : "/api/preset-matrices";
      const method = isEdit ? "PUT" : "POST";
      const res = await fetch(url, {
        method, headers: {"Content-Type":"application/json"},
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const data = await res.json().catch(()=>({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      onSaved();
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  }

  const labelStyle = {display:'block',fontSize:'11px',fontWeight:500,color:'var(--nv-text-muted)',marginBottom:'4px',textTransform:'uppercase',letterSpacing:'0.04em'};
  const jobCount = selectedDatasets.size * selectedPresets.size;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{maxWidth:'700px'}} onClick={e=>e.stopPropagation()}>
        <div className="modal-head">
          <h2 style={{fontSize:'16px',fontWeight:700,color:'#fff'}}>{isEdit ? "Edit Preset Matrix" : "Create Preset Matrix"}</h2>
          <button className="btn btn-ghost btn-icon" onClick={onClose} style={{borderRadius:'50%'}}><IconX /></button>
        </div>
        <form onSubmit={handleSubmit}>
          <div style={{padding:'24px',display:'flex',flexDirection:'column',gap:'16px',maxHeight:'65vh',overflowY:'auto'}}>
            {error && (
              <div style={{padding:'10px 14px',borderRadius:'8px',background:'rgba(255,50,50,0.08)',border:'1px solid rgba(255,50,50,0.2)',color:'#ff5050',fontSize:'13px'}}>{error}</div>
            )}
            <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'12px'}}>
              <div>
                <label style={labelStyle}>Name *</label>
                <input className="input" style={{width:'100%'}} value={name} onChange={e=>setName(e.target.value)} placeholder="e.g. nightly_full_suite" required />
              </div>
              <div>
                <label style={labelStyle}>Description</label>
                <input className="input" style={{width:'100%'}} value={description} onChange={e=>setDescription(e.target.value)} placeholder="Optional description" />
              </div>
            </div>

            <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'16px'}}>
              <div>
                <label style={labelStyle}>Datasets * <span style={{color:'var(--nv-green)',fontWeight:700}}>({selectedDatasets.size} selected)</span></label>
                <div style={{border:'1px solid var(--nv-border)',borderRadius:'8px',maxHeight:'200px',overflowY:'auto',padding:'4px'}}>
                  {availableDatasets.length === 0 ? (
                    <div style={{padding:'12px',textAlign:'center',color:'var(--nv-text-dim)',fontSize:'12px'}}>Loading…</div>
                  ) : availableDatasets.map(d => (
                    <label key={d} style={{display:'flex',alignItems:'center',gap:'8px',padding:'6px 8px',cursor:'pointer',borderRadius:'4px',fontSize:'13px',color:selectedDatasets.has(d)?'#fff':'var(--nv-text-muted)',background:selectedDatasets.has(d)?'rgba(118,185,0,0.08)':'transparent'}}>
                      <input type="checkbox" checked={selectedDatasets.has(d)} onChange={()=>toggleDataset(d)} style={{accentColor:'var(--nv-green)'}} />
                      {d}
                    </label>
                  ))}
                </div>
              </div>
              <div>
                <label style={labelStyle}>Presets * <span style={{color:'var(--nv-green)',fontWeight:700}}>({selectedPresets.size} selected)</span></label>
                <div style={{border:'1px solid var(--nv-border)',borderRadius:'8px',maxHeight:'200px',overflowY:'auto',padding:'4px'}}>
                  {availablePresets.length === 0 ? (
                    <div style={{padding:'12px',textAlign:'center',color:'var(--nv-text-dim)',fontSize:'12px'}}>Loading…</div>
                  ) : availablePresets.map(p => (
                    <label key={p} style={{display:'flex',alignItems:'center',gap:'8px',padding:'6px 8px',cursor:'pointer',borderRadius:'4px',fontSize:'13px',color:selectedPresets.has(p)?'#fff':'var(--nv-text-muted)',background:selectedPresets.has(p)?'rgba(118,185,0,0.08)':'transparent'}}>
                      <input type="checkbox" checked={selectedPresets.has(p)} onChange={()=>togglePreset(p)} style={{accentColor:'var(--nv-green)'}} />
                      {p}
                    </label>
                  ))}
                </div>
              </div>
            </div>

            <div style={{padding:'12px 16px',borderRadius:'8px',background:jobCount>0?'rgba(118,185,0,0.06)':'rgba(255,255,255,0.02)',border:'1px solid '+(jobCount>0?'rgba(118,185,0,0.2)':'var(--nv-border)'),textAlign:'center'}}>
              <span style={{fontSize:'14px',fontWeight:600,color:jobCount>0?'var(--nv-green)':'var(--nv-text-dim)'}}>
                {selectedDatasets.size} dataset{selectedDatasets.size!==1?'s':''} &times; {selectedPresets.size} preset{selectedPresets.size!==1?'s':''} = {jobCount} job{jobCount!==1?'s':''}
              </span>
              <div style={{fontSize:'11px',color:'var(--nv-text-dim)',marginTop:'4px'}}>Each trigger will create {jobCount} job{jobCount!==1?'s':''} in the queue</div>
            </div>

            <div style={{padding:'12px 16px',borderRadius:'8px',background:'rgba(255,255,255,0.02)',border:'1px solid var(--nv-border)'}}>
              <div style={{fontSize:'11px',fontWeight:600,color:'var(--nv-text-muted)',textTransform:'uppercase',letterSpacing:'0.04em',marginBottom:'10px'}}>Runner Targeting (optional)</div>
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'12px'}}>
                <div>
                  <label style={labelStyle}>Preferred Runner</label>
                  <select className="input" style={{width:'100%'}} value={preferredRunnerId} onChange={e=>{setPreferredRunnerId(e.target.value); if(e.target.value) setGpuTypeFilter("");}}>
                    <option value="">Any runner</option>
                    {runnersList.map(r => <option key={r.id} value={r.id}>#{r.id} — {r.name}{r.gpu_type ? ` (${r.gpu_type})` : ''}</option>)}
                  </select>
                </div>
                <div>
                  <label style={labelStyle}>GPU Type Filter</label>
                  <select className="input" style={{width:'100%'}} value={gpuTypeFilter} disabled={!!preferredRunnerId} onChange={e=>setGpuTypeFilter(e.target.value)}>
                    <option value="">Any GPU</option>
                    {gpuTypes.map(g => <option key={g} value={g}>{g}</option>)}
                  </select>
                  {preferredRunnerId && <div style={{fontSize:'10px',color:'var(--nv-text-dim)',marginTop:'4px'}}>Disabled when a preferred runner is selected</div>}
                </div>
              </div>
            </div>

            {/* Git Checkout Override */}
            <div style={{padding:'12px 16px',borderRadius:'8px',background:'rgba(255,255,255,0.02)',border:'1px solid var(--nv-border)'}}>
              <div style={{fontSize:'11px',fontWeight:600,color:'var(--nv-text-muted)',textTransform:'uppercase',letterSpacing:'0.04em',marginBottom:'10px'}}>
                Git Checkout Override (optional)
              </div>
              <div style={{fontSize:'11px',color:'var(--nv-text-dim)',lineHeight:'1.5',marginBottom:'12px'}}>
                Pin all jobs from this matrix to a specific branch or commit. Overrides the global Runner Execution Branch setting{defaultRef ? ` (${defaultRef})` : ''}.
              </div>
              <div style={{display:'flex',gap:'6px',marginBottom:'12px'}}>
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

              {(gitMode === "branch" || gitMode === "commit") && (
                <div style={{display:'flex',flexDirection:'column',gap:'10px'}}>
                  <div>
                    <label style={labelStyle}>Remote / Branch</label>
                    <input className="input" style={{width:'100%'}} value={gitRef}
                      onChange={e=>setGitRef(e.target.value)}
                      placeholder="e.g. nvidia/main or origin/feat/my-branch" />
                    {remoteBranches.length > 0 && (
                      <div style={{marginTop:'6px',display:'flex',gap:'4px',flexWrap:'wrap',maxHeight:'60px',overflow:'auto'}}>
                        {remoteBranches.slice(0, 15).map(b => (
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
                  </div>
                  {gitMode === "commit" && (
                    <div>
                      <label style={labelStyle}>Commit SHA</label>
                      <input className="input mono" style={{width:'100%',fontSize:'12px'}} value={gitCommit}
                        onChange={e=>setGitCommit(e.target.value)}
                        placeholder="e.g. a1b2c3d4e5f6 or full 40-char SHA" />
                      <div style={{fontSize:'10px',color:'var(--nv-text-dim)',marginTop:'4px'}}>
                        Leave empty to use the latest commit from the branch above.
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Profiling */}
            <div style={{padding:'12px 16px',borderRadius:'8px',background:'rgba(255,255,255,0.02)',border:'1px solid var(--nv-border)'}}>
              <div style={{fontSize:'11px',fontWeight:600,color:'var(--nv-text-muted)',textTransform:'uppercase',letterSpacing:'0.04em',marginBottom:'10px'}}>
                Profiling (optional)
              </div>
              <label style={{display:'flex',alignItems:'center',gap:'8px',cursor:'pointer',fontSize:'13px',color:'var(--nv-text-muted)'}}>
                <input type="checkbox" checked={nsysProfile} onChange={e => setNsysProfile(e.target.checked)}
                  style={{width:'16px',height:'16px',accentColor:'var(--nv-green)'}} />
                Enable Nsight Systems Profile
              </label>
              <div style={{fontSize:'10px',color:'var(--nv-text-dim)',marginTop:'4px',lineHeight:'1.5'}}>
                All jobs spawned by this matrix will be wrapped with <code style={{fontSize:'10px'}}>nsys profile</code>. Can be overridden at trigger time.
              </div>
            </div>

            <div>
              <label style={labelStyle}>Tags</label>
              <input className="input" style={{width:'100%'}} value={tags} onChange={e=>setTags(e.target.value)} placeholder="Comma-separated tags applied to all spawned jobs" />
            </div>
          </div>
          <div className="modal-foot">
            <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button type="submit" disabled={saving||!name.trim()||selectedDatasets.size===0||selectedPresets.size===0} className="btn btn-primary" style={{flex:1,justifyContent:'center'}}>
              {saving ? <><span className="spinner" style={{marginRight:'8px'}}></span>Saving…</> : isEdit ? "Update Matrix" : "Create Matrix"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
