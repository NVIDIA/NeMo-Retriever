/* ===== Datasets View ===== */
function DatasetsView({ managedDatasets, loading, onRefresh }) {
  const [showForm, setShowForm] = useState(false);
  const [editDataset, setEditDataset] = useState(null);
  const [importing, setImporting] = useState(false);
  const [tab, setTab] = useState("active");
  const [inactiveDatasets, setInactiveDatasets] = useState([]);
  const [inactiveLoading, setInactiveLoading] = useState(false);
  const pg = usePagination(tab === "active" ? managedDatasets : inactiveDatasets, 25);

  const fetchInactive = useCallback(async () => {
    setInactiveLoading(true);
    try {
      const res = await fetch("/api/managed-datasets/inactive");
      if (res.ok) setInactiveDatasets(await res.json());
    } catch {} finally { setInactiveLoading(false); }
  }, []);

  useEffect(() => { if (tab === "inactive") fetchInactive(); }, [tab]);

  function handleCreate() { setEditDataset(null); setShowForm(true); }
  function handleEdit(ds) { setEditDataset(ds); setShowForm(true); }
  async function handleDelete(id, name) {
    if (!confirm(`Delete dataset "${name}"? It will be moved to the Inactive tab and can be restored later.`)) return;
    try {
      await fetch(`/api/managed-datasets/${id}`, { method: "DELETE" });
      onRefresh();
      if (tab === "inactive") fetchInactive();
    } catch {}
  }
  async function handleRestore(id, name) {
    if (!confirm(`Restore dataset "${name}"?`)) return;
    try {
      await fetch(`/api/managed-datasets/${id}/restore`, { method: "POST" });
      fetchInactive();
      onRefresh();
    } catch {}
  }

  function handleExport() {
    window.location.href = "/api/managed-datasets/export.yaml";
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
        const res = await fetch("/api/managed-datasets/import", { method: "POST", body: formData });
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

  const tabBtnStyle = (active) => ({
    padding:'6px 16px',borderRadius:'6px',fontSize:'13px',fontWeight:600,cursor:'pointer',border:'none',
    background: active ? 'rgba(118,185,0,0.15)' : 'transparent',
    color: active ? 'var(--nv-green)' : 'var(--nv-text-muted)',
    transition:'all 0.15s',
  });

  const isActive = tab === "active";
  const displayData = isActive ? managedDatasets : inactiveDatasets;
  const isLoading = isActive ? loading : inactiveLoading;

  return (
    <>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'20px',flexWrap:'wrap',gap:'12px'}}>
        <div style={{display:'flex',gap:'8px',alignItems:'center'}}>
          <div style={{display:'flex',gap:'4px',background:'var(--nv-surface)',borderRadius:'8px',padding:'3px',border:'1px solid var(--nv-border)'}}>
            <button style={tabBtnStyle(isActive)} onClick={()=>setTab("active")}>Active</button>
            <button style={tabBtnStyle(!isActive)} onClick={()=>setTab("inactive")}>
              Inactive{inactiveDatasets.length > 0 ? ` (${inactiveDatasets.length})` : ''}
            </button>
          </div>
          {isActive && <>
            <button className="btn btn-primary" onClick={handleCreate}><IconPlus /> Add Dataset</button>
            <button className="btn btn-secondary" onClick={handleExport} disabled={managedDatasets.length===0} title="Export all datasets to YAML"><IconDownload /> Export YAML</button>
            <button className="btn btn-secondary" onClick={handleImport} disabled={importing} title="Import datasets from a YAML file">
              {importing ? <><span className="spinner" style={{marginRight:'6px'}}></span>Importing…</> : <><IconUpload /> Import YAML</>}
            </button>
          </>}
        </div>
        <button className="btn btn-secondary btn-icon" onClick={()=>{onRefresh(); if(!isActive) fetchInactive();}} title="Refresh"><IconRefresh /></button>
      </div>

      <div className="card">
        <div style={{overflowX:'auto'}}>
          <table className="runs-table">
            <thead>
              <tr>
                <th>Name</th><th>Path</th><th>Input Type</th><th>Eval Mode</th><th>Query CSV</th>
                <th>Recall</th><th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                <tr><td colSpan="7" style={{textAlign:'center',padding:'60px',color:'var(--nv-text-muted)'}}>
                  <div className="spinner spinner-lg" style={{margin:'0 auto 12px'}}></div><div>Loading datasets…</div>
                </td></tr>
              ) : displayData.length === 0 ? (
                <tr><td colSpan="7" style={{textAlign:'center',padding:'40px',color:'var(--nv-text-muted)'}}>
                  {isActive ? <>
                    <div style={{marginBottom:'8px',fontSize:'15px'}}>No datasets configured</div>
                    <div style={{fontSize:'12px',color:'var(--nv-text-dim)'}}>Click "Add Dataset" to register one. Datasets from test_configs.yaml are imported automatically on startup.</div>
                  </> : <>
                    <div style={{marginBottom:'8px',fontSize:'15px'}}>No inactive datasets</div>
                    <div style={{fontSize:'12px',color:'var(--nv-text-dim)'}}>Deleted datasets appear here and can be restored.</div>
                  </>}
                </td></tr>
              ) : pg.pageData.map(ds => (
                <tr key={ds.id} style={!isActive ? {opacity:0.7} : undefined}>
                  <td>
                    <div style={{display:'flex',alignItems:'center',gap:'6px'}}>
                      <span style={{color:'#fff',fontWeight:600}}>{ds.name}</span>
                      {ds.distribute && (
                        <span title="Distributed to runners" style={{
                          padding:'1px 6px',borderRadius:'4px',fontSize:'9px',fontWeight:700,
                          textTransform:'uppercase',letterSpacing:'0.05em',
                          background:'rgba(100,180,255,0.12)',color:'rgb(100,180,255)',
                          border:'1px solid rgba(100,180,255,0.25)',
                        }}>Dist</span>
                      )}
                      {!isActive && (
                        <span style={{
                          padding:'1px 6px',borderRadius:'4px',fontSize:'9px',fontWeight:700,
                          textTransform:'uppercase',letterSpacing:'0.05em',
                          background:'rgba(255,180,50,0.12)',color:'rgb(255,180,50)',
                          border:'1px solid rgba(255,180,50,0.25)',
                        }}>Inactive</span>
                      )}
                    </div>
                  </td>
                  <td className="mono" style={{fontSize:'12px',color:'var(--nv-text-muted)',maxWidth:'250px',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}} title={ds.path}>{ds.path}</td>
                  <td><span className="badge badge-na">{ds.input_type}</span></td>
                  <td><span className="badge" style={{background: ds.evaluation_mode==='beir' ? 'rgba(118,185,0,0.15)' : 'rgba(100,180,255,0.1)', color: ds.evaluation_mode==='beir' ? 'var(--nv-green)' : 'rgb(100,180,255)', border: ds.evaluation_mode==='beir' ? '1px solid rgba(118,185,0,0.3)' : '1px solid rgba(100,180,255,0.2)'}}>{ds.evaluation_mode || "recall"}</span></td>
                  <td className="mono" style={{fontSize:'11px',color:'var(--nv-text-dim)',maxWidth:'200px',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}} title={ds.query_csv||''}>{ds.query_csv || "\u2014"}</td>
                  <td>{ds.recall_required ? <span className="badge badge-pass">Yes</span> : <span className="badge badge-na">No</span>}</td>
                  <td>
                    {isActive ? (
                      <div style={{display:'flex',gap:'6px',flexWrap:'nowrap'}}>
                        <button className="btn btn-secondary btn-sm" onClick={()=>handleEdit(ds)} title="Edit"><IconEdit /> Edit</button>
                        <button className="btn btn-sm" onClick={()=>handleDelete(ds.id,ds.name)} title="Delete" style={{background:'rgba(255,80,80,0.1)',color:'#ff5050',border:'1px solid rgba(255,80,80,0.2)'}}><IconTrash /> Delete</button>
                      </div>
                    ) : (
                      <button className="btn btn-sm" onClick={()=>handleRestore(ds.id,ds.name)} title="Restore" style={{background:'rgba(118,185,0,0.1)',color:'var(--nv-green)',border:'1px solid rgba(118,185,0,0.2)'}}>
                        <IconRefresh /> Restore
                      </button>
                    )}
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
        <DatasetFormModal
          dataset={editDataset}
          onClose={()=>setShowForm(false)}
          onSaved={()=>{setShowForm(false);onRefresh();}}
        />
      )}
    </>
  );
}

function DatasetFormModal({ dataset, onClose, onSaved }) {
  const isEdit = !!dataset;
  const [form, setForm] = useState({
    name: dataset?.name || "",
    path: dataset?.path || "",
    query_csv: dataset?.query_csv || "",
    input_type: dataset?.input_type || "pdf",
    recall_required: dataset?.recall_required || false,
    recall_match_mode: dataset?.recall_match_mode || "pdf_page",
    recall_adapter: dataset?.recall_adapter || "none",
    evaluation_mode: dataset?.evaluation_mode || "recall",
    beir_loader: dataset?.beir_loader || "vidore_hf",
    beir_dataset_name: dataset?.beir_dataset_name || "",
    beir_split: dataset?.beir_split || "test",
    beir_query_language: dataset?.beir_query_language || "",
    beir_doc_id_field: dataset?.beir_doc_id_field || "pdf_basename",
    beir_ks: (dataset?.beir_ks || [1,3,5,10]).join(", "),
    embed_model_name: dataset?.embed_model_name || "",
    embed_modality: dataset?.embed_modality || "text",
    embed_granularity: dataset?.embed_granularity || "element",
    extract_page_as_image: dataset?.extract_page_as_image || false,
    extract_infographics: dataset?.extract_infographics || false,
    distribute: dataset ? (dataset.distribute ?? true) : true,
    description: dataset?.description || "",
    tags: (dataset?.tags || []).join(", "),
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const isBeir = form.evaluation_mode === "beir";

  function set(field, val) { setForm(f=>({...f,[field]:val})); }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!form.name.trim() || !form.path.trim()) return;
    setSaving(true);
    setError("");
    const parsedKs = form.beir_ks ? form.beir_ks.split(",").map(s=>parseInt(s.trim(),10)).filter(n=>!isNaN(n)&&n>0) : null;
    const payload = {
      ...form,
      query_csv: form.query_csv || null,
      beir_dataset_name: form.beir_dataset_name || null,
      beir_query_language: form.beir_query_language || null,
      beir_ks: (isBeir && parsedKs && parsedKs.length > 0) ? parsedKs : null,
      embed_model_name: form.embed_model_name || null,
      description: form.description || null,
      tags: form.tags ? form.tags.split(",").map(t=>t.trim()).filter(Boolean) : [],
    };
    try {
      const url = isEdit ? `/api/managed-datasets/${dataset.id}` : "/api/managed-datasets";
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

  const labelStyle = {display:'block',fontSize:'12px',fontWeight:500,color:'var(--nv-text-muted)',marginBottom:'6px',textTransform:'uppercase',letterSpacing:'0.04em'};
  const sectionStyle = {padding:'12px',borderRadius:'8px',background:'rgba(118,185,0,0.04)',border:'1px solid rgba(118,185,0,0.15)',display:'flex',flexDirection:'column',gap:'12px'};

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{maxWidth:'640px'}} onClick={e=>e.stopPropagation()}>
        <div className="modal-head">
          <h2 style={{fontSize:'16px',fontWeight:700,color:'#fff'}}>{isEdit ? "Edit Dataset" : "Add Dataset"}</h2>
          <button className="btn btn-ghost btn-icon" onClick={onClose} style={{borderRadius:'50%'}}><IconX /></button>
        </div>
        <form onSubmit={handleSubmit}>
          <div style={{padding:'24px',display:'flex',flexDirection:'column',gap:'16px',maxHeight:'60vh',overflowY:'auto'}}>
            {error && (
              <div style={{padding:'10px 14px',borderRadius:'8px',background:'rgba(255,50,50,0.08)',border:'1px solid rgba(255,50,50,0.2)',color:'#ff5050',fontSize:'13px'}}>{error}</div>
            )}
            <div>
              <label style={labelStyle}>Name *</label>
              <input className="input" style={{width:'100%'}} value={form.name} onChange={e=>set('name',e.target.value)} placeholder="e.g. bo20" required />
            </div>
            <div>
              <label style={labelStyle}>Dataset Path *</label>
              <input className="input" style={{width:'100%'}} value={form.path} onChange={e=>set('path',e.target.value)} placeholder="/path/to/dataset" required />
            </div>
            <div>
              <label style={labelStyle}>Query CSV</label>
              <input className="input" style={{width:'100%'}} value={form.query_csv} onChange={e=>set('query_csv',e.target.value)} placeholder="/path/to/query_gt.csv (optional)" />
            </div>
            <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'12px'}}>
              <div>
                <label style={labelStyle}>Input Type</label>
                <select className="select" style={{width:'100%'}} value={form.input_type} onChange={e=>set('input_type',e.target.value)}>
                  <option value="pdf">pdf</option>
                  <option value="image">image</option>
                  <option value="text">text</option>
                </select>
              </div>
              <div>
                <label style={labelStyle}>Evaluation Mode</label>
                <select className="select" style={{width:'100%'}} value={form.evaluation_mode} onChange={e=>set('evaluation_mode',e.target.value)}>
                  <option value="recall">recall</option>
                  <option value="beir">beir</option>
                </select>
              </div>
            </div>

            {!isBeir && (
              <>
                <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'12px'}}>
                  <div>
                    <label style={labelStyle}>Recall Required</label>
                    <label style={{display:'flex',alignItems:'center',gap:'8px',cursor:'pointer',fontSize:'13px',color:'#fff',marginTop:'6px'}}>
                      <input type="checkbox" checked={form.recall_required} onChange={e=>set('recall_required',e.target.checked)} />
                      {form.recall_required ? "Yes" : "No"}
                    </label>
                  </div>
                  <div>
                    <label style={labelStyle}>Match Mode</label>
                    <select className="select" style={{width:'100%'}} value={form.recall_match_mode} onChange={e=>set('recall_match_mode',e.target.value)}>
                      <option value="pdf_page">pdf_page</option>
                      <option value="pdf_only">pdf_only</option>
                    </select>
                  </div>
                </div>
                <div>
                  <label style={labelStyle}>Recall Adapter</label>
                  <select className="select" style={{width:'100%'}} value={form.recall_adapter} onChange={e=>set('recall_adapter',e.target.value)}>
                    <option value="none">none</option>
                    <option value="page_plus_one">page_plus_one</option>
                    <option value="financebench_json">financebench_json</option>
                  </select>
                </div>
              </>
            )}

            {isBeir && (
              <div style={sectionStyle}>
                <div style={{fontSize:'12px',fontWeight:600,color:'var(--nv-green)',textTransform:'uppercase',letterSpacing:'0.05em'}}>BEIR Evaluation</div>
                <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'12px'}}>
                  <div>
                    <label style={labelStyle}>BEIR Loader</label>
                    <select className="select" style={{width:'100%'}} value={form.beir_loader} onChange={e=>set('beir_loader',e.target.value)}>
                      <option value="vidore_hf">vidore_hf</option>
                    </select>
                  </div>
                  <div>
                    <label style={labelStyle}>BEIR Dataset Name</label>
                    <input className="input" style={{width:'100%'}} value={form.beir_dataset_name} onChange={e=>set('beir_dataset_name',e.target.value)} placeholder="e.g. vidore_v3_computer_science" />
                  </div>
                </div>
                <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'12px'}}>
                  <div>
                    <label style={labelStyle}>BEIR Split</label>
                    <input className="input" style={{width:'100%'}} value={form.beir_split} onChange={e=>set('beir_split',e.target.value)} placeholder="test" />
                  </div>
                  <div>
                    <label style={labelStyle}>BEIR Query Language</label>
                    <input className="input" style={{width:'100%'}} value={form.beir_query_language} onChange={e=>set('beir_query_language',e.target.value)} placeholder="Optional (e.g. en, fr)" />
                  </div>
                </div>
                <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'12px'}}>
                  <div>
                    <label style={labelStyle}>Doc ID Field</label>
                    <select className="select" style={{width:'100%'}} value={form.beir_doc_id_field} onChange={e=>set('beir_doc_id_field',e.target.value)}>
                      <option value="pdf_basename">pdf_basename</option>
                      <option value="pdf_page">pdf_page</option>
                      <option value="source_id">source_id</option>
                      <option value="path">path</option>
                    </select>
                  </div>
                  <div>
                    <label style={labelStyle}>K Values</label>
                    <input className="input" style={{width:'100%'}} value={form.beir_ks} onChange={e=>set('beir_ks',e.target.value)} placeholder="1, 3, 5, 10" />
                  </div>
                </div>
              </div>
            )}

            <div style={{...sectionStyle, background:'rgba(100,180,255,0.04)', border:'1px solid rgba(100,180,255,0.15)'}}>
              <div style={{fontSize:'12px',fontWeight:600,color:'rgb(100,180,255)',textTransform:'uppercase',letterSpacing:'0.05em'}}>Embedding & Extraction</div>
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'12px'}}>
                <div>
                  <label style={labelStyle}>Embed Model</label>
                  <input className="input" style={{width:'100%'}} value={form.embed_model_name} onChange={e=>set('embed_model_name',e.target.value)} placeholder="nvidia/llama-nemotron-embed-1b-v2" />
                </div>
                <div>
                  <label style={labelStyle}>Embed Modality</label>
                  <select className="select" style={{width:'100%'}} value={form.embed_modality} onChange={e=>set('embed_modality',e.target.value)}>
                    <option value="text">text</option>
                    <option value="image">image</option>
                    <option value="text_image">text_image</option>
                  </select>
                </div>
              </div>
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:'12px'}}>
                <div>
                  <label style={labelStyle}>Embed Granularity</label>
                  <select className="select" style={{width:'100%'}} value={form.embed_granularity} onChange={e=>set('embed_granularity',e.target.value)}>
                    <option value="element">element</option>
                    <option value="page">page</option>
                  </select>
                </div>
                <div>
                  <label style={labelStyle}>Page as Image</label>
                  <label style={{display:'flex',alignItems:'center',gap:'8px',cursor:'pointer',fontSize:'13px',color:'#fff',marginTop:'6px'}}>
                    <input type="checkbox" checked={form.extract_page_as_image} onChange={e=>set('extract_page_as_image',e.target.checked)} />
                    {form.extract_page_as_image ? "Yes" : "No"}
                  </label>
                </div>
                <div>
                  <label style={labelStyle}>Infographics</label>
                  <label style={{display:'flex',alignItems:'center',gap:'8px',cursor:'pointer',fontSize:'13px',color:'#fff',marginTop:'6px'}}>
                    <input type="checkbox" checked={form.extract_infographics} onChange={e=>set('extract_infographics',e.target.checked)} />
                    {form.extract_infographics ? "Yes" : "No"}
                  </label>
                </div>
              </div>
            </div>

            <div style={{display:'flex',alignItems:'center',gap:'12px',borderTop:'1px solid var(--nv-border)',paddingTop:'16px'}}>
              <label className="toggle">
                <input type="checkbox" checked={form.distribute} onChange={e=>set('distribute',e.target.checked)} />
                <span className="toggle-slider"></span>
              </label>
              <div>
                <span style={{fontSize:'14px',color:'var(--nv-text)'}}>Distribute to runners</span>
                <div style={{fontSize:'11px',color:'var(--nv-text-dim)',marginTop:'2px'}}>
                  When enabled, runners will automatically download this dataset from the portal instead of requiring it to exist locally.
                </div>
              </div>
            </div>

            <div>
              <label style={labelStyle}>Description</label>
              <input className="input" style={{width:'100%'}} value={form.description} onChange={e=>set('description',e.target.value)} placeholder="Optional description" />
            </div>
            <div>
              <label style={labelStyle}>Tags</label>
              <input className="input" style={{width:'100%'}} value={form.tags} onChange={e=>set('tags',e.target.value)} placeholder="Comma-separated tags" />
            </div>
          </div>
          <div className="modal-foot">
            <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button type="submit" disabled={saving||!form.name.trim()||!form.path.trim()} className="btn btn-primary" style={{flex:1,justifyContent:'center'}}>
              {saving ? <><span className="spinner" style={{marginRight:'8px'}}></span>Saving…</> : isEdit ? "Update Dataset" : "Add Dataset"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
