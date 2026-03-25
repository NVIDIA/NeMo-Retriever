/* ===== Trigger Modal ===== */
function TriggerModal({ onClose, onTriggered }) {
  const [datasets, setDatasets] = useState([]);
  const [presets, setPresets] = useState([]);
  const [runners, setRunners] = useState([]);
  const [dataset, setDataset] = useState("");
  const [preset, setPreset] = useState("");
  const [runnerId, setRunnerId] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    fetch("/api/config").then(r=>r.json()).then(cfg => {
      setDatasets(cfg.datasets || []);
      setPresets(cfg.presets || []);
      if (cfg.datasets?.length) setDataset(cfg.datasets[0]);
      if (cfg.presets?.length) setPreset(cfg.presets[0]);
    });
    fetch("/api/runners").then(r=>r.json()).then(setRunners).catch(()=>{});
  }, []);

  const onlineRunners = runners.filter(r => r.status === "online" || r.status === "paused");

  async function handleSubmit(e) {
    e.preventDefault();
    if (!dataset) return;
    setSubmitting(true);
    try {
      const payload = { dataset, preset: preset || null, runner_id: runnerId ? parseInt(runnerId, 10) : null };
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

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{maxWidth:'460px'}} onClick={e=>e.stopPropagation()}>
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
              <div style={{fontSize:'11px',color:'var(--nv-text-dim)',marginTop:'4px'}}>
                {onlineRunners.length === 0
                  ? "No runners online. The job will wait until a runner becomes available."
                  : `${onlineRunners.length} runner${onlineRunners.length!==1?'s':''} online`}
              </div>
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
