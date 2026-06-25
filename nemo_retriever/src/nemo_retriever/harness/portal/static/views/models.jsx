/* ===== Models Playground View ===== */
function ModelsView() {
  const [models, setModels] = useState([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState(null);
  const [activeTab, setActiveTab] = useState("embed");

  const [embedTexts, setEmbedTexts] = useState("What is machine learning?");
  const [embedPrefix, setEmbedPrefix] = useState("query: ");
  const [embedResult, setEmbedResult] = useState(null);
  const [embedLoading, setEmbedLoading] = useState(false);
  const [embedError, setEmbedError] = useState(null);

  const [rerankQuery, setRerankQuery] = useState("What is machine learning?");
  const [rerankDocs, setRerankDocs] = useState("Machine learning is a subset of artificial intelligence that enables systems to learn from data.\nParis is the capital of France and is known for the Eiffel Tower.\nDeep learning uses neural networks with many layers to model complex patterns.");
  const [rerankResult, setRerankResult] = useState(null);
  const [rerankLoading, setRerankLoading] = useState(false);
  const [rerankError, setRerankError] = useState(null);

  const [imageB64, setImageB64] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [ocrMerge, setOcrMerge] = useState("paragraph");
  const [detectThreshold, setDetectThreshold] = useState(0.25);
  const [visionResult, setVisionResult] = useState(null);
  const [visionLoading, setVisionLoading] = useState(false);
  const [visionError, setVisionError] = useState(null);

  useEffect(() => {
    fetch("/api/models").then(r => r.json()).then(data => {
      setModels(data);
      if (data.length > 0) {
        const embedModel = data.find(m => m.type === "embedding");
        if (embedModel) setSelectedModel(embedModel.id);
      }
    }).catch(() => {}).finally(() => setModelsLoading(false));
  }, []);

  const embedModels = models.filter(m => m.type === "embedding");
  const rerankModels = models.filter(m => m.type === "reranker");
  const visionModels = models.filter(m => m.input_type === "image");

  function handleImageUpload(e) {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    setVisionResult(null); setVisionError(null);
    const img = new window.Image();
    img.onload = () => {
      const MAX_DIM = 2048;
      let w = img.width, h = img.height;
      if (w > MAX_DIM || h > MAX_DIM) {
        const scale = MAX_DIM / Math.max(w, h);
        w = Math.round(w * scale);
        h = Math.round(h * scale);
      }
      const canvas = document.createElement("canvas");
      canvas.width = w; canvas.height = h;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, w, h);
      const dataUrl = canvas.toDataURL("image/png");
      setImageB64(dataUrl);
      const THUMB = 600;
      let tw = w, th = h;
      if (tw > THUMB || th > THUMB) {
        const ts = THUMB / Math.max(tw, th);
        tw = Math.round(tw * ts); th = Math.round(th * ts);
      }
      const tc = document.createElement("canvas");
      tc.width = tw; tc.height = th;
      tc.getContext("2d").drawImage(img, 0, 0, tw, th);
      setImagePreview(tc.toDataURL("image/jpeg", 0.8));
      URL.revokeObjectURL(img.src);
    };
    img.onerror = () => { setVisionError("Failed to load image file."); URL.revokeObjectURL(img.src); };
    img.src = URL.createObjectURL(file);
  }

  function selectTab(model) {
    setSelectedModel(model.id);
    if (model.type === "reranker") setActiveTab("rerank");
    else if (model.input_type === "image") setActiveTab("vision");
    else setActiveTab("embed");
  }

  async function runEmbed() {
    const texts = embedTexts.split("\n").map(t => t.trim()).filter(Boolean);
    if (texts.length === 0) return;
    const modelId = selectedModel || (embedModels[0] && embedModels[0].id);
    if (!modelId) return;
    setEmbedLoading(true); setEmbedError(null); setEmbedResult(null);
    try {
      const res = await fetch("/api/models/embed", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ model_id: modelId, texts, prefix: embedPrefix }),
      });
      if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(d.detail || `HTTP ${res.status}`); }
      setEmbedResult(await res.json());
    } catch (err) { setEmbedError(err.message); } finally { setEmbedLoading(false); }
  }

  async function runRerank() {
    const docs = rerankDocs.split("\n").map(t => t.trim()).filter(Boolean);
    if (!rerankQuery.trim() || docs.length === 0) return;
    const modelId = rerankModels.length > 0 ? rerankModels[0].id : "nvidia/llama-nemotron-rerank-1b-v2";
    setRerankLoading(true); setRerankError(null); setRerankResult(null);
    try {
      const res = await fetch("/api/models/rerank", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ model_id: modelId, query: rerankQuery.trim(), documents: docs }),
      });
      if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(d.detail || `HTTP ${res.status}`); }
      setRerankResult(await res.json());
    } catch (err) { setRerankError(err.message); } finally { setRerankLoading(false); }
  }

  async function runVision() {
    if (!imageB64 || !selectedModel) return;
    const model = models.find(m => m.id === selectedModel);
    if (!model || model.input_type !== "image") return;
    setVisionLoading(true); setVisionError(null); setVisionResult(null);
    const endpoint = model.type === "ocr" ? "/api/models/ocr"
      : model.type === "document-parser" ? "/api/models/parse"
      : model.type === "object-detection" ? "/api/models/detect"
      : null;
    if (!endpoint) { setVisionError(`No test endpoint for model type: ${model.type}`); setVisionLoading(false); return; }
    try {
      const payload = { model_id: selectedModel, image_b64: imageB64 };
      if (model.type === "ocr") payload.merge_level = ocrMerge;
      if (model.type === "object-detection") payload.score_threshold = detectThreshold;
      const res = await fetch(endpoint, {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(d.detail || `HTTP ${res.status}`); }
      setVisionResult(await res.json());
    } catch (err) { setVisionError(err.message); } finally { setVisionLoading(false); }
  }

  const labelStyle = {display:'block',fontSize:'11px',fontWeight:600,color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.04em',marginBottom:'6px'};
  const tabStyle = (active) => ({
    padding:'8px 20px',fontSize:'13px',fontWeight:active?600:400,cursor:'pointer',
    borderBottom:active?'2px solid var(--nv-green)':'2px solid transparent',
    color:active?'var(--nv-green)':'var(--nv-text-muted)',background:'transparent',border:'none',borderBottomWidth:'2px',borderBottomStyle:'solid',
    transition:'all 0.15s',
  });
  const TYPE_COLORS = { embedding:'#76b900', reranker:'#bb86fc', ocr:'#64b4ff', 'object-detection':'#fcd34d', 'document-parser':'#ff8c00', asr:'#00d4aa' };

  function scoreColor(score) {
    if (score > 10) return '#76b900';
    if (score > 0) return '#8bc34a';
    if (score > -10) return '#ff8c00';
    return '#ff5050';
  }

  if (modelsLoading) {
    return <div style={{textAlign:'center',padding:'60px'}}><div className="spinner spinner-lg" style={{margin:'0 auto 12px'}}></div><div style={{color:'var(--nv-text-muted)'}}>Loading models…</div></div>;
  }

  const categories = [...new Set(models.map(m => m.category || "Other"))];

  return (
    <>
      {/* Model cards grouped by category */}
      {categories.map(cat => (
        <div key={cat} style={{marginBottom:'24px'}}>
          <div style={{fontSize:'12px',fontWeight:700,color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.06em',marginBottom:'10px',paddingLeft:'2px'}}>{cat}</div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(260px,1fr))',gap:'12px'}}>
            {models.filter(m => (m.category || "Other") === cat).map(m => {
              const tc = TYPE_COLORS[m.type] || '#aaa';
              return (
                <div key={m.id} className="card" style={{padding:'14px',cursor:'pointer',
                  border: selectedModel === m.id ? `1px solid ${tc}` : '1px solid var(--nv-border)',
                  background: selectedModel === m.id ? `${tc}08` : 'var(--nv-surface)',
                }} onClick={() => selectTab(m)}>
                  <div style={{display:'flex',alignItems:'center',gap:'8px',marginBottom:'6px'}}>
                    <IconCpu />
                    <span style={{fontSize:'13px',fontWeight:600,color:'#fff'}}>{m.name}</span>
                  </div>
                  <div style={{fontSize:'11px',color:'var(--nv-text-muted)',marginBottom:'8px',lineHeight:'1.5'}}>{m.description}</div>
                  <div style={{display:'flex',gap:'6px',flexWrap:'wrap'}}>
                    <span style={{fontSize:'10px',padding:'2px 8px',borderRadius:'4px',fontWeight:600,textTransform:'uppercase',
                      background:`${tc}18`,color:tc,border:`1px solid ${tc}30`}}>{m.type}</span>
                    {m.input_type && <span style={{fontSize:'10px',padding:'2px 6px',borderRadius:'4px',background:'rgba(255,255,255,0.04)',color:'var(--nv-text-dim)'}}>{m.input_type}</span>}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}

      {/* Tab bar */}
      <div style={{display:'flex',borderBottom:'1px solid var(--nv-border)',marginBottom:'20px',flexWrap:'wrap'}}>
        <button style={tabStyle(activeTab==='embed')} onClick={()=>setActiveTab('embed')}>Embedding</button>
        <button style={tabStyle(activeTab==='rerank')} onClick={()=>setActiveTab('rerank')}>Reranker</button>
        <button style={tabStyle(activeTab==='vision')} onClick={()=>setActiveTab('vision')}>Document AI</button>
      </div>

      {/* Embedding test panel */}
      {activeTab === 'embed' && (
        <div className="card" style={{padding:'24px'}}>
          <div style={{fontSize:'15px',fontWeight:600,color:'#fff',marginBottom:'16px'}}>Embedding Model Test</div>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'16px',marginBottom:'16px'}}>
            <div>
              <label style={labelStyle}>Model</label>
              <select className="select" style={{width:'100%'}} value={selectedModel || ''} onChange={e=>setSelectedModel(e.target.value)}>
                {embedModels.map(m => <option key={m.id} value={m.id}>{m.name} ({m.id})</option>)}
              </select>
            </div>
            <div>
              <label style={labelStyle}>Query Prefix</label>
              <input className="input" style={{width:'100%'}} value={embedPrefix} onChange={e=>setEmbedPrefix(e.target.value)} placeholder="query: " />
              <div style={{fontSize:'10px',color:'var(--nv-text-dim)',marginTop:'4px'}}>Use "query: " for queries, "passage: " for documents.</div>
            </div>
          </div>
          <div style={{marginBottom:'16px'}}>
            <label style={labelStyle}>Input Texts (one per line)</label>
            <textarea className="input" rows={5} style={{width:'100%',fontFamily:'var(--font-mono)',fontSize:'12px',resize:'vertical'}}
              value={embedTexts} onChange={e=>setEmbedTexts(e.target.value)} placeholder={"Enter one or more texts, each on its own line…"} />
          </div>
          <button className="btn btn-primary" onClick={runEmbed} disabled={embedLoading || !embedTexts.trim()}>
            {embedLoading ? <><span className="spinner" style={{marginRight:'8px'}}></span>Embedding…</> : 'Run Embedding'}
          </button>
          {embedError && <div style={{marginTop:'16px',padding:'12px 16px',borderRadius:'6px',background:'rgba(255,80,80,0.08)',border:'1px solid rgba(255,80,80,0.2)',color:'#ff5050',fontSize:'13px'}}>{embedError}</div>}
          {embedResult && (
            <div style={{marginTop:'20px'}}>
              <div style={{display:'flex',gap:'16px',flexWrap:'wrap',marginBottom:'16px'}}>
                {[{label:'Model',value:embedResult.model_id},{label:'Vectors',value:embedResult.count},{label:'Dimensions',value:embedResult.embedding_dim},{label:'Load Time',value:`${embedResult.model_load_ms}ms`},{label:'Embed Time',value:`${embedResult.embed_ms}ms`}].map(s => (
                  <div key={s.label} style={{background:'rgba(118,185,0,0.06)',border:'1px solid rgba(118,185,0,0.12)',borderRadius:'6px',padding:'8px 14px',minWidth:'100px'}}>
                    <div style={{fontSize:'10px',color:'var(--nv-text-dim)',textTransform:'uppercase',fontWeight:600,marginBottom:'2px'}}>{s.label}</div>
                    <div className="mono" style={{fontSize:'13px',color:'var(--nv-green)',fontWeight:600}}>{s.value}</div>
                  </div>
                ))}
              </div>
              <table className="runs-table" style={{fontSize:'12px'}}>
                <thead><tr><th>#</th><th>Text</th><th>Dim</th><th>Norm</th><th>Vector Preview</th></tr></thead>
                <tbody>
                  {embedResult.results.map((r, i) => (
                    <tr key={i}>
                      <td>{i+1}</td>
                      <td style={{maxWidth:'300px',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{r.text}</td>
                      <td className="mono">{r.embedding_dim}</td>
                      <td className="mono">{r.embedding_norm}</td>
                      <td className="mono" style={{fontSize:'10px',color:'var(--nv-text-dim)',maxWidth:'300px',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>
                        [{r.embedding_preview.map(v => v.toFixed(4)).join(', ')}, …]
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Reranker test panel */}
      {activeTab === 'rerank' && (
        <div className="card" style={{padding:'24px'}}>
          <div style={{fontSize:'15px',fontWeight:600,color:'#fff',marginBottom:'16px'}}>Reranker Model Test</div>
          <div style={{marginBottom:'16px'}}>
            <label style={labelStyle}>Model</label>
            <select className="select" style={{width:'100%',maxWidth:'500px'}} value={rerankModels[0]?.id || ''} disabled={rerankModels.length <= 1}>
              {rerankModels.map(m => <option key={m.id} value={m.id}>{m.name} ({m.id})</option>)}
              {rerankModels.length === 0 && <option value="">No reranker models available</option>}
            </select>
          </div>
          <div style={{marginBottom:'16px'}}>
            <label style={labelStyle}>Query</label>
            <input className="input" style={{width:'100%'}} value={rerankQuery} onChange={e=>setRerankQuery(e.target.value)} placeholder="Enter your search query…" />
          </div>
          <div style={{marginBottom:'16px'}}>
            <label style={labelStyle}>Documents (one per line)</label>
            <textarea className="input" rows={6} style={{width:'100%',fontFamily:'var(--font-mono)',fontSize:'12px',resize:'vertical'}}
              value={rerankDocs} onChange={e=>setRerankDocs(e.target.value)} placeholder={"Enter candidate documents, one per line…"} />
            <div style={{fontSize:'10px',color:'var(--nv-text-dim)',marginTop:'4px'}}>Each line is scored independently against the query. Higher scores indicate greater relevance.</div>
          </div>
          <button className="btn btn-primary" onClick={runRerank} disabled={rerankLoading || !rerankQuery.trim() || !rerankDocs.trim()}>
            {rerankLoading ? <><span className="spinner" style={{marginRight:'8px'}}></span>Scoring…</> : 'Run Reranking'}
          </button>
          {rerankError && <div style={{marginTop:'16px',padding:'12px 16px',borderRadius:'6px',background:'rgba(255,80,80,0.08)',border:'1px solid rgba(255,80,80,0.2)',color:'#ff5050',fontSize:'13px'}}>{rerankError}</div>}
          {rerankResult && (
            <div style={{marginTop:'20px'}}>
              <div style={{display:'flex',gap:'16px',flexWrap:'wrap',marginBottom:'16px'}}>
                {[{label:'Model',value:rerankResult.model_id},{label:'Query',value:rerankResult.query.length>40?rerankResult.query.slice(0,40)+'…':rerankResult.query},{label:'Documents',value:rerankResult.count},{label:'Load Time',value:`${rerankResult.model_load_ms}ms`},{label:'Score Time',value:`${rerankResult.score_ms}ms`}].map(s => (
                  <div key={s.label} style={{background:'rgba(138,43,226,0.06)',border:'1px solid rgba(138,43,226,0.12)',borderRadius:'6px',padding:'8px 14px',minWidth:'100px'}}>
                    <div style={{fontSize:'10px',color:'var(--nv-text-dim)',textTransform:'uppercase',fontWeight:600,marginBottom:'2px'}}>{s.label}</div>
                    <div className="mono" style={{fontSize:'13px',color:'#bb86fc',fontWeight:600}}>{s.value}</div>
                  </div>
                ))}
              </div>
              <div style={{fontSize:'13px',fontWeight:600,color:'#fff',marginBottom:'10px'}}>Results (sorted by relevance)</div>
              {rerankResult.results.map((r, i) => {
                const maxScore = Math.max(...rerankResult.results.map(x => x.score));
                const minScore = Math.min(...rerankResult.results.map(x => x.score));
                const range = maxScore - minScore || 1;
                const barPct = Math.max(5, ((r.score - minScore) / range) * 100);
                return (
                  <div key={i} style={{marginBottom:'8px',background:'var(--nv-surface)',border:'1px solid var(--nv-border)',borderRadius:'8px',padding:'12px 16px'}}>
                    <div style={{display:'flex',alignItems:'center',gap:'12px',marginBottom:'6px'}}>
                      <span style={{fontSize:'18px',fontWeight:700,color:scoreColor(r.score),minWidth:'28px'}}>#{r.rank}</span>
                      <div style={{flex:1,minWidth:0}}>
                        <div style={{fontSize:'13px',color:'#fff',lineHeight:'1.5',wordBreak:'break-word'}}>{r.document}</div>
                      </div>
                      <span className="mono" style={{fontSize:'16px',fontWeight:700,color:scoreColor(r.score),whiteSpace:'nowrap'}}>
                        {r.score > 0 ? '+' : ''}{r.score}
                      </span>
                    </div>
                    <div style={{height:'4px',borderRadius:'2px',background:'rgba(255,255,255,0.04)',overflow:'hidden'}}>
                      <div style={{height:'100%',width:`${barPct}%`,borderRadius:'2px',background:scoreColor(r.score),transition:'width 0.3s'}} />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Document AI / Vision test panel */}
      {activeTab === 'vision' && (
        <div className="card" style={{padding:'24px'}}>
          <div style={{fontSize:'15px',fontWeight:600,color:'#fff',marginBottom:'16px'}}>Document AI Model Test</div>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'16px',marginBottom:'16px'}}>
            <div>
              <label style={labelStyle}>Model</label>
              <select className="select" style={{width:'100%'}} value={selectedModel || ''} onChange={e=>{setSelectedModel(e.target.value);setVisionResult(null);setVisionError(null);}}>
                {visionModels.map(m => <option key={m.id} value={m.id}>{m.name} ({m.type})</option>)}
              </select>
            </div>
            {models.find(m => m.id === selectedModel)?.type === 'ocr' && (
              <div>
                <label style={labelStyle}>Merge Level</label>
                <select className="select" style={{width:'100%'}} value={ocrMerge} onChange={e=>setOcrMerge(e.target.value)}>
                  <option value="word">Word</option>
                  <option value="sentence">Sentence</option>
                  <option value="paragraph">Paragraph</option>
                </select>
                <div style={{fontSize:'10px',color:'var(--nv-text-dim)',marginTop:'4px'}}>Controls how recognized text regions are grouped together.</div>
              </div>
            )}
            {models.find(m => m.id === selectedModel)?.type === 'object-detection' && (
              <div>
                <label style={labelStyle}>Score Threshold</label>
                <div style={{display:'flex',alignItems:'center',gap:'10px'}}>
                  <input type="range" min="0" max="1" step="0.05" value={detectThreshold}
                    onChange={e=>setDetectThreshold(parseFloat(e.target.value))}
                    style={{flex:1,accentColor:'var(--nv-green)'}} />
                  <span className="mono" style={{fontSize:'13px',color:'var(--nv-green)',fontWeight:600,minWidth:'40px',textAlign:'right'}}>{(detectThreshold*100).toFixed(0)}%</span>
                </div>
                <div style={{fontSize:'10px',color:'var(--nv-text-dim)',marginTop:'4px'}}>Only detections above this confidence score are shown.</div>
              </div>
            )}
          </div>
          <div style={{marginBottom:'16px'}}>
            <label style={labelStyle}>Upload Document Image</label>
            <div style={{border:'2px dashed var(--nv-border)',borderRadius:'8px',padding:'24px',textAlign:'center',cursor:'pointer',
              background:imagePreview?'transparent':'rgba(255,255,255,0.02)',position:'relative',overflow:'hidden'}}
              onClick={()=>document.getElementById('model-img-input').click()}>
              <input id="model-img-input" type="file" accept="image/*,.pdf" style={{display:'none'}} onChange={handleImageUpload} />
              {imagePreview ? (
                <div>
                  <img src={imagePreview} alt="uploaded" style={{maxHeight:'300px',maxWidth:'100%',borderRadius:'6px',marginBottom:'8px'}} />
                  <div style={{fontSize:'11px',color:'var(--nv-text-dim)'}}>Click to replace image</div>
                </div>
              ) : (
                <div>
                  <div style={{fontSize:'28px',marginBottom:'6px',opacity:0.3}}><IconUpload /></div>
                  <div style={{fontSize:'13px',color:'var(--nv-text-muted)'}}>Click or drag to upload a document image (PNG, JPEG)</div>
                </div>
              )}
            </div>
          </div>
          <button className="btn btn-primary" onClick={runVision} disabled={visionLoading || !imageB64 || !selectedModel}>
            {visionLoading ? <><span className="spinner" style={{marginRight:'8px'}}></span>Processing…</> : 'Run Model'}
          </button>
          {visionError && <div style={{marginTop:'16px',padding:'12px 16px',borderRadius:'6px',background:'rgba(255,80,80,0.08)',border:'1px solid rgba(255,80,80,0.2)',color:'#ff5050',fontSize:'13px'}}>{visionError}</div>}
          {visionResult && (
            <div style={{marginTop:'20px'}}>
              <div style={{display:'flex',gap:'16px',flexWrap:'wrap',marginBottom:'16px'}}>
                {[
                  {label:'Model',value:visionResult.model_id},
                  {label:'Load Time',value:`${visionResult.model_load_ms}ms`},
                  {label:'Inference',value:`${visionResult.inference_ms}ms`},
                  visionResult.detection_count != null ? {label:'Detections',value:visionResult.detection_count} : null,
                  visionResult.image_size ? {label:'Image',value:`${visionResult.image_size[0]}×${visionResult.image_size[1]}`} : null,
                ].filter(Boolean).map(s => (
                  <div key={s.label} style={{background:'rgba(100,180,255,0.06)',border:'1px solid rgba(100,180,255,0.12)',borderRadius:'6px',padding:'8px 14px',minWidth:'100px'}}>
                    <div style={{fontSize:'10px',color:'var(--nv-text-dim)',textTransform:'uppercase',fontWeight:600,marginBottom:'2px'}}>{s.label}</div>
                    <div className="mono" style={{fontSize:'13px',color:'#64b4ff',fontWeight:600}}>{s.value}</div>
                  </div>
                ))}
              </div>

              {/* Annotated image for detection models */}
              {visionResult.annotated_image && (
                <div style={{marginBottom:'20px'}}>
                  <div style={{fontSize:'13px',fontWeight:600,color:'#fff',marginBottom:'8px'}}>Annotated Image</div>
                  <div style={{background:'var(--nv-bg)',border:'1px solid var(--nv-border)',borderRadius:'8px',padding:'12px',textAlign:'center',overflow:'auto'}}>
                    <img src={visionResult.annotated_image} alt="annotated detections"
                      style={{maxWidth:'100%',borderRadius:'4px',cursor:'pointer'}}
                      onClick={(e) => {
                        const w = window.open();
                        if (w) { w.document.write(`<img src="${visionResult.annotated_image}" style="max-width:100%">`); w.document.title = "Detection Result"; }
                      }}
                      title="Click to open full-size in new tab" />
                  </div>
                </div>
              )}

              {/* Detections table */}
              {visionResult.detections && visionResult.detections.length > 0 && (
                <div>
                  <div style={{fontSize:'13px',fontWeight:600,color:'#fff',marginBottom:'8px'}}>Detections ({visionResult.detections.length})</div>
                  <div style={{overflowX:'auto'}}>
                    <table className="runs-table" style={{fontSize:'12px'}}>
                      <thead><tr><th>#</th><th>Label</th><th>Score</th><th>Box (px)</th></tr></thead>
                      <tbody>
                        {visionResult.detections.map((d, i) => {
                          const DET_COLORS = {'table':'#76b900','chart':'#64b4ff','title':'#ff8c00','infographic':'#bb86fc',
                            'text':'#ff5050','header_footer':'#00d4aa','cell':'#76b900','merged_cell':'#fcd34d','row':'#64b4ff','column':'#ff8c00',
                            'chart_title':'#76b900','x_axis_title':'#64b4ff','y_axis_title':'#ff8c00','legend_title':'#bb86fc','legend_label':'#fcd34d',
                            'marker_label':'#ff5050','value_label':'#00d4aa','x_tick_label':'#ff69b4','y_tick_label':'#00bfff','other_label':'#90ee90'};
                          const color = DET_COLORS[d.label] || '#aaa';
                          return (
                            <tr key={i}>
                              <td>{i+1}</td>
                              <td>
                                <span style={{display:'inline-flex',alignItems:'center',gap:'6px'}}>
                                  <span style={{width:'10px',height:'10px',borderRadius:'2px',background:color,flexShrink:0}}></span>
                                  <span style={{fontWeight:600}}>{d.label}</span>
                                </span>
                              </td>
                              <td>
                                <div style={{display:'flex',alignItems:'center',gap:'8px'}}>
                                  <div style={{width:'50px',height:'6px',borderRadius:'3px',background:'rgba(255,255,255,0.06)',overflow:'hidden'}}>
                                    <div style={{height:'100%',width:`${d.score*100}%`,borderRadius:'3px',background:color}} />
                                  </div>
                                  <span className="mono" style={{color,fontWeight:600}}>{(d.score*100).toFixed(1)}%</span>
                                </div>
                              </td>
                              <td className="mono" style={{fontSize:'11px',color:'var(--nv-text-dim)'}}>
                                [{d.box.map(v=>Math.round(v)).join(', ')}]
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* OCR text */}
              {visionResult.text && (
                <div>
                  <div style={{fontSize:'13px',fontWeight:600,color:'#fff',marginBottom:'8px'}}>Extracted Text</div>
                  <pre style={{background:'var(--nv-bg)',border:'1px solid var(--nv-border)',borderRadius:'8px',padding:'16px',
                    fontSize:'12px',color:'var(--nv-text-muted)',whiteSpace:'pre-wrap',wordBreak:'break-word',maxHeight:'400px',overflow:'auto',lineHeight:'1.6'}}>
                    {visionResult.text}
                  </pre>
                </div>
              )}

              {/* Parse markdown */}
              {visionResult.markdown && (
                <div>
                  <div style={{fontSize:'13px',fontWeight:600,color:'#fff',marginBottom:'8px'}}>Parsed Output (Markdown)</div>
                  <pre style={{background:'var(--nv-bg)',border:'1px solid var(--nv-border)',borderRadius:'8px',padding:'16px',
                    fontSize:'12px',color:'var(--nv-text-muted)',whiteSpace:'pre-wrap',wordBreak:'break-word',maxHeight:'500px',overflow:'auto',lineHeight:'1.6'}}>
                    {visionResult.markdown}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </>
  );
}
