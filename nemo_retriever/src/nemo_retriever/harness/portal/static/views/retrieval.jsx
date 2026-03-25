/* ===== Retrieval Playground View ===== */
function RetrievalView({ runs }) {
  const [selectedRunId, setSelectedRunId] = useState("");
  const [lanceInfo, setLanceInfo] = useState(null);
  const [lanceLoading, setLanceLoading] = useState(false);
  const [queryText, setQueryText] = useState("");
  const [topK, setTopK] = useState(10);
  const [results, setResults] = useState(null);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState("");
  const [expandedRow, setExpandedRow] = useState(null);

  const successRuns = useMemo(() => (runs || []).filter(r => r.success === 1), [runs]);

  useEffect(() => {
    if (!selectedRunId) { setLanceInfo(null); setResults(null); return; }
    setLanceLoading(true);
    setLanceInfo(null);
    setResults(null);
    setError("");
    fetch(`/api/runs/${selectedRunId}/lancedb-info`)
      .then(r => r.json())
      .then(setLanceInfo)
      .catch(e => setError(e.message))
      .finally(() => setLanceLoading(false));
  }, [selectedRunId]);

  async function handleSearch(e) {
    e && e.preventDefault();
    if (!queryText.trim() || !selectedRunId) return;
    setSearching(true);
    setError("");
    setExpandedRow(null);
    try {
      const res = await fetch(`/api/runs/${selectedRunId}/retrieval`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ query: queryText.trim(), top_k: topK }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      setResults(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setSearching(false);
    }
  }

  const labelStyle = {display:'block',fontSize:'11px',fontWeight:600,color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.04em',marginBottom:'6px'};

  return (
    <>
      {/* Run Selector */}
      <div className="card" style={{padding:'20px',marginBottom:'20px'}}>
        <div style={{display:'grid',gridTemplateColumns:'1fr auto',gap:'16px',alignItems:'end'}}>
          <div>
            <label style={labelStyle}>Select a Run</label>
            <select className="select" style={{width:'100%'}} value={selectedRunId}
              onChange={e => setSelectedRunId(e.target.value)}>
              <option value="">— Choose a successful run —</option>
              {successRuns.map(r => (
                <option key={r.id} value={r.id}>
                  Run #{r.id} — {r.dataset} ({r.preset || "default"}) — {r.hostname || "unknown"} — {fmtTs(r.timestamp)}
                </option>
              ))}
            </select>
          </div>
          {lanceLoading && <span className="spinner"></span>}
        </div>

        {lanceInfo && (
          <div style={{marginTop:'14px',display:'flex',gap:'16px',alignItems:'center',flexWrap:'wrap'}}>
            {lanceInfo.available ? (
              <>
                <span className="badge badge-pass" style={{fontSize:'11px'}}>LanceDB Available</span>
                <span style={{fontSize:'12px',color:'var(--nv-text-muted)'}}>
                  {lanceInfo.row_count.toLocaleString()} rows in <span className="mono">{lanceInfo.table}</span>
                </span>
                <span className="mono" style={{fontSize:'11px',color:'var(--nv-text-dim)',maxWidth:'400px',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}} title={lanceInfo.uri}>
                  {lanceInfo.uri}
                </span>
              </>
            ) : (
              <>
                <span className="badge badge-fail" style={{fontSize:'11px'}}>LanceDB Not Available</span>
                {lanceInfo.error && <span style={{fontSize:'12px',color:'#ff5050'}}>{lanceInfo.error}</span>}
              </>
            )}
          </div>
        )}
      </div>

      {/* Query Input */}
      {lanceInfo && lanceInfo.available && (
        <div className="card" style={{padding:'20px',marginBottom:'20px'}}>
          <form onSubmit={handleSearch}>
            <label style={labelStyle}>Retrieval Query</label>
            <div style={{display:'flex',gap:'12px',alignItems:'stretch'}}>
              <input className="input" style={{flex:1,fontSize:'14px'}} value={queryText}
                onChange={e => setQueryText(e.target.value)}
                placeholder="Enter a natural language query to search the ingested documents…"
                disabled={searching} />
              <div style={{display:'flex',flexDirection:'column',gap:'2px',minWidth:'80px'}}>
                <label style={{fontSize:'10px',color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.04em'}}>Top K</label>
                <input type="number" className="input" min="1" max="100" value={topK}
                  onChange={e => setTopK(Math.max(1, Math.min(100, parseInt(e.target.value) || 10)))}
                  style={{width:'80px',textAlign:'center'}} />
              </div>
              <button type="submit" className="btn btn-primary" disabled={searching || !queryText.trim()}
                style={{height:'auto',minHeight:'42px',paddingLeft:'20px',paddingRight:'20px'}}>
                {searching ? <><span className="spinner" style={{marginRight:'8px'}}></span>Searching…</> : <><IconSearch /> Search</>}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{marginBottom:'16px',padding:'12px 16px',borderRadius:'8px',background:'rgba(255,50,50,0.08)',border:'1px solid rgba(255,50,50,0.2)',color:'#ff5050',fontSize:'13px'}}>
          {error}
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="card">
          <div style={{padding:'16px 20px',borderBottom:'1px solid var(--nv-border)',display:'flex',justifyContent:'space-between',alignItems:'center',flexWrap:'wrap',gap:'8px'}}>
            <div style={{display:'flex',alignItems:'center',gap:'12px'}}>
              <span style={{fontSize:'15px',fontWeight:600,color:'#fff'}}>{results.result_count} result{results.result_count !== 1 ? 's' : ''}</span>
              <span style={{fontSize:'12px',color:'var(--nv-text-dim)'}}>for "{results.query}"</span>
            </div>
            <div style={{display:'flex',gap:'12px',alignItems:'center',fontSize:'12px',color:'var(--nv-text-dim)'}}>
              <span>Top K: {results.top_k}</span>
              <span className="mono">{results.embed_model}</span>
            </div>
          </div>
          {results.results.length === 0 ? (
            <div style={{padding:'40px',textAlign:'center',color:'var(--nv-text-muted)',fontSize:'14px'}}>
              No results found for this query.
            </div>
          ) : (
            <div style={{overflowX:'auto'}}>
              <table className="runs-table">
                <thead>
                  <tr>
                    <th style={{width:'40px'}}>#</th>
                    <th>Source</th>
                    <th style={{width:'60px'}}>Page</th>
                    <th style={{width:'90px',textAlign:'right'}}>Distance</th>
                    <th>Text Preview</th>
                  </tr>
                </thead>
                <tbody>
                  {results.results.map((hit, idx) => {
                    const isExpanded = expandedRow === idx;
                    return React.createElement(React.Fragment, {key: idx},
                      React.createElement("tr", {
                        style: {cursor: 'pointer'},
                        onClick: () => setExpandedRow(isExpanded ? null : idx),
                      },
                        React.createElement("td", {style:{color:'var(--nv-text-dim)',fontSize:'12px',fontWeight:600}}, idx + 1),
                        React.createElement("td", {className:"mono", style:{fontSize:'12px',color:'var(--nv-text-muted)',maxWidth:'200px',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}, title:hit.source||''}, hit.source || "\u2014"),
                        React.createElement("td", {style:{textAlign:'center',fontSize:'12px',color:'#fff',fontWeight:500}}, hit.page_number != null ? hit.page_number : "\u2014"),
                        React.createElement("td", {style:{textAlign:'right',fontFamily:"'JetBrains Mono', monospace",fontSize:'12px',color: typeof hit._distance === 'number' ? (hit._distance < 0.5 ? 'var(--nv-green)' : hit._distance < 1.0 ? '#ffb84d' : '#ff5050') : 'var(--nv-text-muted)'}},
                          typeof hit._distance === 'number' ? hit._distance.toFixed(4) : "\u2014"
                        ),
                        React.createElement("td", {style:{fontSize:'12px',color:'var(--nv-text-muted)',maxWidth:'400px',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}, title: hit.text||''},
                          hit.text ? (hit.text.length > 120 ? hit.text.slice(0, 120) + "\u2026" : hit.text) : "\u2014"
                        ),
                      ),
                      isExpanded && React.createElement("tr", null,
                        React.createElement("td", {colSpan: 5, style:{padding:0,background:'var(--nv-bg)'}},
                          React.createElement("div", {style:{padding:'16px 20px',borderTop:'1px solid var(--nv-border)',borderBottom:'1px solid var(--nv-border)'}},
                            React.createElement("div", {style:{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'16px',marginBottom:'16px'}},
                              React.createElement("div", null,
                                React.createElement("div", {style:{fontSize:'10px',fontWeight:600,color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.05em',marginBottom:'4px'}}, "Source"),
                                React.createElement("div", {className:"mono",style:{fontSize:'12px',color:'#fff',wordBreak:'break-all'}}, hit.source || "\u2014"),
                              ),
                              React.createElement("div", {style:{display:'flex',gap:'24px'}},
                                React.createElement("div", null,
                                  React.createElement("div", {style:{fontSize:'10px',fontWeight:600,color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.05em',marginBottom:'4px'}}, "Page"),
                                  React.createElement("div", {style:{fontSize:'13px',color:'#fff',fontWeight:600}}, hit.page_number != null ? hit.page_number : "\u2014"),
                                ),
                                React.createElement("div", null,
                                  React.createElement("div", {style:{fontSize:'10px',fontWeight:600,color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.05em',marginBottom:'4px'}}, "Distance"),
                                  React.createElement("div", {className:"mono",style:{fontSize:'13px',color:'var(--nv-green)',fontWeight:600}}, typeof hit._distance === 'number' ? hit._distance.toFixed(6) : "\u2014"),
                                ),
                              ),
                            ),
                            React.createElement("div", null,
                              React.createElement("div", {style:{fontSize:'10px',fontWeight:600,color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.05em',marginBottom:'6px'}}, "Full Text"),
                              React.createElement("div", {style:{
                                fontSize:'13px',color:'#fff',lineHeight:'1.7',fontFamily:"'Inter', sans-serif",
                                padding:'14px',borderRadius:'8px',background:'rgba(255,255,255,0.03)',border:'1px solid var(--nv-border)',
                                maxHeight:'300px',overflowY:'auto',whiteSpace:'pre-wrap',wordBreak:'break-word',
                              }}, hit.text || "\u2014"),
                            ),
                            hit.metadata && typeof hit.metadata === 'object' && Object.keys(hit.metadata).length > 0 && (
                              React.createElement("div", {style:{marginTop:'14px'}},
                                React.createElement("div", {style:{fontSize:'10px',fontWeight:600,color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.05em',marginBottom:'6px'}}, "Metadata"),
                                React.createElement("div", {style:{display:'grid',gridTemplateColumns:'repeat(auto-fill, minmax(200px, 1fr))',gap:'6px'}},
                                  Object.entries(hit.metadata).map(([k, v]) =>
                                    React.createElement("div", {key: k, style:{fontSize:'11px'}},
                                      React.createElement("span", {style:{color:'var(--nv-text-dim)'}}, k + ": "),
                                      React.createElement("span", {className:"mono",style:{color:'var(--nv-text-muted)'}}, String(v)),
                                    )
                                  )
                                ),
                              )
                            ),
                          ),
                        ),
                      ),
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {!selectedRunId && (
        <div className="card" style={{padding:'60px 40px',textAlign:'center'}}>
          <div style={{fontSize:'48px',marginBottom:'16px',opacity:0.3}}>&#128269;</div>
          <div style={{fontSize:'16px',fontWeight:600,color:'#fff',marginBottom:'8px'}}>Retrieval Playground</div>
          <div style={{fontSize:'13px',color:'var(--nv-text-muted)',maxWidth:'460px',margin:'0 auto',lineHeight:'1.6'}}>
            Select a successful run above to load its LanceDB database, then enter natural language queries
            to test retrieval against the ingested documents.
          </div>
        </div>
      )}
    </>
  );
}
