/* ===== MCP View ===== */
function McpView() {
  const [portalSettings, setPortalSettings] = useState({});
  const [mcpTools, setMcpTools] = useState([]);
  const [mcpToolsLoading, setMcpToolsLoading] = useState(true);
  const [mcpAuditLog, setMcpAuditLog] = useState([]);
  const [mcpAuditStats, setMcpAuditStats] = useState(null);
  const [mcpAuditLoading, setMcpAuditLoading] = useState(true);
  const [mcpAuditFilter, setMcpAuditFilter] = useState({ tool_name: '', agent_name: '' });
  const [mcpCursorConfig, setMcpCursorConfig] = useState(null);
  const [mcpToolFilter, setMcpToolFilter] = useState('');

  async function fetchPortalSettings() {
    try {
      const res = await fetch("/api/portal-settings");
      setPortalSettings(await res.json());
    } catch (err) { console.error("Failed to fetch portal settings:", err); }
  }

  async function fetchMcpTools() {
    setMcpToolsLoading(true);
    try {
      const res = await fetch("/api/mcp/tools");
      setMcpTools(await res.json());
    } catch (err) { console.error("Failed to fetch MCP tools:", err); }
    finally { setMcpToolsLoading(false); }
  }

  async function fetchMcpAuditLog() {
    setMcpAuditLoading(true);
    try {
      const params = new URLSearchParams({ limit: '100' });
      if (mcpAuditFilter.tool_name) params.set('tool_name', mcpAuditFilter.tool_name);
      if (mcpAuditFilter.agent_name) params.set('agent_name', mcpAuditFilter.agent_name);
      const [logRes, statsRes] = await Promise.all([
        fetch(`/api/mcp/audit-log?${params}`),
        fetch("/api/mcp/audit-log/stats"),
      ]);
      setMcpAuditLog(await logRes.json());
      setMcpAuditStats(await statsRes.json());
    } catch (err) { console.error("Failed to fetch MCP audit log:", err); }
    finally { setMcpAuditLoading(false); }
  }

  async function fetchMcpCursorConfig() {
    try {
      const res = await fetch("/api/mcp/cursor-config");
      setMcpCursorConfig(await res.json());
    } catch (err) { console.error("Failed to fetch Cursor config:", err); }
  }

  async function toggleMcpTool(toolKey, enabled) {
    try {
      await fetch(`/api/mcp/tools/${encodeURIComponent(toolKey)}/toggle`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      fetchMcpTools();
    } catch (err) { console.error("Failed to toggle tool:", err); }
  }

  async function saveMcpSetting(key, value) {
    try {
      await fetch("/api/portal-settings", {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [key]: value }),
      });
      fetchPortalSettings();
    } catch (err) { console.error("Failed to save MCP setting:", err); }
  }

  useEffect(() => { fetchPortalSettings(); fetchMcpTools(); fetchMcpAuditLog(); fetchMcpCursorConfig(); }, []);

  const labelStyle = {display:'block',fontSize:'11px',fontWeight:600,color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.04em',marginBottom:'6px'};

  return (
    <>
      {/* ===== MCP Server Configuration ===== */}
      <div className="card" style={{padding:'24px',marginBottom:'20px'}}>
        <div className="section-title" style={{marginBottom:'6px'}}>MCP Server</div>
        <div style={{fontSize:'12px',color:'var(--nv-text-dim)',lineHeight:'1.6',marginBottom:'16px'}}>
          The MCP (Model Context Protocol) server allows AI agents like Cursor, Claude Desktop, and custom tools to interact with this portal programmatically.
        </div>

        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'16px',marginBottom:'16px'}}>
          <div>
            <label style={labelStyle}>Server Status</label>
            <div style={{display:'flex',alignItems:'center',gap:'8px'}}>
              <span style={{width:'8px',height:'8px',borderRadius:'50%',background: portalSettings.mcp_enabled === 'true' ? 'var(--nv-green)' : '#ff5050'}}></span>
              <span style={{fontSize:'13px',fontWeight:600,color: portalSettings.mcp_enabled === 'true' ? 'var(--nv-green)' : '#ff5050'}}>
                {portalSettings.mcp_enabled === 'true' ? 'Enabled' : 'Disabled'}
              </span>
              <button className="btn btn-sm" style={{marginLeft:'8px',fontSize:'10px',padding:'2px 10px'}}
                onClick={() => saveMcpSetting('mcp_enabled', portalSettings.mcp_enabled === 'true' ? 'false' : 'true')}>
                {portalSettings.mcp_enabled === 'true' ? 'Disable' : 'Enable'}
              </button>
            </div>
          </div>
          <div>
            <label style={labelStyle}>Rate Limit (req/min per agent)</label>
            <div style={{display:'flex',gap:'8px',alignItems:'center'}}>
              <input className="input" style={{width:'80px'}} type="number"
                value={portalSettings.mcp_rate_limit || '60'}
                onChange={e => saveMcpSetting('mcp_rate_limit', e.target.value)} />
              <span style={{fontSize:'11px',color:'var(--nv-text-dim)'}}>requests/minute</span>
            </div>
          </div>
        </div>

        <div style={{marginBottom:'16px'}}>
          <label style={labelStyle}>Allowed Origins (CORS)</label>
          <input className="input" style={{width:'100%'}}
            value={portalSettings.mcp_allowed_origins || '*'}
            onBlur={e => saveMcpSetting('mcp_allowed_origins', e.target.value)}
            onChange={e => setPortalSettings({...portalSettings, mcp_allowed_origins: e.target.value})}
            placeholder="* (allow all)" />
          <div style={{fontSize:'10px',color:'var(--nv-text-dim)',marginTop:'4px'}}>Comma-separated origins, or * for all</div>
        </div>

        {mcpCursorConfig && (
          <div style={{marginBottom:'0'}}>
            <label style={labelStyle}>Cursor IDE Configuration</label>
            <div style={{fontSize:'11px',color:'var(--nv-text-dim)',marginBottom:'8px'}}>
              Add this to your <code className="mono" style={{background:'rgba(255,255,255,0.06)',padding:'1px 5px',borderRadius:'4px'}}>.cursor/mcp.json</code> to connect Cursor to this portal:
            </div>
            <div style={{position:'relative'}}>
              <pre className="mono" style={{
                fontSize:'11px',padding:'12px',borderRadius:'8px',
                background:'var(--nv-bg)',border:'1px solid var(--nv-border)',
                color:'var(--nv-green)',margin:0,overflow:'auto',
              }}>{JSON.stringify(mcpCursorConfig, null, 2)}</pre>
              <button className="btn btn-sm" style={{position:'absolute',top:'6px',right:'6px',fontSize:'10px',padding:'2px 8px'}}
                onClick={() => { navigator.clipboard.writeText(JSON.stringify(mcpCursorConfig, null, 2)); }}>
                Copy
              </button>
            </div>
          </div>
        )}
      </div>

      {/* ===== Exposed Tools ===== */}
      <div className="card" style={{padding:'24px',marginBottom:'20px'}}>
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'16px'}}>
          <div>
            <div className="section-title" style={{marginBottom:'4px'}}>Exposed Tools</div>
            <div style={{fontSize:'12px',color:'var(--nv-text-dim)'}}>
              {mcpTools.length} tool{mcpTools.length !== 1 ? 's' : ''} registered — {mcpTools.filter(t => t.enabled).length} enabled
            </div>
          </div>
          <div style={{display:'flex',gap:'8px',alignItems:'center'}}>
            <input className="input" style={{width:'200px',fontSize:'12px'}} placeholder="Filter tools…"
              value={mcpToolFilter} onChange={e => setMcpToolFilter(e.target.value)} />
            <button className="btn btn-sm" onClick={fetchMcpTools} style={{fontSize:'10px',padding:'4px 10px'}}>
              <IconRefresh />
            </button>
          </div>
        </div>

        {mcpToolsLoading ? (
          <div style={{padding:'40px',textAlign:'center'}}>
            <span className="spinner"></span>
          </div>
        ) : (
          <div style={{borderRadius:'8px',border:'1px solid var(--nv-border)',overflow:'hidden'}}>
            <table style={{width:'100%',borderCollapse:'collapse',fontSize:'12px'}}>
              <thead>
                <tr style={{background:'rgba(255,255,255,0.02)'}}>
                  <th style={{textAlign:'left',padding:'8px 12px',color:'var(--nv-text-dim)',fontWeight:600,fontSize:'10px',textTransform:'uppercase',letterSpacing:'0.04em'}}>Tool</th>
                  <th style={{textAlign:'left',padding:'8px 12px',color:'var(--nv-text-dim)',fontWeight:600,fontSize:'10px',textTransform:'uppercase',letterSpacing:'0.04em'}}>Category</th>
                  <th style={{textAlign:'left',padding:'8px 12px',color:'var(--nv-text-dim)',fontWeight:600,fontSize:'10px',textTransform:'uppercase',letterSpacing:'0.04em'}}>Description</th>
                  <th style={{textAlign:'center',padding:'8px 12px',color:'var(--nv-text-dim)',fontWeight:600,fontSize:'10px',textTransform:'uppercase',letterSpacing:'0.04em',width:'80px'}}>Status</th>
                </tr>
              </thead>
              <tbody>
                {mcpTools
                  .filter(t => !mcpToolFilter || t.name.toLowerCase().includes(mcpToolFilter.toLowerCase())
                    || t.category.toLowerCase().includes(mcpToolFilter.toLowerCase())
                    || t.description.toLowerCase().includes(mcpToolFilter.toLowerCase()))
                  .map(tool => (
                  <tr key={tool.key} style={{borderTop:'1px solid var(--nv-border)'}}>
                    <td style={{padding:'8px 12px'}}>
                      <span className="mono" style={{color:'#fff',fontWeight:600,fontSize:'12px'}}>{tool.name}</span>
                      {tool.tags && tool.tags.length > 0 && (
                        <div style={{marginTop:'2px',display:'flex',gap:'4px'}}>
                          {tool.tags.map(tag => (
                            <span key={tag} style={{fontSize:'9px',padding:'1px 5px',borderRadius:'3px',
                              background:'rgba(100,180,255,0.1)',color:'#64b4ff',fontWeight:500}}>{tag}</span>
                          ))}
                        </div>
                      )}
                    </td>
                    <td style={{padding:'8px 12px',color:'var(--nv-text-muted)',fontSize:'11px'}}>{tool.category}</td>
                    <td style={{padding:'8px 12px',color:'var(--nv-text-dim)',fontSize:'11px',maxWidth:'300px'}}>{tool.description}</td>
                    <td style={{padding:'8px 12px',textAlign:'center'}}>
                      <button className="btn btn-sm" style={{
                        fontSize:'10px',padding:'2px 10px',minWidth:'60px',
                        background: tool.enabled ? 'rgba(118,185,0,0.1)' : 'rgba(255,50,50,0.08)',
                        color: tool.enabled ? 'var(--nv-green)' : '#ff5050',
                        border: `1px solid ${tool.enabled ? 'rgba(118,185,0,0.3)' : 'rgba(255,50,50,0.2)'}`,
                      }} onClick={() => toggleMcpTool(tool.key, !tool.enabled)}>
                        {tool.enabled ? 'Enabled' : 'Disabled'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ===== Agent Activity Log ===== */}
      <div className="card" style={{padding:'24px',marginBottom:'20px'}}>
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'16px'}}>
          <div>
            <div className="section-title" style={{marginBottom:'4px'}}>Agent Activity Log</div>
            <div style={{fontSize:'12px',color:'var(--nv-text-dim)'}}>MCP tool invocations from AI agents</div>
          </div>
          <button className="btn btn-sm" onClick={fetchMcpAuditLog} style={{fontSize:'10px',padding:'4px 10px'}}>
            <IconRefresh />
          </button>
        </div>

        {/* Stats Cards */}
        {mcpAuditStats && (
          <div style={{display:'grid',gridTemplateColumns:'repeat(4, 1fr)',gap:'12px',marginBottom:'16px'}}>
            <div style={{padding:'12px 16px',borderRadius:'8px',background:'rgba(255,255,255,0.02)',border:'1px solid var(--nv-border)'}}>
              <div style={{fontSize:'20px',fontWeight:700,color:'#fff'}}>{mcpAuditStats.total_requests}</div>
              <div style={{fontSize:'10px',color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.04em',marginTop:'2px'}}>Total Requests</div>
            </div>
            <div style={{padding:'12px 16px',borderRadius:'8px',background:'rgba(255,255,255,0.02)',border:'1px solid var(--nv-border)'}}>
              <div style={{fontSize:'20px',fontWeight:700,color:'var(--nv-green)'}}>{mcpAuditStats.unique_agents}</div>
              <div style={{fontSize:'10px',color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.04em',marginTop:'2px'}}>Unique Agents</div>
            </div>
            <div style={{padding:'12px 16px',borderRadius:'8px',background:'rgba(255,255,255,0.02)',border:'1px solid var(--nv-border)'}}>
              <div style={{fontSize:'20px',fontWeight:700,color:'var(--nv-green)'}}>{mcpAuditStats.success_count}</div>
              <div style={{fontSize:'10px',color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.04em',marginTop:'2px'}}>Successful</div>
            </div>
            <div style={{padding:'12px 16px',borderRadius:'8px',background:'rgba(255,255,255,0.02)',border:'1px solid var(--nv-border)'}}>
              <div style={{fontSize:'20px',fontWeight:700,color: mcpAuditStats.error_count > 0 ? '#ff5050' : 'var(--nv-text-muted)'}}>{mcpAuditStats.error_count}</div>
              <div style={{fontSize:'10px',color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.04em',marginTop:'2px'}}>Errors</div>
            </div>
          </div>
        )}

        {/* Top tools and agents */}
        {mcpAuditStats && (mcpAuditStats.top_tools?.length > 0 || mcpAuditStats.top_agents?.length > 0) && (
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'16px',marginBottom:'16px'}}>
            {mcpAuditStats.top_tools?.length > 0 && (
              <div style={{padding:'12px 16px',borderRadius:'8px',background:'rgba(255,255,255,0.02)',border:'1px solid var(--nv-border)'}}>
                <div style={{fontSize:'10px',fontWeight:600,color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.04em',marginBottom:'8px'}}>Most Used Tools</div>
                {mcpAuditStats.top_tools.map(t => (
                  <div key={t.tool_name} style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'3px 0'}}>
                    <span className="mono" style={{fontSize:'11px',color:'var(--nv-text-muted)'}}>{t.tool_name}</span>
                    <span style={{fontSize:'11px',fontWeight:600,color:'#fff'}}>{t.count}</span>
                  </div>
                ))}
              </div>
            )}
            {mcpAuditStats.top_agents?.length > 0 && (
              <div style={{padding:'12px 16px',borderRadius:'8px',background:'rgba(255,255,255,0.02)',border:'1px solid var(--nv-border)'}}>
                <div style={{fontSize:'10px',fontWeight:600,color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.04em',marginBottom:'8px'}}>Top Agents</div>
                {mcpAuditStats.top_agents.map(a => (
                  <div key={a.agent_name} style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'3px 0'}}>
                    <span style={{fontSize:'11px',color:'var(--nv-text-muted)'}}>{a.agent_name}</span>
                    <span style={{fontSize:'11px',fontWeight:600,color:'#fff'}}>{a.count}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Filters */}
        <div style={{display:'flex',gap:'10px',marginBottom:'12px'}}>
          <input className="input" style={{flex:1,fontSize:'12px'}} placeholder="Filter by tool name…"
            value={mcpAuditFilter.tool_name}
            onChange={e => setMcpAuditFilter({...mcpAuditFilter, tool_name: e.target.value})} />
          <input className="input" style={{flex:1,fontSize:'12px'}} placeholder="Filter by agent name…"
            value={mcpAuditFilter.agent_name}
            onChange={e => setMcpAuditFilter({...mcpAuditFilter, agent_name: e.target.value})} />
          <button className="btn btn-sm" onClick={fetchMcpAuditLog} style={{fontSize:'10px',padding:'4px 10px'}}>Apply</button>
        </div>

        {/* Log Table */}
        {mcpAuditLoading ? (
          <div style={{padding:'40px',textAlign:'center'}}>
            <span className="spinner"></span>
          </div>
        ) : mcpAuditLog.length === 0 ? (
          <div style={{padding:'40px',textAlign:'center',color:'var(--nv-text-dim)',fontSize:'13px'}}>
            No agent activity recorded yet. Agents will appear here once they start making MCP requests.
          </div>
        ) : (
          <div style={{borderRadius:'8px',border:'1px solid var(--nv-border)',overflow:'hidden',maxHeight:'400px',overflowY:'auto'}}>
            <table style={{width:'100%',borderCollapse:'collapse',fontSize:'11px'}}>
              <thead style={{position:'sticky',top:0,background:'var(--nv-surface)'}}>
                <tr>
                  <th style={{textAlign:'left',padding:'6px 10px',color:'var(--nv-text-dim)',fontWeight:600,fontSize:'9px',textTransform:'uppercase',letterSpacing:'0.04em'}}>Time</th>
                  <th style={{textAlign:'left',padding:'6px 10px',color:'var(--nv-text-dim)',fontWeight:600,fontSize:'9px',textTransform:'uppercase',letterSpacing:'0.04em'}}>Agent</th>
                  <th style={{textAlign:'left',padding:'6px 10px',color:'var(--nv-text-dim)',fontWeight:600,fontSize:'9px',textTransform:'uppercase',letterSpacing:'0.04em'}}>Tool</th>
                  <th style={{textAlign:'left',padding:'6px 10px',color:'var(--nv-text-dim)',fontWeight:600,fontSize:'9px',textTransform:'uppercase',letterSpacing:'0.04em'}}>Arguments</th>
                  <th style={{textAlign:'right',padding:'6px 10px',color:'var(--nv-text-dim)',fontWeight:600,fontSize:'9px',textTransform:'uppercase',letterSpacing:'0.04em',width:'60px'}}>ms</th>
                  <th style={{textAlign:'center',padding:'6px 10px',color:'var(--nv-text-dim)',fontWeight:600,fontSize:'9px',textTransform:'uppercase',letterSpacing:'0.04em',width:'50px'}}>OK</th>
                </tr>
              </thead>
              <tbody>
                {mcpAuditLog.map(entry => (
                  <tr key={entry.id} style={{borderTop:'1px solid var(--nv-border)'}}>
                    <td style={{padding:'5px 10px',color:'var(--nv-text-dim)',whiteSpace:'nowrap',fontSize:'10px'}}>
                      {entry.timestamp ? new Date(entry.timestamp).toLocaleString() : '—'}
                    </td>
                    <td style={{padding:'5px 10px',color:'var(--nv-text-muted)',fontSize:'11px'}}>{entry.agent_name || entry.agent_id || '—'}</td>
                    <td style={{padding:'5px 10px'}}>
                      <span className="mono" style={{color:'#fff',fontWeight:500,fontSize:'11px'}}>{entry.tool_name}</span>
                    </td>
                    <td style={{padding:'5px 10px',color:'var(--nv-text-dim)',maxWidth:'200px',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap',fontSize:'10px'}}>
                      {entry.arguments || '—'}
                    </td>
                    <td style={{padding:'5px 10px',textAlign:'right',color:'var(--nv-text-muted)',fontSize:'10px',fontFamily:'var(--font-mono)'}}>
                      {entry.duration_ms != null ? Math.round(entry.duration_ms) : '—'}
                    </td>
                    <td style={{padding:'5px 10px',textAlign:'center'}}>
                      <span style={{
                        display:'inline-block',width:'6px',height:'6px',borderRadius:'50%',
                        background: entry.success ? 'var(--nv-green)' : '#ff5050',
                      }}></span>
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
