/* ===== Sidebar ===== */
function Sidebar({ activeView, onNavigate, alertBadgeCount, githubRepoUrl }) {
  const [versionInfo, setVersionInfo] = useState(null);
  useEffect(() => {
    fetch('/api/version').then(r => r.json()).then(setVersionInfo).catch(() => {});
  }, []);
  return (
    <div className="sidebar">
      <div className="sidebar-logo">
        <img src="/static/nvidia-logo.svg" alt="NVIDIA" style={{height:'24px',width:'auto'}} />
      </div>
      <nav className="sidebar-nav">
        <div className="sidebar-section-label">Views</div>
        <div className={`nav-item ${activeView==='runs'?'active':''}`} onClick={()=>onNavigate('runs')}>
          <IconDatabase /><span>Runs</span>
        </div>
        <div className={`nav-item ${activeView==='analytics'?'active':''}`} onClick={()=>onNavigate('analytics')}>
          <IconChart /><span>Analytics</span>
        </div>
        <div className={`nav-item ${activeView==='reporting'?'active':''}`} onClick={()=>onNavigate('reporting')}>
          <IconFileText /><span>Reporting</span>
        </div>
        <div className={`nav-item ${activeView==='datasets'?'active':''}`} onClick={()=>onNavigate('datasets')}>
          <IconFolder /><span>Datasets</span>
        </div>
        <div className={`nav-item ${activeView==='presets'?'active':''}`} onClick={()=>onNavigate('presets')}>
          <IconSliders /><span>Presets</span>
        </div>
        <div className={`nav-item ${activeView==='runners'?'active':''}`} onClick={()=>onNavigate('runners')}>
          <IconServer /><span>Runners</span>
        </div>
        <div className={`nav-item ${activeView==='scheduling'?'active':''}`} onClick={()=>onNavigate('scheduling')}>
          <IconCalendar /><span>Scheduling</span>
        </div>
        <div className="sidebar-section-label">Monitoring</div>
        <div className={`nav-item ${activeView==='alerts'?'active':''}`} onClick={()=>onNavigate('alerts')}>
          <IconBell /><span>Alerts</span>
          {typeof alertBadgeCount === 'number' && alertBadgeCount > 0 && (
            <span style={{marginLeft:'auto',background:'#ff5050',color:'#fff',fontSize:'10px',fontWeight:700,padding:'1px 7px',borderRadius:'100px',lineHeight:'16px'}}>{alertBadgeCount}</span>
          )}
        </div>
        <div className="sidebar-section-label">Playground</div>
        <div className={`nav-item ${activeView==='ingestion'?'active':''}`} onClick={()=>onNavigate('ingestion')}>
          <IconUpload /><span>Ingestion</span>
        </div>
        <div className={`nav-item ${activeView==='retrieval'?'active':''}`} onClick={()=>onNavigate('retrieval')}>
          <IconSearch /><span>Retrieval</span>
        </div>
        <div className={`nav-item ${activeView==='models'?'active':''}`} onClick={()=>onNavigate('models')}>
          <IconCpu /><span>Models</span>
        </div>
        <div className={`nav-item ${activeView==='designer'?'active':''}`} onClick={()=>onNavigate('designer')}>
          <IconSliders /><span>Designer</span>
        </div>
        <div className="sidebar-section-label">System</div>
        <div className={`nav-item ${activeView==='settings'?'active':''}`} onClick={()=>onNavigate('settings')}>
          <IconSettings /><span>Settings</span>
        </div>
        <div className={`nav-item ${activeView==='database'?'active':''}`} onClick={()=>onNavigate('database')}>
          <IconHardDrive /><span>Database</span>
        </div>
        <div className={`nav-item ${activeView==='mcp'?'active':''}`} onClick={()=>onNavigate('mcp')}>
          <IconLink /><span>MCP</span>
        </div>
      </nav>
      <div className="sidebar-bottom">
        <div style={{fontSize:'11px',color:'var(--nv-text-dim)',marginBottom:'4px'}}>nemo_retriever harness</div>
        {versionInfo && (
          <div style={{fontSize:'10px',color:'var(--nv-text-dim)',lineHeight:'1.5',fontFamily:'var(--font-mono)'}}>
            <span style={{color:'var(--nv-green)',fontWeight:600}}>{versionInfo.version}</span>
            {versionInfo.git_sha && versionInfo.git_sha !== 'unknown' && (
              githubRepoUrl ? (
                <a href={`${githubRepoUrl}/commit/${versionInfo.git_sha}`} target="_blank" rel="noopener noreferrer"
                  style={{marginLeft:'6px',opacity:0.7,color:'var(--nv-green)',textDecoration:'none',borderBottom:'1px dashed var(--nv-green)'}}
                  title={`View commit ${versionInfo.git_sha} on GitHub`}
                  onClick={e=>e.stopPropagation()}>({versionInfo.git_sha})</a>
              ) : (
                <span style={{marginLeft:'6px',opacity:0.6}} title={`Git SHA: ${versionInfo.git_sha}`}>({versionInfo.git_sha})</span>
              )
            )}
            {versionInfo.build_date && versionInfo.build_date !== 'unknown' && (
              <div style={{opacity:0.5,marginTop:'1px'}}>{versionInfo.build_date}</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* ===== Header ===== */
function Header({ title, children }) {
  return (
    <div className="header">
      <h2 style={{fontSize:'16px',fontWeight:600,color:'#fff'}}>{title}</h2>
      <div style={{display:'flex',alignItems:'center',gap:'8px'}}>{children}</div>
    </div>
  );
}

/* ===== Footer ===== */
function Footer({ children }) {
  return <div className="footer">{children}</div>;
}
