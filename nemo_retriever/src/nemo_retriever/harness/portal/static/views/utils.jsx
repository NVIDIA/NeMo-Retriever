/* ===== Utilities ===== */
function fmt(val, decimals = 2) {
  if (val === null || val === undefined) return "\u2014";
  return typeof val === "number" ? val.toFixed(decimals) : String(val);
}

function fmtTs(ts) {
  if (!ts) return "\u2014";
  if (/^\d{8}_\d{6}_UTC$/.test(ts)) {
    const y = ts.slice(0,4), mo = ts.slice(4,6), d = ts.slice(6,8);
    const h = ts.slice(9,11), mi = ts.slice(11,13), s = ts.slice(13,15);
    return `${y}-${mo}-${d} ${h}:${mi}:${s} UTC`;
  }
  try { return new Date(ts).toLocaleString(); } catch { return ts; }
}

/* ===== Commit Link ===== */
function CommitLink({ sha, repoUrl, truncate = 10, style = {} }) {
  if (!sha) return <span style={style}>{"\u2014"}</span>;
  const display = truncate ? sha.slice(0, truncate) : sha;
  if (repoUrl) {
    const href = `${repoUrl}/commit/${sha}`;
    return (
      <a href={href} target="_blank" rel="noopener noreferrer" className="mono"
        title={`View commit ${sha} on GitHub`}
        style={{fontSize:'12px',color:'var(--nv-green)',textDecoration:'none',borderBottom:'1px dashed var(--nv-green)',cursor:'pointer',...style}}
        onClick={e => e.stopPropagation()}>
        {display}
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
          style={{width:'10px',height:'10px',marginLeft:'3px',verticalAlign:'middle',opacity:0.6}}>
          <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/>
        </svg>
      </a>
    );
  }
  return <span className="mono" style={{fontSize:'12px',color:'var(--nv-green)',...style}}>{display}</span>;
}

/* ===== Pagination ===== */
const PAGE_SIZE_OPTIONS = [10, 25, 50, 100, 250];

function usePagination(data, defaultPageSize = 25) {
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(defaultPageSize);
  const totalPages = Math.max(1, Math.ceil(data.length / pageSize));
  const safePage = Math.min(page, totalPages - 1);
  useEffect(() => { if (page !== safePage) setPage(safePage); }, [page, safePage]);
  const pageData = data.slice(safePage * pageSize, safePage * pageSize + pageSize);
  return {
    page: safePage, setPage, pageSize, setPageSize: (s) => { setPageSize(s); setPage(0); },
    totalPages, pageData, totalItems: data.length,
  };
}

function Pagination({ page, totalPages, totalItems, pageSize, onPageChange, onPageSizeChange }) {
  if (totalItems === 0) return null;
  const start = page * pageSize + 1;
  const end = Math.min((page + 1) * pageSize, totalItems);
  return (
    <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',padding:'10px 16px',borderTop:'1px solid var(--nv-border)',fontSize:'12px',color:'var(--nv-text-muted)',flexWrap:'wrap',gap:'8px'}}>
      <div style={{display:'flex',alignItems:'center',gap:'8px'}}>
        <span>Rows per page:</span>
        <select className="select" value={pageSize} onChange={e=>onPageSizeChange(parseInt(e.target.value,10))}
          style={{padding:'3px 8px',fontSize:'12px',minWidth:'60px',borderRadius:'6px'}}>
          {PAGE_SIZE_OPTIONS.map(s=><option key={s} value={s}>{s}</option>)}
        </select>
      </div>
      <span>{start}–{end} of {totalItems}</span>
      <div style={{display:'flex',alignItems:'center',gap:'2px'}}>
        <button className="btn btn-ghost btn-sm btn-icon" disabled={page===0} onClick={()=>onPageChange(0)}
          style={{padding:'4px 6px',opacity:page===0?0.3:1}}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{width:'14px',height:'14px'}}>
            <polyline points="11 17 6 12 11 7"/><polyline points="18 17 13 12 18 7"/>
          </svg>
        </button>
        <button className="btn btn-ghost btn-sm btn-icon" disabled={page===0} onClick={()=>onPageChange(page-1)}
          style={{padding:'4px 6px',opacity:page===0?0.3:1}}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{width:'14px',height:'14px'}}>
            <polyline points="15 18 9 12 15 6"/>
          </svg>
        </button>
        <span style={{padding:'0 8px',fontSize:'12px',color:'var(--nv-text-muted)'}}>
          Page {page+1} of {totalPages}
        </span>
        <button className="btn btn-ghost btn-sm btn-icon" disabled={page>=totalPages-1} onClick={()=>onPageChange(page+1)}
          style={{padding:'4px 6px',opacity:page>=totalPages-1?0.3:1}}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{width:'14px',height:'14px'}}>
            <polyline points="9 18 15 12 9 6"/>
          </svg>
        </button>
        <button className="btn btn-ghost btn-sm btn-icon" disabled={page>=totalPages-1} onClick={()=>onPageChange(totalPages-1)}
          style={{padding:'4px 6px',opacity:page>=totalPages-1?0.3:1}}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{width:'14px',height:'14px'}}>
            <polyline points="13 17 18 12 13 7"/><polyline points="6 17 11 12 6 7"/>
          </svg>
        </button>
      </div>
    </div>
  );
}

/* ===== Badges ===== */
function StatusBadge({ success }) {
  if (success === 1) return <span className="badge badge-pass">{"\u25CF"} Pass</span>;
  if (success === 0) return <span className="badge badge-fail">{"\u25CF"} Fail</span>;
  return <span className="badge badge-na">{"\u25CF"} N/A</span>;
}

function JobStatusBadge({ status }) {
  const cls = { pending:"badge-pending", running:"badge-running", completed:"badge-pass", failed:"badge-fail", error:"badge-fail", cancelled:"badge-na", cancelling:"badge-cancel" };
  return <span className={`badge ${cls[status] || "badge-na"}`}>{status}</span>;
}

function TriggerSourceBadge({ source }) {
  if (!source) return <span className="badge badge-na">—</span>;
  const map = {
    manual: { cls: "badge-manual", label: "Manual" },
    scheduled: { cls: "badge-scheduled", label: "Scheduled" },
    github_push: { cls: "badge-github", label: "GitHub" },
    playground: { cls: "badge-manual", label: "Playground" },
  };
  const info = map[source] || { cls: "badge-na", label: source };
  return <span className={`badge ${info.cls}`}>{info.label}</span>;
}

function ScheduleTypeBadge({ type }) {
  if (type === "github_push") return <span className="badge badge-github-type"><IconGithub /> GitHub</span>;
  return <span className="badge badge-cron"><IconCalendar /> Cron</span>;
}
