/* ===== Analytics View ===== */
const ANALYTICS_METRICS = [
  { key: "pages_per_sec", label: "Pages / sec" },
  { key: "recall_1", label: "Recall@1" },
  { key: "recall_5", label: "Recall@5" },
  { key: "recall_10", label: "Recall@10" },
  { key: "ingest_secs", label: "Ingest Time (s)" },
  { key: "pages", label: "Pages" },
  { key: "files", label: "Files" },
];

const CHART_COLORS = [
  '#76b900','#64b4ff','#a882ff','#fcd34d','#ff5050',
  '#ff8c00','#00d4aa','#ff69b4','#4ecdc4','#95e1d3',
  '#f38181','#aa96da','#45b7d1','#96ceb4','#dfe6e9',
];

function parseRunTimestamp(ts) {
  if (!ts) return null;
  if (/^\d{8}_\d{6}_UTC$/.test(ts)) {
    return new Date(Date.UTC(+ts.slice(0,4), +ts.slice(4,6)-1, +ts.slice(6,8), +ts.slice(9,11), +ts.slice(11,13), +ts.slice(13,15)));
  }
  try { return new Date(ts); } catch { return null; }
}

function AnalyticsView({ runs, datasets, loading, onRefresh }) {
  const canvasRef = React.useRef(null);
  const chartRef = React.useRef(null);

  const [chartType, setChartType] = useState("line");
  const [yMetric, setYMetric] = useState("pages_per_sec");
  const [xMetric, setXMetric] = useState("pages");
  const [groupBy, setGroupBy] = useState("dataset");
  const [filterDataset, setFilterDataset] = useState("");
  const [filterPreset, setFilterPreset] = useState("");
  const [filterStatus, setFilterStatus] = useState("all");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");

  const allDatasets = useMemo(() => [...new Set(runs.map(r => r.dataset).filter(Boolean))].sort(), [runs]);
  const allPresets = useMemo(() => [...new Set(runs.map(r => r.preset).filter(Boolean))].sort(), [runs]);

  const filteredRuns = useMemo(() => {
    return runs.filter(r => {
      if (filterDataset && r.dataset !== filterDataset) return false;
      if (filterPreset && r.preset !== filterPreset) return false;
      if (filterStatus === "pass" && r.success !== 1) return false;
      if (filterStatus === "fail" && r.success !== 0) return false;
      if (dateFrom || dateTo) {
        const d = parseRunTimestamp(r.timestamp);
        if (!d) return false;
        if (dateFrom && d < new Date(dateFrom + "T00:00:00Z")) return false;
        if (dateTo && d > new Date(dateTo + "T23:59:59Z")) return false;
      }
      return true;
    });
  }, [runs, filterDataset, filterPreset, filterStatus, dateFrom, dateTo]);

  useEffect(() => {
    if (!canvasRef.current) return;
    if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; }
    if (filteredRuns.length === 0) return;

    const yKey = yMetric;

    if (chartType === "line" || chartType === "scatter") {
      const groups = {};
      filteredRuns.forEach(r => {
        const group = r[groupBy] || "unknown";
        if (!groups[group]) groups[group] = [];
        const yVal = r[yKey];
        if (yVal == null) return;
        if (chartType === "line") {
          const date = parseRunTimestamp(r.timestamp);
          if (date) groups[group].push({ x: date, y: yVal, _run: r });
        } else {
          const xVal = r[xMetric];
          if (xVal != null) groups[group].push({ x: xVal, y: yVal, _run: r });
        }
      });

      const chartDatasets = Object.entries(groups).map(([name, points], i) => ({
        label: name,
        data: points.sort((a, b) => (a.x instanceof Date ? a.x.getTime() - b.x.getTime() : a.x - b.x)),
        borderColor: CHART_COLORS[i % CHART_COLORS.length],
        backgroundColor: CHART_COLORS[i % CHART_COLORS.length] + '40',
        pointBackgroundColor: CHART_COLORS[i % CHART_COLORS.length],
        tension: 0.3,
        pointRadius: 5,
        pointHoverRadius: 8,
        fill: false,
        borderWidth: 2,
      }));

      const yLabel = (ANALYTICS_METRICS.find(m => m.key === yMetric) || {}).label || yMetric;
      const xLabel = chartType === "line" ? "Date" : (ANALYTICS_METRICS.find(m => m.key === xMetric) || {}).label || xMetric;

      chartRef.current = new Chart(canvasRef.current, {
        type: chartType === "line" ? "line" : "scatter",
        data: { datasets: chartDatasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 400 },
          interaction: { mode: 'nearest', intersect: false },
          plugins: {
            legend: { position: 'top', labels: { color: '#e0e0e0', font: { family: 'Inter', size: 12 }, usePointStyle: true, pointStyle: 'circle', padding: 16 } },
            tooltip: {
              backgroundColor: '#1e1e1e', titleColor: '#fff', bodyColor: '#ccc',
              borderColor: '#333', borderWidth: 1, padding: 12,
              titleFont: { family: 'Inter', weight: '600' },
              bodyFont: { family: 'JetBrains Mono', size: 12 },
              callbacks: {
                title: (items) => {
                  const r = items[0]?.raw?._run;
                  return r ? `Run #${r.id} — ${r.dataset}` : '';
                },
                label: (ctx) => {
                  const lines = [`  ${ctx.dataset.label}`];
                  lines.push(`  ${yLabel}: ${ctx.parsed.y?.toFixed(3)}`);
                  if (chartType === "line") {
                    lines.push(`  Date: ${new Date(ctx.parsed.x).toLocaleString()}`);
                  } else {
                    lines.push(`  ${xLabel}: ${ctx.parsed.x?.toFixed(3)}`);
                  }
                  const r = ctx.raw?._run;
                  if (r?.preset) lines.push(`  Preset: ${r.preset}`);
                  if (r?.hostname) lines.push(`  Host: ${r.hostname}`);
                  return lines;
                },
              },
            },
          },
          scales: {
            x: chartType === "line" ? {
              type: 'time',
              time: { tooltipFormat: 'MMM d, yyyy HH:mm', displayFormats: { hour: 'MMM d HH:mm', day: 'MMM d', week: 'MMM d', month: 'MMM yyyy' } },
              ticks: { color: '#888', font: { size: 11 }, maxRotation: 45 },
              grid: { color: 'rgba(255,255,255,0.04)' },
              title: { display: true, text: 'Date', color: '#888', font: { size: 12 } },
            } : {
              type: 'linear',
              ticks: { color: '#888', font: { size: 11 } },
              grid: { color: 'rgba(255,255,255,0.04)' },
              title: { display: true, text: xLabel, color: '#888', font: { size: 12 } },
            },
            y: {
              ticks: { color: '#888', font: { size: 11 } },
              grid: { color: 'rgba(255,255,255,0.04)' },
              title: { display: true, text: yLabel, color: '#888', font: { size: 12 } },
            },
          },
        },
      });
    } else if (chartType === "bar") {
      const groups = {};
      filteredRuns.forEach(r => {
        const group = r[groupBy] || "unknown";
        if (!groups[group]) groups[group] = [];
        const val = r[yKey];
        if (val != null) groups[group].push(val);
      });
      const labels = Object.keys(groups).sort();
      const avgData = labels.map(l => { const v = groups[l]; return v.reduce((a,b)=>a+b,0)/v.length; });
      const minData = labels.map(l => Math.min(...groups[l]));
      const maxData = labels.map(l => Math.max(...groups[l]));
      const yLabel = (ANALYTICS_METRICS.find(m => m.key === yMetric) || {}).label || yMetric;

      chartRef.current = new Chart(canvasRef.current, {
        type: 'bar',
        data: {
          labels,
          datasets: [
            { label: `Avg ${yLabel}`, data: avgData, backgroundColor: labels.map((_,i) => CHART_COLORS[i % CHART_COLORS.length] + 'cc'), borderColor: labels.map((_,i) => CHART_COLORS[i % CHART_COLORS.length]), borderWidth: 1, borderRadius: 4 },
            { label: `Min`, data: minData, backgroundColor: 'rgba(255,255,255,0.06)', borderColor: 'rgba(255,255,255,0.15)', borderWidth: 1, borderRadius: 4 },
            { label: `Max`, data: maxData, backgroundColor: 'rgba(118,185,0,0.15)', borderColor: 'rgba(118,185,0,0.4)', borderWidth: 1, borderRadius: 4 },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 400 },
          plugins: {
            legend: { position: 'top', labels: { color: '#e0e0e0', font: { family: 'Inter', size: 12 }, padding: 16 } },
            tooltip: { backgroundColor: '#1e1e1e', titleColor: '#fff', bodyColor: '#ccc', borderColor: '#333', borderWidth: 1, bodyFont: { family: 'JetBrains Mono', size: 12 } },
          },
          scales: {
            x: { ticks: { color: '#888', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
            y: { ticks: { color: '#888', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: yLabel, color: '#888', font: { size: 12 } } },
          },
        },
      });
    }

    return () => { if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; } };
  }, [filteredRuns, chartType, yMetric, xMetric, groupBy]);

  const stats = useMemo(() => {
    const vals = filteredRuns.map(r => r[yMetric]).filter(v => v != null);
    if (!vals.length) return null;
    const avg = vals.reduce((a,b)=>a+b,0)/vals.length;
    const sorted = [...vals].sort((a,b)=>a-b);
    const median = sorted.length%2===0 ? (sorted[sorted.length/2-1]+sorted[sorted.length/2])/2 : sorted[Math.floor(sorted.length/2)];
    const stddev = Math.sqrt(vals.reduce((sum,v)=>sum+Math.pow(v-avg,2),0)/vals.length);
    return { count: vals.length, avg, min: sorted[0], max: sorted[sorted.length-1], median, stddev };
  }, [filteredRuns, yMetric]);

  const labelStyle = {display:'block',fontSize:'10px',fontWeight:600,color:'var(--nv-text-dim)',marginBottom:'4px',textTransform:'uppercase',letterSpacing:'0.05em'};
  const yLabel = (ANALYTICS_METRICS.find(m=>m.key===yMetric)||{}).label || yMetric;

  return (
    <>
      {/* Controls */}
      <div className="card" style={{padding:'16px 20px',marginBottom:'20px'}}>
        <div style={{display:'flex',gap:'16px',alignItems:'flex-end',flexWrap:'wrap'}}>
          {/* Chart Type */}
          <div>
            <div style={labelStyle}>Chart Type</div>
            <div style={{display:'flex',gap:'2px',background:'var(--nv-bg)',borderRadius:'8px',padding:'2px',border:'1px solid var(--nv-border)'}}>
              {[{k:'line',l:'Line'},{k:'scatter',l:'Scatter'},{k:'bar',l:'Bar'}].map(t=>(
                <button key={t.k} className="btn btn-sm" onClick={()=>setChartType(t.k)}
                  style={{borderRadius:'6px',padding:'5px 14px',fontSize:'12px',
                    background:chartType===t.k?'var(--nv-green)':'transparent',
                    color:chartType===t.k?'#000':'var(--nv-text-muted)',
                    fontWeight:chartType===t.k?700:500,border:'none'}}>
                  {t.l}
                </button>
              ))}
            </div>
          </div>

          {/* Y Axis */}
          <div>
            <div style={labelStyle}>Y Axis</div>
            <select className="select" value={yMetric} onChange={e=>setYMetric(e.target.value)} style={{minWidth:'140px'}}>
              {ANALYTICS_METRICS.map(m=><option key={m.key} value={m.key}>{m.label}</option>)}
            </select>
          </div>

          {/* X Axis (scatter only) */}
          {chartType === "scatter" && (
            <div>
              <div style={labelStyle}>X Axis</div>
              <select className="select" value={xMetric} onChange={e=>setXMetric(e.target.value)} style={{minWidth:'140px'}}>
                {ANALYTICS_METRICS.filter(m=>m.key!==yMetric).map(m=><option key={m.key} value={m.key}>{m.label}</option>)}
              </select>
            </div>
          )}

          {/* Group By */}
          <div>
            <div style={labelStyle}>Group By</div>
            <select className="select" value={groupBy} onChange={e=>setGroupBy(e.target.value)} style={{minWidth:'110px'}}>
              <option value="dataset">Dataset</option>
              <option value="preset">Preset</option>
              <option value="hostname">Host</option>
              <option value="gpu_type">GPU Type</option>
            </select>
          </div>

          <div style={{flex:'1 0 0',minWidth:'40px'}}></div>
          <button className="btn btn-secondary btn-icon" onClick={onRefresh} title="Refresh"><IconRefresh /></button>
        </div>

        {/* Filters row */}
        <div style={{display:'flex',gap:'12px',alignItems:'flex-end',flexWrap:'wrap',marginTop:'12px',paddingTop:'12px',borderTop:'1px solid var(--nv-border)'}}>
          <div>
            <div style={labelStyle}>Dataset</div>
            <select className="select" value={filterDataset} onChange={e=>setFilterDataset(e.target.value)} style={{minWidth:'150px'}}>
              <option value="">All datasets</option>
              {allDatasets.map(d=><option key={d} value={d}>{d}</option>)}
            </select>
          </div>
          <div>
            <div style={labelStyle}>Preset</div>
            <select className="select" value={filterPreset} onChange={e=>setFilterPreset(e.target.value)} style={{minWidth:'130px'}}>
              <option value="">All presets</option>
              {allPresets.map(p=><option key={p} value={p}>{p}</option>)}
            </select>
          </div>
          <div>
            <div style={labelStyle}>Status</div>
            <select className="select" value={filterStatus} onChange={e=>setFilterStatus(e.target.value)} style={{minWidth:'100px'}}>
              <option value="all">All</option>
              <option value="pass">Pass only</option>
              <option value="fail">Fail only</option>
            </select>
          </div>
          <div>
            <div style={labelStyle}>From</div>
            <input type="date" className="input" value={dateFrom} onChange={e=>setDateFrom(e.target.value)} style={{minWidth:'140px',colorScheme:'dark'}} />
          </div>
          <div>
            <div style={labelStyle}>To</div>
            <input type="date" className="input" value={dateTo} onChange={e=>setDateTo(e.target.value)} style={{minWidth:'140px',colorScheme:'dark'}} />
          </div>
          {(filterDataset||filterPreset||filterStatus!=='all'||dateFrom||dateTo) && (
            <button className="btn btn-ghost btn-sm" onClick={()=>{setFilterDataset('');setFilterPreset('');setFilterStatus('all');setDateFrom('');setDateTo('');}}
              style={{color:'var(--nv-text-muted)',whiteSpace:'nowrap'}}>
              <IconX /> Clear filters
            </button>
          )}
        </div>
      </div>

      {/* Chart */}
      <div className="card" style={{padding:'20px',marginBottom:'20px'}}>
        <div style={{position:'relative',height:'450px'}}>
          {loading ? (
            <div style={{display:'flex',alignItems:'center',justifyContent:'center',height:'100%',flexDirection:'column',gap:'12px',color:'var(--nv-text-muted)'}}>
              <div className="spinner spinner-lg"></div><div>Loading data…</div>
            </div>
          ) : filteredRuns.length === 0 ? (
            <div style={{display:'flex',alignItems:'center',justifyContent:'center',height:'100%',flexDirection:'column',gap:'8px',color:'var(--nv-text-muted)'}}>
              <IconChart />
              <div style={{fontSize:'15px'}}>No data matching filters</div>
              <div style={{fontSize:'12px',color:'var(--nv-text-dim)'}}>Adjust filters or trigger some runs to see analytics.</div>
            </div>
          ) : (
            <canvas ref={canvasRef}></canvas>
          )}
        </div>
      </div>

      {/* Summary Stats */}
      {stats && (
        <div style={{display:'grid',gridTemplateColumns:'repeat(6, 1fr)',gap:'12px'}}>
          <div className="metric-card">
            <div className="metric-value" style={{fontSize:'20px',color:'#fff'}}>{stats.count}</div>
            <div className="metric-label">Runs</div>
          </div>
          <div className="metric-card">
            <div className="metric-value" style={{fontSize:'20px'}}>{stats.avg.toFixed(2)}</div>
            <div className="metric-label">Avg {yLabel}</div>
          </div>
          <div className="metric-card">
            <div className="metric-value" style={{fontSize:'20px'}}>{stats.median.toFixed(2)}</div>
            <div className="metric-label">Median</div>
          </div>
          <div className="metric-card">
            <div className="metric-value" style={{fontSize:'20px',color:'#64b4ff'}}>{stats.min.toFixed(2)}</div>
            <div className="metric-label">Min</div>
          </div>
          <div className="metric-card">
            <div className="metric-value" style={{fontSize:'20px',color:'#fcd34d'}}>{stats.max.toFixed(2)}</div>
            <div className="metric-label">Max</div>
          </div>
          <div className="metric-card">
            <div className="metric-value" style={{fontSize:'20px',color:'var(--nv-text-muted)'}}>{stats.stddev.toFixed(2)}</div>
            <div className="metric-label">Std Dev</div>
          </div>
        </div>
      )}
    </>
  );
}
