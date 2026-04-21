/* ===== Reporting View ===== */

const REPORT_CHARTS = [
  { id: "trend_pps", label: "Pages/sec Over Time", metric: "pages_per_sec", type: "line" },
  { id: "trend_recall5", label: "Recall@5 Over Time", metric: "recall_5", type: "line" },
  { id: "trend_recall1", label: "Recall@1 Over Time", metric: "recall_1", type: "line" },
  { id: "trend_recall10", label: "Recall@10 Over Time", metric: "recall_10", type: "line" },
  { id: "trend_ingest", label: "Ingest Time Over Time", metric: "ingest_secs", type: "line" },
  { id: "bar_pps", label: "Avg Pages/sec by Dataset", metric: "pages_per_sec", type: "bar" },
  { id: "bar_recall5", label: "Avg Recall@5 by Dataset", metric: "recall_5", type: "bar" },
  { id: "gpu_pps", label: "Pages/sec by GPU SKU (per Dataset)", metric: "pages_per_sec", type: "gpu_bar" },
  { id: "gpu_recall5", label: "Recall@5 by GPU SKU (per Dataset)", metric: "recall_5", type: "gpu_bar" },
  { id: "gpu_recall1", label: "Recall@1 by GPU SKU (per Dataset)", metric: "recall_1", type: "gpu_bar" },
  { id: "gpu_recall10", label: "Recall@10 by GPU SKU (per Dataset)", metric: "recall_10", type: "gpu_bar" },
  { id: "gpu_ingest", label: "Ingest Time by GPU SKU (per Dataset)", metric: "ingest_secs", type: "gpu_bar" },
];

const PRESET_MATRIX_PRESETS = [
  { key: "PE_GE_OCR_TE_DENSE", label: "TE Dense", desc: "Page Elements + Graphic Elements + OCR + Text Embedding + Dense" },
  { key: "PE_GE_OCR_TE_HYBRID", label: "TE Hybrid", desc: "Page Elements + Graphic Elements + OCR + Text Embedding + Hybrid" },
  { key: "PE_GE_OCR_VL_IMAGE_ONLY_DENSE", label: "VL Image-Only Dense", desc: "Page Elements + Graphic Elements + OCR + llama-nemotron-embed-vl-1b-v2 (image only) + Dense" },
  { key: "PE_GE_OCR_VL_IMAGE_ONLY_HYBRID", label: "VL Image-Only Hybrid", desc: "Page Elements + Graphic Elements + OCR + llama-nemotron-embed-vl-1b-v2 (image only) + Hybrid" },
  { key: "PE_GE_OCR_VL_IMAGE_TEXT_DENSE", label: "VL Image+Text Dense", desc: "Page Elements + Graphic Elements + OCR + llama-nemotron-embed-vl-1b-v2 (image and text) + Dense" },
  { key: "PE_GE_OCR_VL_IMAGE_TEXT_HYBRID", label: "VL Image+Text Hybrid", desc: "Page Elements + Graphic Elements + OCR + llama-nemotron-embed-vl-1b-v2 (image and text) + Hybrid" },
  { key: "PE_GE_OCR_VL_TEXT_ONLY_DENSE", label: "VL Text-Only Dense", desc: "Page Elements + Graphic Elements + OCR + llama-nemotron-embed-vl-1b-v2 (Text only) + Dense" },
  { key: "PE_GE_OCR_VL_TEXT_ONLY_HYBRID", label: "VL Text-Only Hybrid", desc: "Page Elements + Graphic Elements + OCR + llama-nemotron-embed-vl-1b-v2 (Text only) + Hybrid" },
];

function DataExportSection({ filteredRuns, filterDataset, filterPreset, filterStatus, dateFrom, dateTo }) {
  const [includeRaw, setIncludeRaw] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [exportFormat, setExportFormat] = useState("json");

  const buildExportFilename = (ext) => `harness_runs_export_${new Date().toISOString().slice(0,10)}.${ext}`;

  const exportJson = async () => {
    setExporting(true);
    try {
      const params = new URLSearchParams();
      if (filterDataset) params.set('dataset', filterDataset);
      if (filterPreset) params.set('preset', filterPreset);
      if (filterStatus && filterStatus !== 'all') params.set('status', filterStatus);
      if (dateFrom) params.set('date_from', dateFrom);
      if (dateTo) params.set('date_to', dateTo);
      params.set('include_raw', includeRaw ? 'true' : 'false');

      const resp = await fetch(`/api/reports/export?${params}`);
      if (resp.ok) {
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = buildExportFilename('json');
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch (e) { console.error('Export failed', e); }
    setExporting(false);
  };

  const exportCsv = () => {
    if (filteredRuns.length === 0) return;
    const fields = ['id','timestamp','dataset','preset','success','hostname','gpu_type',
      'pages_per_sec','recall_1','recall_5','recall_10','ingest_secs','pages','files',
      'trigger_source','git_commit','execution_commit','num_gpus','ray_cluster_mode','ray_dashboard_url'];
    const header = fields.join(',');
    const rows = filteredRuns.map(r =>
      fields.map(f => {
        const v = r[f];
        if (v == null) return '';
        const s = String(v);
        return s.includes(',') || s.includes('"') || s.includes('\n') ? `"${s.replace(/"/g, '""')}"` : s;
      }).join(',')
    );
    const csv = [header, ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = buildExportFilename('csv');
    a.click();
    URL.revokeObjectURL(url);
  };

  const labelStyle = {display:'block',fontSize:'10px',fontWeight:600,color:'var(--nv-text-dim)',marginBottom:'4px',textTransform:'uppercase',letterSpacing:'0.05em'};

  return (
    <div className="card" style={{padding:'20px',marginBottom:'20px'}}>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'16px'}}>
        <h3 style={{margin:0,fontSize:'16px',fontWeight:600}}>Data Export</h3>
        <span style={{fontSize:'12px',color:'var(--nv-text-dim)'}}>{filteredRuns.length} run{filteredRuns.length!==1?'s':''} matching filters</span>
      </div>

      <div style={{display:'flex',gap:'16px',alignItems:'flex-end',flexWrap:'wrap',marginBottom:'16px'}}>
        <div>
          <div style={labelStyle}>Format</div>
          <div style={{display:'flex',gap:'2px',background:'var(--nv-bg)',borderRadius:'8px',padding:'2px',border:'1px solid var(--nv-border)'}}>
            {[{k:'json',l:'JSON'},{k:'csv',l:'CSV'}].map(t=>(
              <button key={t.k} className="btn btn-sm" onClick={()=>setExportFormat(t.k)}
                style={{borderRadius:'6px',padding:'5px 16px',fontSize:'12px',
                  background:exportFormat===t.k?'var(--nv-green)':'transparent',
                  color:exportFormat===t.k?'#000':'var(--nv-text-muted)',
                  fontWeight:exportFormat===t.k?600:400,border:'none'}}>
                {t.l}
              </button>
            ))}
          </div>
        </div>

        {exportFormat === 'json' && (
          <label style={{display:'flex',alignItems:'center',gap:'6px',fontSize:'13px',color:'var(--nv-text-muted)',cursor:'pointer'}}>
            <input type="checkbox" checked={includeRaw} onChange={e=>setIncludeRaw(e.target.checked)}
              style={{accentColor:'var(--nv-green)'}} />
            Include full raw result JSON
          </label>
        )}
      </div>

      <div style={{fontSize:'12px',color:'var(--nv-text-dim)',marginBottom:'16px',lineHeight:'1.5'}}>
        {exportFormat === 'json' ? (
          <>
            Exports a structured JSON file containing all matching runs with metadata, metrics, and
            {includeRaw ? ' full raw result payloads' : ' summary data only'}.
            Ideal for programmatic analysis with Python, notebooks, or other tools.
          </>
        ) : (
          <>
            Exports a flat CSV file with key metrics for each matching run.
            Ideal for spreadsheet analysis, quick reviews, or importing into BI tools.
          </>
        )}
      </div>

      <button className="btn btn-primary" onClick={exportFormat === 'json' ? exportJson : exportCsv}
        disabled={exporting || filteredRuns.length === 0}>
        {exporting ? <><div className="spinner" style={{width:14,height:14}}></div> Exporting…</> : <><IconDownload /> Export {exportFormat.toUpperCase()}</>}
      </button>
    </div>
  );
}

function ReportingView({ runs, datasets, loading }) {
  const [filterDataset, setFilterDataset] = useState("");
  const [filterPreset, setFilterPreset] = useState("");
  const [filterStatus, setFilterStatus] = useState("all");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [selectedCharts, setSelectedCharts] = useState(REPORT_CHARTS.map(c => c.id));
  const [reportTitle, setReportTitle] = useState("Harness Performance Report");
  const [generating, setGenerating] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const [previewImages, setPreviewImages] = useState([]);
  const chartRefs = useRef({});
  const canvasRefs = useRef({});

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

  const activeCharts = useMemo(() => REPORT_CHARTS.filter(c => selectedCharts.includes(c.id)), [selectedCharts]);

  const toggleChart = (id) => {
    setSelectedCharts(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
  };

  const summaryStats = useMemo(() => {
    const out = {};
    for (const m of ANALYTICS_METRICS) {
      const vals = filteredRuns.map(r => r[m.key]).filter(v => v != null);
      if (vals.length === 0) { out[m.key] = null; continue; }
      const avg = vals.reduce((a,b)=>a+b,0)/vals.length;
      const sorted = [...vals].sort((a,b)=>a-b);
      out[m.key] = { avg, min: sorted[0], max: sorted[sorted.length-1], count: vals.length };
    }
    return out;
  }, [filteredRuns]);

  const presetMatrix = useMemo(() => {
    const map = {};
    runs.forEach(r => {
      if (r.success !== 1 || !r.preset || !r.dataset) return;
      const key = `${r.preset}|||${r.dataset}`;
      const ts = parseRunTimestamp(r.timestamp);
      if (!ts) return;
      if (!map[key] || ts > map[key].ts) {
        map[key] = { ts, pps: r.pages_per_sec, recall5: r.recall_5 };
      }
    });
    const datasets = [...new Set(runs.map(r => r.dataset).filter(Boolean))].sort();
    return { map, datasets };
  }, [runs]);

  const buildChart = useCallback((canvasEl, chartDef) => {
    if (!canvasEl) return null;
    const metricLabel = (ANALYTICS_METRICS.find(m => m.key === chartDef.metric) || {}).label || chartDef.metric;

    if (chartDef.type === "line") {
      const groups = {};
      filteredRuns.forEach(r => {
        const group = r.dataset || "unknown";
        if (!groups[group]) groups[group] = [];
        const yVal = r[chartDef.metric];
        if (yVal == null) return;
        const date = parseRunTimestamp(r.timestamp);
        if (date) groups[group].push({ x: date, y: yVal });
      });
      const chartDatasets = Object.entries(groups).map(([name, points], i) => ({
        label: name,
        data: points.sort((a, b) => a.x.getTime() - b.x.getTime()),
        borderColor: CHART_COLORS[i % CHART_COLORS.length],
        backgroundColor: CHART_COLORS[i % CHART_COLORS.length] + '40',
        pointBackgroundColor: CHART_COLORS[i % CHART_COLORS.length],
        tension: 0.3, pointRadius: 4, fill: false, borderWidth: 2,
      }));
      return new Chart(canvasEl, {
        type: 'line',
        data: { datasets: chartDatasets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: { duration: 300 },
          plugins: {
            legend: { position: 'top', labels: { color: '#333', font: { family: 'Inter', size: 11 }, usePointStyle: true, padding: 12 } },
            title: { display: true, text: chartDef.label, color: '#222', font: { family: 'Inter', size: 14, weight: '600' }, padding: { bottom: 10 } },
          },
          scales: {
            x: { type: 'time', time: { tooltipFormat: 'MMM d, yyyy HH:mm', displayFormats: { hour: 'MMM d HH:mm', day: 'MMM d' } },
              ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#eee' }, title: { display: true, text: 'Date', color: '#666' } },
            y: { ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#eee' }, title: { display: true, text: metricLabel, color: '#666' } },
          },
        },
      });
    } else if (chartDef.type === "bar") {
      const groups = {};
      filteredRuns.forEach(r => {
        const group = r.dataset || "unknown";
        if (!groups[group]) groups[group] = [];
        const val = r[chartDef.metric];
        if (val != null) groups[group].push(val);
      });
      const labels = Object.keys(groups).sort();
      const avgData = labels.map(l => { const v = groups[l]; return v.reduce((a,b)=>a+b,0)/v.length; });
      return new Chart(canvasEl, {
        type: 'bar',
        data: {
          labels,
          datasets: [
            { label: `Avg ${metricLabel}`, data: avgData, backgroundColor: labels.map((_,i) => CHART_COLORS[i % CHART_COLORS.length] + 'cc'),
              borderColor: labels.map((_,i) => CHART_COLORS[i % CHART_COLORS.length]), borderWidth: 1, borderRadius: 4 },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: { duration: 300 },
          plugins: {
            legend: { display: false },
            title: { display: true, text: chartDef.label, color: '#222', font: { family: 'Inter', size: 14, weight: '600' }, padding: { bottom: 10 } },
          },
          scales: {
            x: { ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#eee' } },
            y: { ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#eee' }, title: { display: true, text: metricLabel, color: '#666' } },
          },
        },
      });
    } else if (chartDef.type === "gpu_bar") {
      const nested = {};
      filteredRuns.forEach(r => {
        const gpuBase = r.gpu_type || "Unknown GPU";
        const gpu = r.num_gpus != null ? `${gpuBase} (x${r.num_gpus})` : gpuBase;
        const ds = r.dataset || "unknown";
        const val = r[chartDef.metric];
        if (val == null) return;
        if (!nested[gpu]) nested[gpu] = {};
        if (!nested[gpu][ds]) nested[gpu][ds] = [];
        nested[gpu][ds].push(val);
      });
      const gpuLabels = Object.keys(nested).sort();
      const allDatasets = [...new Set(filteredRuns.map(r => r.dataset || "unknown"))].sort();
      const chartDatasets = allDatasets.map((ds, i) => ({
        label: ds,
        data: gpuLabels.map(gpu => {
          const vals = (nested[gpu] && nested[gpu][ds]) || [];
          return vals.length > 0 ? +(vals.reduce((a,b)=>a+b,0) / vals.length).toFixed(3) : null;
        }),
        backgroundColor: CHART_COLORS[i % CHART_COLORS.length] + 'cc',
        borderColor: CHART_COLORS[i % CHART_COLORS.length],
        borderWidth: 1, borderRadius: 4,
      }));
      return new Chart(canvasEl, {
        type: 'bar',
        data: { labels: gpuLabels, datasets: chartDatasets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: { duration: 300 },
          plugins: {
            legend: { position: 'top', labels: { color: '#333', font: { family: 'Inter', size: 11 }, usePointStyle: true, padding: 12 } },
            title: { display: true, text: chartDef.label, color: '#222', font: { family: 'Inter', size: 14, weight: '600' }, padding: { bottom: 10 } },
            tooltip: {
              callbacks: {
                label: function(ctx) {
                  const v = ctx.parsed.y;
                  return v != null ? `${ctx.dataset.label}: ${v.toFixed(2)}` : '';
                },
                afterBody: function(items) {
                  if (!items.length) return '';
                  const gpu = items[0].label;
                  const ds = items[0].dataset.label;
                  const vals = (nested[gpu] && nested[gpu][ds]) || [];
                  if (vals.length <= 1) return '';
                  return `  (${vals.length} runs)`;
                },
              },
            },
          },
          scales: {
            x: { ticks: { color: '#555', font: { size: 10 }, maxRotation: 45, minRotation: 0 }, grid: { color: '#eee' },
              title: { display: true, text: 'GPU SKU', color: '#666' } },
            y: { ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#eee' },
              title: { display: true, text: metricLabel, color: '#666' }, beginAtZero: true },
          },
        },
      });
    }
    return null;
  }, [filteredRuns]);

  const generatePreview = useCallback(async () => {
    setPreviewing(true);
    setPreviewImages([]);
    Object.values(chartRefs.current).forEach(c => { if (c) c.destroy(); });
    chartRefs.current = {};
    await new Promise(r => setTimeout(r, 100));

    const images = [];
    for (const chartDef of activeCharts) {
      const canvas = canvasRefs.current[chartDef.id];
      if (!canvas) continue;
      const chart = buildChart(canvas, chartDef);
      if (chart) {
        chartRefs.current[chartDef.id] = chart;
        await new Promise(r => setTimeout(r, 500));
        images.push({ id: chartDef.id, label: chartDef.label, dataUrl: chart.toBase64Image('image/png', 1) });
      }
    }
    setPreviewImages(images);
    setPreviewing(false);
  }, [activeCharts, buildChart]);

  useEffect(() => {
    return () => { Object.values(chartRefs.current).forEach(c => { if (c) c.destroy(); }); };
  }, []);

  const downloadSinglePng = (img) => {
    const a = document.createElement('a');
    a.href = img.dataUrl;
    a.download = `${img.id}.png`;
    a.click();
  };

  const downloadAllPngs = async () => {
    if (previewImages.length === 0) return;
    setGenerating(true);
    try {
      const resp = await fetch('/api/reports/bundle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ images: previewImages.map(img => ({ filename: `${img.id}.png`, data_url: img.dataUrl })) }),
      });
      if (resp.ok) {
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `harness_report_${new Date().toISOString().slice(0,10)}.zip`;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch (e) { console.error('Download failed', e); }
    setGenerating(false);
  };

  const downloadSummaryPng = async () => {
    const w = 1200, h = 800;
    const offscreen = document.createElement('canvas');
    offscreen.width = w;
    offscreen.height = h;
    const ctx = offscreen.getContext('2d');

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#76b900';
    ctx.fillRect(0, 0, w, 60);
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 22px Inter, sans-serif';
    ctx.fillText(reportTitle, 24, 40);

    const now = new Date().toLocaleString();
    ctx.font = '12px Inter, sans-serif';
    ctx.fillStyle = '#ffffffcc';
    ctx.fillText(`Generated: ${now}`, w - 260, 40);

    let y = 80;
    ctx.fillStyle = '#333';
    ctx.font = 'bold 16px Inter, sans-serif';
    ctx.fillText('Filters', 24, y);
    y += 24;
    ctx.font = '12px Inter, sans-serif';
    ctx.fillStyle = '#555';
    const filters = [];
    if (filterDataset) filters.push(`Dataset: ${filterDataset}`);
    if (filterPreset) filters.push(`Preset: ${filterPreset}`);
    if (filterStatus !== 'all') filters.push(`Status: ${filterStatus}`);
    if (dateFrom) filters.push(`From: ${dateFrom}`);
    if (dateTo) filters.push(`To: ${dateTo}`);
    ctx.fillText(filters.length > 0 ? filters.join('  |  ') : 'None (all runs)', 24, y);
    y += 30;

    ctx.fillStyle = '#333';
    ctx.font = 'bold 16px Inter, sans-serif';
    ctx.fillText(`Summary (${filteredRuns.length} runs)`, 24, y);
    y += 8;

    const metricCols = ANALYTICS_METRICS.filter(m => summaryStats[m.key]);
    const colW = Math.min(160, (w - 48) / metricCols.length);
    metricCols.forEach((m, i) => {
      const s = summaryStats[m.key];
      const x = 24 + i * colW;
      ctx.fillStyle = '#f0f0f0';
      ctx.fillRect(x, y, colW - 8, 64);
      ctx.fillStyle = '#76b900';
      ctx.font = 'bold 14px Inter, sans-serif';
      ctx.fillText(s.avg.toFixed(2), x + 8, y + 24);
      ctx.fillStyle = '#888';
      ctx.font = '10px Inter, sans-serif';
      ctx.fillText(`min ${s.min.toFixed(2)} / max ${s.max.toFixed(2)}`, x + 8, y + 42);
      ctx.fillStyle = '#333';
      ctx.font = '11px Inter, sans-serif';
      ctx.fillText(m.label, x + 8, y + 58);
    });
    y += 80;

    if (previewImages.length > 0) {
      const chartW = (w - 48 - 16) / 2;
      const chartH = Math.min(250, (h - y - 40) / Math.ceil(previewImages.length / 2));
      for (let i = 0; i < previewImages.length; i++) {
        const col = i % 2;
        const row = Math.floor(i / 2);
        const px = 24 + col * (chartW + 16);
        const py = y + row * (chartH + 12);
        if (py + chartH > h - 10) break;
        const img = new Image();
        img.src = previewImages[i].dataUrl;
        await new Promise(r => { img.onload = r; img.onerror = r; });
        ctx.drawImage(img, px, py, chartW, chartH);
      }
    }

    const dataUrl = offscreen.toDataURL('image/png', 1);
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = `harness_summary_${new Date().toISOString().slice(0,10)}.png`;
    a.click();
  };

  const downloadPresetMatrixPng = () => {
    const { map, datasets } = presetMatrix;
    const presets = PRESET_MATRIX_PRESETS;
    const colW = 210, rowHeaderW = 280, headerH = 60, rowH = 48, pad = 32;
    const titleBarH = 64;
    const tableW = rowHeaderW + datasets.length * colW;
    const w = pad * 2 + tableW;
    const h = titleBarH + pad + headerH + presets.length * rowH + pad;
    const offscreen = document.createElement('canvas');
    offscreen.width = w;
    offscreen.height = h;
    const ctx = offscreen.getContext('2d');

    const bgColor = '#0e0e0e';
    const cardColor = '#1a1a1a';
    const borderColor = '#262626';
    const textColor = '#e0e0e0';
    const textMuted = '#888888';
    const textDim = '#555555';
    const green = '#76b900';

    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, w, h);

    ctx.fillStyle = green;
    ctx.fillRect(0, 0, w, titleBarH);
    ctx.fillStyle = '#000000';
    ctx.font = 'bold 22px Inter, sans-serif';
    ctx.fillText('Preset \u00d7 Dataset Matrix \u2014 Last Successful Run', pad, 40);
    const now = new Date().toLocaleString();
    ctx.font = '12px Inter, sans-serif';
    ctx.fillStyle = '#000000aa';
    ctx.textAlign = 'right';
    ctx.fillText(`Generated: ${now}`, w - pad, 40);
    ctx.textAlign = 'left';

    const tableTop = titleBarH + pad;
    const fmtV = (v) => v != null ? v.toFixed(2) : '\u2014';

    const bestPps = {};
    const bestRecall = {};
    datasets.forEach(ds => {
      presets.forEach(p => {
        const cell = map[`${p.key}|||${ds}`];
        if (cell && cell.pps != null && (bestPps[ds] == null || cell.pps > bestPps[ds])) bestPps[ds] = cell.pps;
        if (cell && cell.recall5 != null && (bestRecall[ds] == null || cell.recall5 > bestRecall[ds])) bestRecall[ds] = cell.recall5;
      });
    });

    ctx.fillStyle = cardColor;
    ctx.fillRect(pad, tableTop, tableW, headerH);
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, tableTop, tableW, headerH);

    ctx.fillStyle = textDim;
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('PRESET', pad + 14, tableTop + headerH / 2 + 4);

    ctx.textAlign = 'center';
    datasets.forEach((ds, i) => {
      const cx = pad + rowHeaderW + i * colW + colW / 2;
      ctx.fillStyle = textColor;
      ctx.font = 'bold 13px Inter, sans-serif';
      ctx.fillText(ds, cx, tableTop + 24);
      ctx.fillStyle = textDim;
      ctx.font = '10px Inter, sans-serif';
      ctx.fillText('PPS / Recall@5', cx, tableTop + 44);
    });

    presets.forEach((p, ri) => {
      const ry = tableTop + headerH + ri * rowH;
      ctx.fillStyle = ri % 2 === 0 ? cardColor : bgColor;
      ctx.fillRect(pad, ry, tableW, rowH);

      ctx.strokeStyle = borderColor;
      ctx.beginPath();
      ctx.moveTo(pad, ry + rowH);
      ctx.lineTo(pad + tableW, ry + rowH);
      ctx.stroke();

      ctx.fillStyle = textColor;
      ctx.font = '13px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(p.label, pad + 14, ry + rowH / 2 + 5);

      datasets.forEach((ds, ci) => {
        const cell = map[`${p.key}|||${ds}`];
        const cx = pad + rowHeaderW + ci * colW + colW / 2;
        const ppsVal = cell ? cell.pps : null;
        const recallVal = cell ? cell.recall5 : null;
        const isBest = (ppsVal != null && ppsVal === bestPps[ds]) || (recallVal != null && recallVal === bestRecall[ds]);
        const cellText = `${fmtV(ppsVal)} / ${fmtV(recallVal)}`;

        ctx.textAlign = 'center';
        ctx.fillStyle = isBest ? green : textMuted;
        ctx.font = (isBest ? 'bold ' : '') + '13px Inter, sans-serif';
        ctx.fillText(cellText, cx, ry + rowH / 2 + 5);
      });
    });

    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, tableTop, tableW, headerH + presets.length * rowH);
    for (let i = 0; i <= datasets.length; i++) {
      const x = pad + rowHeaderW + i * colW;
      ctx.beginPath();
      ctx.moveTo(x, tableTop);
      ctx.lineTo(x, tableTop + headerH + presets.length * rowH);
      ctx.stroke();
    }
    const rhX = pad + rowHeaderW;
    ctx.beginPath();
    ctx.moveTo(rhX, tableTop);
    ctx.lineTo(rhX, tableTop + headerH + presets.length * rowH);
    ctx.stroke();

    const dataUrl = offscreen.toDataURL('image/png', 1);
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = `preset_matrix_${new Date().toISOString().slice(0,10)}.png`;
    a.click();
  };

  const labelStyle = {display:'block',fontSize:'10px',fontWeight:600,color:'var(--nv-text-dim)',marginBottom:'4px',textTransform:'uppercase',letterSpacing:'0.05em'};

  return (
    <>
      {/* Report Configuration */}
      <div className="card" style={{padding:'20px',marginBottom:'20px'}}>
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'16px'}}>
          <h3 style={{margin:0,fontSize:'16px',fontWeight:600}}>Report Configuration</h3>
        </div>

        <div style={{marginBottom:'16px'}}>
          <div style={labelStyle}>Report Title</div>
          <input className="input" value={reportTitle} onChange={e=>setReportTitle(e.target.value)}
            style={{maxWidth:'400px'}} placeholder="Enter report title" />
        </div>

        <div style={{display:'flex',gap:'16px',alignItems:'flex-end',flexWrap:'wrap',marginBottom:'16px'}}>
          <div>
            <div style={labelStyle}>Dataset</div>
            <select className="select" value={filterDataset} onChange={e=>setFilterDataset(e.target.value)} style={{minWidth:'140px'}}>
              <option value="">All Datasets</option>
              {allDatasets.map(d=><option key={d} value={d}>{d}</option>)}
            </select>
          </div>
          <div>
            <div style={labelStyle}>Preset</div>
            <select className="select" value={filterPreset} onChange={e=>setFilterPreset(e.target.value)} style={{minWidth:'130px'}}>
              <option value="">All Presets</option>
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
        </div>

        <div style={{marginBottom:'16px'}}>
          <div style={labelStyle}>Charts to Include</div>
          <div style={{display:'flex',flexWrap:'wrap',gap:'8px'}}>
            {REPORT_CHARTS.map(c => (
              <label key={c.id} style={{display:'flex',alignItems:'center',gap:'6px',fontSize:'13px',color:'var(--nv-text-muted)',
                cursor:'pointer',padding:'6px 12px',borderRadius:'6px',border:'1px solid var(--nv-border)',
                background:selectedCharts.includes(c.id)?'rgba(118,185,0,0.1)':'transparent'}}>
                <input type="checkbox" checked={selectedCharts.includes(c.id)} onChange={()=>toggleChart(c.id)}
                  style={{accentColor:'var(--nv-green)'}} />
                {c.label}
              </label>
            ))}
          </div>
        </div>

        <div style={{display:'flex',gap:'8px',borderTop:'1px solid var(--nv-border)',paddingTop:'16px'}}>
          <button className="btn btn-primary" onClick={generatePreview} disabled={previewing || activeCharts.length === 0 || filteredRuns.length === 0}>
            {previewing ? <><div className="spinner" style={{width:14,height:14}}></div> Generating…</> : <><IconChart /> Generate Preview</>}
          </button>
          <button className="btn btn-ghost" onClick={downloadSummaryPng} disabled={previewImages.length === 0}>
            <IconDownload /> Summary PNG
          </button>
          <button className="btn btn-ghost" onClick={downloadAllPngs} disabled={previewImages.length === 0 || generating}>
            {generating ? <><div className="spinner" style={{width:14,height:14}}></div> Packaging…</> : <><IconDownload /> Download All (ZIP)</>}
          </button>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="card" style={{padding:'16px 20px',marginBottom:'20px'}}>
        <h3 style={{margin:'0 0 12px 0',fontSize:'14px',fontWeight:600,color:'var(--nv-text-muted)'}}>
          Report Summary — {filteredRuns.length} runs
        </h3>
        <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill, minmax(150px, 1fr))',gap:'10px'}}>
          {ANALYTICS_METRICS.map(m => {
            const s = summaryStats[m.key];
            if (!s) return (
              <div key={m.key} className="metric-card" style={{opacity:0.5}}>
                <div className="metric-value" style={{fontSize:'16px'}}>—</div>
                <div className="metric-label">{m.label}</div>
              </div>
            );
            return (
              <div key={m.key} className="metric-card">
                <div className="metric-value" style={{fontSize:'18px'}}>{s.avg.toFixed(2)}</div>
                <div className="metric-label">{m.label} (avg)</div>
                <div style={{fontSize:'10px',color:'var(--nv-text-dim)',marginTop:'4px'}}>
                  min {s.min.toFixed(2)} · max {s.max.toFixed(2)} · {s.count} pts
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* GPU Performance Comparison Table */}
      {(() => {
        const gpuRows = {};
        filteredRuns.forEach(r => {
          const gpuBase = r.gpu_type || "Unknown GPU";
          const gpu = r.num_gpus != null ? `${gpuBase} (x${r.num_gpus})` : gpuBase;
          const ds = r.dataset || "unknown";
          const key = `${gpu}|||${ds}`;
          if (!gpuRows[key]) gpuRows[key] = { gpu, dataset: ds, pps: [], recall1: [], recall5: [], recall10: [], ingest: [], count: 0 };
          gpuRows[key].count++;
          if (r.pages_per_sec != null) gpuRows[key].pps.push(r.pages_per_sec);
          if (r.recall_1 != null) gpuRows[key].recall1.push(r.recall_1);
          if (r.recall_5 != null) gpuRows[key].recall5.push(r.recall_5);
          if (r.recall_10 != null) gpuRows[key].recall10.push(r.recall_10);
          if (r.ingest_secs != null) gpuRows[key].ingest.push(r.ingest_secs);
        });
        const rows = Object.values(gpuRows).sort((a, b) => a.gpu.localeCompare(b.gpu) || a.dataset.localeCompare(b.dataset));
        const avg = arr => arr.length ? (arr.reduce((s,v) => s+v, 0) / arr.length) : null;
        const fmtN = (v, d=2) => v != null ? v.toFixed(d) : "\u2014";
        if (rows.length === 0) return null;
        const bestPps = {};
        rows.forEach(r => {
          const a = avg(r.pps);
          if (a != null && (!bestPps[r.dataset] || a > bestPps[r.dataset])) bestPps[r.dataset] = a;
        });
        return (
          <div className="card" style={{padding:'16px 20px',marginBottom:'20px'}}>
            <h3 style={{margin:'0 0 12px 0',fontSize:'14px',fontWeight:600,color:'var(--nv-text-muted)'}}>
              GPU Performance Comparison by Dataset
            </h3>
            <div style={{overflowX:'auto'}}>
              <table className="runs-table" style={{fontSize:'12px'}}>
                <thead>
                  <tr>
                    <th>GPU SKU</th><th>Dataset</th><th style={{textAlign:'right'}}>Runs</th>
                    <th style={{textAlign:'right'}}>Avg PPS</th><th style={{textAlign:'right'}}>Best PPS</th>
                    <th style={{textAlign:'right'}}>Recall@1</th><th style={{textAlign:'right'}}>Recall@5</th>
                    <th style={{textAlign:'right'}}>Recall@10</th><th style={{textAlign:'right'}}>Avg Ingest (s)</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => {
                    const avgPps = avg(r.pps);
                    const isBest = avgPps != null && bestPps[r.dataset] === avgPps;
                    return (
                      <tr key={i} style={{background: isBest ? 'rgba(118,185,0,0.04)' : 'transparent'}}>
                        <td style={{fontWeight:500,color:'#fff',whiteSpace:'nowrap'}}>{r.gpu}</td>
                        <td><span className="tag">{r.dataset}</span></td>
                        <td style={{textAlign:'right'}}>{r.count}</td>
                        <td style={{textAlign:'right',fontWeight:600,color:isBest?'var(--nv-green)':'#fff'}}>
                          {fmtN(avgPps)}
                          {isBest && <span style={{marginLeft:'6px',fontSize:'10px',color:'var(--nv-green)'}}>BEST</span>}
                        </td>
                        <td style={{textAlign:'right',color:'var(--nv-text-muted)'}}>{fmtN(r.pps.length ? Math.max(...r.pps) : null)}</td>
                        <td style={{textAlign:'right'}}>{fmtN(avg(r.recall1))}</td>
                        <td style={{textAlign:'right'}}>{fmtN(avg(r.recall5))}</td>
                        <td style={{textAlign:'right'}}>{fmtN(avg(r.recall10))}</td>
                        <td style={{textAlign:'right',color:'var(--nv-text-muted)'}}>{fmtN(avg(r.ingest))}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        );
      })()}

      {/* Preset x Dataset Matrix */}
      {presetMatrix.datasets.length > 0 && (
        <div className="card" style={{padding:'16px 20px',marginBottom:'20px'}}>
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'12px'}}>
            <h3 style={{margin:0,fontSize:'14px',fontWeight:600,color:'var(--nv-text-muted)'}}>
              Preset &times; Dataset Matrix (Last Successful Run)
            </h3>
            <button className="btn btn-ghost btn-sm" onClick={downloadPresetMatrixPng} title="Download as PNG">
              <IconDownload /> Download PNG
            </button>
          </div>
          <div style={{overflowX:'auto'}}>
            <table className="runs-table" style={{fontSize:'12px'}}>
              <thead>
                <tr>
                  <th style={{minWidth:'180px'}}>Preset</th>
                  {presetMatrix.datasets.map(ds => (
                    <th key={ds} style={{textAlign:'center',minWidth:'120px'}}>
                      <div>{ds}</div>
                      <div style={{fontSize:'9px',fontWeight:400,color:'var(--nv-text-dim)',marginTop:'2px'}}>PPS / Recall@5</div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(() => {
                  const bestPps = {};
                  const bestRecall = {};
                  presetMatrix.datasets.forEach(ds => {
                    PRESET_MATRIX_PRESETS.forEach(p => {
                      const cell = presetMatrix.map[`${p.key}|||${ds}`];
                      if (cell && cell.pps != null && (bestPps[ds] == null || cell.pps > bestPps[ds])) bestPps[ds] = cell.pps;
                      if (cell && cell.recall5 != null && (bestRecall[ds] == null || cell.recall5 > bestRecall[ds])) bestRecall[ds] = cell.recall5;
                    });
                  });
                  const fmtN = (v) => v != null ? v.toFixed(2) : '\u2014';
                  return PRESET_MATRIX_PRESETS.map((p, ri) => (
                    <tr key={p.key}>
                      <td style={{fontWeight:500,color:'#fff',whiteSpace:'nowrap'}} title={p.desc}>{p.label}</td>
                      {presetMatrix.datasets.map(ds => {
                        const cell = presetMatrix.map[`${p.key}|||${ds}`];
                        const ppsVal = cell ? cell.pps : null;
                        const recallVal = cell ? cell.recall5 : null;
                        const isBestPps = ppsVal != null && ppsVal === bestPps[ds];
                        const isBestRecall = recallVal != null && recallVal === bestRecall[ds];
                        return (
                          <td key={ds} style={{textAlign:'center',verticalAlign:'middle'}}>
                            <div style={{fontWeight:isBestPps?700:400,color:isBestPps?'var(--nv-green)':'#fff',fontSize:'12px'}}>
                              {fmtN(ppsVal)}
                              {isBestPps && <span style={{marginLeft:'4px',fontSize:'9px',color:'var(--nv-green)'}}>BEST</span>}
                            </div>
                            <div style={{fontWeight:isBestRecall?600:400,color:isBestRecall?'var(--nv-green)':'var(--nv-text-muted)',fontSize:'11px',marginTop:'1px'}}>
                              {fmtN(recallVal)}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ));
                })()}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Data Export */}
      <DataExportSection filteredRuns={filteredRuns} filterDataset={filterDataset} filterPreset={filterPreset}
        filterStatus={filterStatus} dateFrom={dateFrom} dateTo={dateTo} />

      {/* Offscreen render area for chart generation */}
      <div style={{position:'absolute',left:'-9999px',top:0,width:'800px'}}>
        {activeCharts.map(c => (
          <div key={c.id} style={{width:'800px',height:'400px',background:'#fff'}}>
            <canvas ref={el => { canvasRefs.current[c.id] = el; }} width={800} height={400}></canvas>
          </div>
        ))}
      </div>

      {/* Preview */}
      {previewImages.length > 0 && (
        <div style={{marginBottom:'20px'}}>
          <h3 style={{fontSize:'16px',fontWeight:600,marginBottom:'12px'}}>Preview</h3>
          <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill, minmax(380px, 1fr))',gap:'16px'}}>
            {previewImages.map(img => (
              <div key={img.id} className="card" style={{padding:'12px',overflow:'hidden'}}>
                <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'8px'}}>
                  <span style={{fontSize:'13px',fontWeight:600,color:'var(--nv-text-muted)'}}>{img.label}</span>
                  <button className="btn btn-ghost btn-sm" onClick={()=>downloadSinglePng(img)} title="Download PNG">
                    <IconDownload />
                  </button>
                </div>
                <img src={img.dataUrl} alt={img.label} style={{width:'100%',borderRadius:'6px',background:'#fff'}} />
              </div>
            ))}
          </div>
        </div>
      )}

      {filteredRuns.length === 0 && !loading && (
        <div className="card" style={{padding:'40px',textAlign:'center'}}>
          <div style={{color:'var(--nv-text-muted)',fontSize:'15px'}}>No runs match the current filters</div>
          <div style={{color:'var(--nv-text-dim)',fontSize:'12px',marginTop:'8px'}}>Adjust filters or trigger some runs first.</div>
        </div>
      )}
    </>
  );
}
