/* ===== App ===== */
function _parseHash() {
  const h = window.location.hash.replace(/^#/, "");
  if (!h) return { view: "runs", runId: null };
  const parts = h.split("/");
  return { view: parts[0] || "runs", runId: parts[1] || null };
}

function App() {
  const initial = _parseHash();
  const [activeView, setActiveView] = useState(initial.view);
  const [pendingDeepLinkRunId, setPendingDeepLinkRunId] = useState(initial.runId);
  const [runs, setRuns] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [filterDataset, setFilterDataset] = useState("");
  const [filterCommit, setFilterCommit] = useState("");
  const [loading, setLoading] = useState(true);
  const [selectedRun, setSelectedRun] = useState(null);
  const [showTrigger, setShowTrigger] = useState(false);
  const [logViewerJobId, setLogViewerJobId] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [runners, setRunners] = useState([]);
  const [runnersLoading, setRunnersLoading] = useState(true);
  const [schedules, setSchedules] = useState([]);
  const [schedulesLoading, setSchedulesLoading] = useState(true);
  const [managedDatasets, setManagedDatasets] = useState([]);
  const [managedDatasetsLoading, setManagedDatasetsLoading] = useState(true);
  const [managedPresets, setManagedPresets] = useState([]);
  const [managedPresetsLoading, setManagedPresetsLoading] = useState(true);
  const [yamlPresets, setYamlPresets] = useState([]);
  const [presetMatrices, setPresetMatrices] = useState([]);
  const [presetMatricesLoading, setPresetMatricesLoading] = useState(true);
  const [alertRules, setAlertRules] = useState([]);
  const [alertRulesLoading, setAlertRulesLoading] = useState(true);
  const [alertEvents, setAlertEvents] = useState([]);
  const [alertEventsLoading, setAlertEventsLoading] = useState(true);
  const [githubRepoUrl, setGithubRepoUrl] = useState("");

  const fetchRuns = useCallback(async () => {
    setLoading(prev => runs.length === 0 ? true : prev);
    const params = new URLSearchParams();
    if (filterDataset) params.set("dataset", filterDataset);
    if (filterCommit) params.set("commit", filterCommit);
    params.set("limit", "200");
    try {
      const res = await fetch(`/api/runs?${params}`);
      setRuns(await res.json());
    } finally { setLoading(false); }
  }, [filterDataset, filterCommit, runs.length]);

  const fetchDatasets = useCallback(async () => {
    const res = await fetch("/api/datasets");
    setDatasets(await res.json());
    try {
      const cfg = await (await fetch("/api/config")).json();
      if (cfg.github_repo_url) setGithubRepoUrl(cfg.github_repo_url);
    } catch {}
  }, []);

  const fetchManagedDatasets = useCallback(async () => {
    setManagedDatasetsLoading(prev => managedDatasets.length === 0 ? true : prev);
    try {
      const res = await fetch("/api/managed-datasets");
      setManagedDatasets(await res.json());
    } catch {}
    finally { setManagedDatasetsLoading(false); }
  }, [managedDatasets.length]);

  const fetchManagedPresets = useCallback(async () => {
    setManagedPresetsLoading(prev => managedPresets.length === 0 ? true : prev);
    try {
      const [mRes, cRes, yRes] = await Promise.all([
        fetch("/api/managed-presets"),
        fetch("/api/config"),
        fetch("/api/yaml-config"),
      ]);
      const managed = await mRes.json();
      const config = await cRes.json();
      const yamlCfg = await yRes.json();
      setManagedPresets(managed);
      const managedNames = new Set(managed.map(p => p.name));
      const yamlPNames = (config.presets || []).filter(n => !managedNames.has(n));
      const yamlPMap = yamlCfg.presets || {};
      setYamlPresets(yamlPNames.map(n => ({ name: n, config: yamlPMap[n] || {} })));
    } catch {}
    finally { setManagedPresetsLoading(false); }
  }, [managedPresets.length]);

  const fetchPresetMatrices = useCallback(async () => {
    setPresetMatricesLoading(prev => presetMatrices.length === 0 ? true : prev);
    try { const res = await fetch("/api/preset-matrices"); setPresetMatrices(await res.json()); } catch {}
    finally { setPresetMatricesLoading(false); }
  }, [presetMatrices.length]);

  const fetchJobs = useCallback(async () => {
    try { const res = await fetch("/api/jobs"); setJobs(await res.json()); } catch {}
  }, []);

  const fetchRunners = useCallback(async () => {
    setRunnersLoading(prev => runners.length === 0 ? true : prev);
    try { const res = await fetch("/api/runners"); setRunners(await res.json()); } catch {}
    finally { setRunnersLoading(false); }
  }, [runners.length]);

  const fetchSchedules = useCallback(async () => {
    setSchedulesLoading(prev => schedules.length === 0 ? true : prev);
    try { const res = await fetch("/api/schedules"); setSchedules(await res.json()); } catch {}
    finally { setSchedulesLoading(false); }
  }, [schedules.length]);

  const fetchAlertRules = useCallback(async () => {
    setAlertRulesLoading(prev => alertRules.length === 0 ? true : prev);
    try { const res = await fetch("/api/alert-rules"); setAlertRules(await res.json()); } catch {}
    finally { setAlertRulesLoading(false); }
  }, [alertRules.length]);

  const fetchAlertEvents = useCallback(async () => {
    setAlertEventsLoading(prev => alertEvents.length === 0 ? true : prev);
    try { const res = await fetch("/api/alert-events?limit=500"); setAlertEvents(await res.json()); } catch {}
    finally { setAlertEventsLoading(false); }
  }, [alertEvents.length]);

  useEffect(() => { fetchRuns(); fetchDatasets(); fetchJobs(); fetchRunners(); fetchSchedules(); fetchManagedDatasets(); fetchManagedPresets(); fetchPresetMatrices(); fetchAlertRules(); fetchAlertEvents(); }, [fetchRuns, fetchDatasets, fetchJobs, fetchRunners, fetchSchedules, fetchManagedDatasets, fetchManagedPresets, fetchPresetMatrices, fetchAlertRules, fetchAlertEvents]);

  // Sync URL hash <-> activeView & deep-link run modal
  useEffect(() => {
    function onHashChange() {
      const { view, runId } = _parseHash();
      setActiveView(view);
      if (view === "runs" && runId) {
        fetch(`/api/runs/${runId}`).then(r => r.json()).then(setSelectedRun).catch(() => {});
      } else {
        setSelectedRun(null);
      }
    }
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  // Push hash when activeView changes (but not on initial mount from hash)
  const isFirstRender = useRef(true);
  useEffect(() => {
    if (isFirstRender.current) { isFirstRender.current = false; return; }
    const currentHash = _parseHash();
    if (currentHash.view !== activeView) {
      window.history.pushState(null, "", `#${activeView}`);
    }
  }, [activeView]);

  // Open deep-linked run after initial data load
  useEffect(() => {
    if (pendingDeepLinkRunId && !loading) {
      openDetail(pendingDeepLinkRunId);
      setPendingDeepLinkRunId(null);
    }
  }, [pendingDeepLinkRunId, loading]);

  // Auto-refresh the active view every 10 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      fetchJobs();
      fetchAlertEvents();
      if (activeView === "runs")       { fetchRuns(); }
      if (activeView === "datasets")   { fetchManagedDatasets(); }
      if (activeView === "presets")    { fetchManagedPresets(); fetchPresetMatrices(); }
      if (activeView === "runners")    { fetchRunners(); }
      if (activeView === "scheduling") { fetchSchedules(); }
      if (activeView === "alerts")     { fetchAlertRules(); fetchAlertEvents(); }
    }, 10000);
    return () => clearInterval(interval);
  }, [activeView, fetchRuns, fetchJobs, fetchRunners, fetchSchedules, fetchManagedDatasets, fetchManagedPresets, fetchPresetMatrices, fetchAlertRules, fetchAlertEvents]);

  // Fast polling (3s) when jobs are actively running
  useEffect(() => {
    const hasActive = jobs.some(j => j.status==="running" || j.status==="pending" || j.status==="cancelling");
    if (!hasActive) return;
    const interval = setInterval(() => { fetchJobs(); fetchRuns(); }, 3000);
    return () => clearInterval(interval);
  }, [jobs, fetchJobs, fetchRuns]);

  async function openDetail(id) {
    try {
      const res = await fetch(`/api/runs/${id}`);
      const run = await res.json();
      setSelectedRun(run);
      const runId = run.run_id || run.id || id;
      if (window.location.hash !== `#runs/${runId}`) {
        window.history.pushState(null, "", `#runs/${runId}`);
      }
    } catch {}
  }

  function closeDetail() {
    setSelectedRun(null);
    window.history.pushState(null, "", "#runs");
  }

  async function deleteRun(id) {
    try {
      await fetch(`/api/runs/${id}`, { method: "DELETE" });
      closeDetail();
      fetchRuns();
    } catch {}
  }

  const viewTitles = { runs: "Runs", analytics: "Analytics", reporting: "Reporting", datasets: "Datasets", presets: "Presets", runners: "Runners", scheduling: "Scheduling", alerts: "Alerts", ingestion: "Ingestion", retrieval: "Retrieval", models: "Models", settings: "Settings" };

  const activeJobCount = jobs.filter(j => j.status==="running" || j.status==="pending" || j.status==="cancelling").length;

  const footerText = useMemo(() => {
    if (activeView === "runs") {
      const base = loading ? "Loading\u2026" : `${runs.length} run${runs.length!==1?'s':''} found`;
      return activeJobCount > 0 ? `${base} \u00B7 ${activeJobCount} job${activeJobCount!==1?'s':''} active` : base;
    }
    if (activeView === "analytics") {
      return loading ? "Loading\u2026" : `${runs.length} run${runs.length!==1?'s':''} available for analysis`;
    }
    if (activeView === "datasets") {
      return managedDatasetsLoading ? "Loading\u2026" : `${managedDatasets.length} dataset${managedDatasets.length!==1?'s':''} configured`;
    }
    if (activeView === "presets") {
      if (managedPresetsLoading) return "Loading\u2026";
      const parts = [`${managedPresets.length} preset${managedPresets.length!==1?'s':''}`];
      if (presetMatrices.length > 0) parts.push(`${presetMatrices.length} matrix${presetMatrices.length!==1?'es':''}`);
      return parts.join(", ") + " configured";
    }
    if (activeView === "runners") return runnersLoading ? "Loading\u2026" : `${runners.length} runner${runners.length!==1?'s':''} registered`;
    if (activeView === "scheduling") return schedulesLoading ? "Loading\u2026" : `${schedules.length} schedule${schedules.length!==1?'s':''} configured`;
    if (activeView === "alerts") {
      const unack = alertEvents.filter(e=>!e.acknowledged).length;
      return alertEventsLoading ? "Loading\u2026" : `${alertRules.length} rule${alertRules.length!==1?'s':''}, ${alertEvents.length} event${alertEvents.length!==1?'s':''} (${unack} unacknowledged)`;
    }
    if (activeView === "settings") {
      return "Manage portal deployment and system settings";
    }
    if (activeView === "ingestion") {
      return "Upload documents and run custom ingestion jobs";
    }
    if (activeView === "retrieval") {
      const successCount = runs.filter(r => r.success === 1).length;
      return `Query against LanceDB from ${successCount} successful run${successCount !== 1 ? 's' : ''}`;
    }
    if (activeView === "models") {
      return "Send test payloads to HuggingFace models and inspect responses";
    }
    if (activeView === "reporting") {
      return loading ? "Loading\u2026" : `${runs.length} run${runs.length!==1?'s':''} available for reporting`;
    }
    return "";
  }, [activeView, runs.length, activeJobCount, managedDatasets.length, managedPresets.length, presetMatrices.length, yamlPresets.length, runners.length, schedules.length, alertRules.length, alertEvents.length, alertEvents, loading, managedDatasetsLoading, managedPresetsLoading, runnersLoading, schedulesLoading, alertRulesLoading, alertEventsLoading]);

  return (
    <div className="app-layout">
      <Sidebar activeView={activeView} onNavigate={setActiveView} alertBadgeCount={alertEvents.filter(e=>!e.acknowledged).length} githubRepoUrl={githubRepoUrl} />
      <div className="main-wrapper">
        <Header title={viewTitles[activeView] || "Portal"}>
          {activeView==="runs" && (
            <button className="btn btn-ghost btn-icon" onClick={()=>{fetchRuns();fetchJobs();}} title="Refresh"><IconRefresh /></button>
          )}
          {activeView==="analytics" && (
            <button className="btn btn-ghost btn-icon" onClick={fetchRuns} title="Refresh"><IconRefresh /></button>
          )}
          {activeView==="datasets" && (
            <button className="btn btn-ghost btn-icon" onClick={fetchManagedDatasets} title="Refresh"><IconRefresh /></button>
          )}
          {activeView==="presets" && (
            <button className="btn btn-ghost btn-icon" onClick={()=>{fetchManagedPresets();fetchPresetMatrices();}} title="Refresh"><IconRefresh /></button>
          )}
          {activeView==="runners" && (
            <button className="btn btn-ghost btn-icon" onClick={fetchRunners} title="Refresh"><IconRefresh /></button>
          )}
          {activeView==="scheduling" && (
            <button className="btn btn-ghost btn-icon" onClick={fetchSchedules} title="Refresh"><IconRefresh /></button>
          )}
          {activeView==="alerts" && (
            <button className="btn btn-ghost btn-icon" onClick={()=>{fetchAlertRules();fetchAlertEvents();}} title="Refresh"><IconRefresh /></button>
          )}
        </Header>
        <div className="main-content">
          {activeView==="runs" && (
            <RunsView runs={runs} datasets={datasets} loading={loading}
              filterDataset={filterDataset} setFilterDataset={setFilterDataset}
              filterCommit={filterCommit} setFilterCommit={setFilterCommit}
              onRefresh={()=>{fetchRuns();fetchJobs();}} onSelectRun={openDetail}
              onDeleteRun={deleteRun}
              onTrigger={()=>setShowTrigger(true)} jobs={jobs} runners={runners} githubRepoUrl={githubRepoUrl}
              onViewLogs={setLogViewerJobId} />
          )}
          {activeView==="analytics" && (
            <AnalyticsView runs={runs} datasets={datasets} loading={loading} onRefresh={fetchRuns} />
          )}
          {activeView==="reporting" && (
            <ReportingView runs={runs} datasets={datasets} loading={loading} />
          )}
          {activeView==="datasets" && (
            <DatasetsView managedDatasets={managedDatasets}
              loading={managedDatasetsLoading} onRefresh={fetchManagedDatasets} runners={runners} />
          )}
          {activeView==="presets" && (
            <PresetsView managedPresets={managedPresets} yamlPresets={yamlPresets}
              loading={managedPresetsLoading} onRefresh={()=>{fetchManagedPresets();fetchPresetMatrices();}}
              presetMatrices={presetMatrices} presetMatricesLoading={presetMatricesLoading} />
          )}
          {activeView==="runners" && <RunnersView runners={runners} loading={runnersLoading} onRefresh={fetchRunners} githubRepoUrl={githubRepoUrl} />}
          {activeView==="scheduling" && <SchedulingView schedules={schedules} loading={schedulesLoading} onRefresh={fetchSchedules} runners={runners} />}
          {activeView==="alerts" && (
            <AlertsView alertRules={alertRules} alertEvents={alertEvents}
              alertRulesLoading={alertRulesLoading} alertEventsLoading={alertEventsLoading}
              onRefresh={()=>{fetchAlertRules();fetchAlertEvents();}}
              onSelectRun={openDetail} githubRepoUrl={githubRepoUrl} />
          )}
          {activeView==="settings" && (
            <SettingsView />
          )}
          {activeView==="ingestion" && (
            <IngestionView jobs={jobs} onViewLogs={setLogViewerJobId} />
          )}
          {activeView==="retrieval" && (
            <RetrievalView runs={runs} />
          )}
          {activeView==="models" && (
            <ModelsView />
          )}
        </div>
        <Footer>
          <span>{footerText}</span>
          <span>Harness Portal</span>
        </Footer>
      </div>
      {selectedRun && <RunDetailModal run={selectedRun} onClose={closeDetail} onDelete={deleteRun} githubRepoUrl={githubRepoUrl} />}
      {showTrigger && <TriggerModal onClose={()=>setShowTrigger(false)} onTriggered={()=>{fetchJobs();setTimeout(fetchJobs,1000);}} />}
      {logViewerJobId && <LogViewerModal jobId={logViewerJobId} onClose={()=>setLogViewerJobId(null)} />}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
