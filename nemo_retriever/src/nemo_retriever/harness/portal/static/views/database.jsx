/* ===== Database Management View ===== */
function DatabaseView() {
  const [dbInfo, setDbInfo] = useState(null);
  const [dbLoading, setDbLoading] = useState(true);
  const [backups, setBackups] = useState([]);
  const [backupsLoading, setBackupsLoading] = useState(true);

  const [storageType, setStorageType] = useState("local");
  const [destination, setDestination] = useState("");
  const [backupLabel, setBackupLabel] = useState("");
  const [backupBusy, setBackupBusy] = useState(false);
  const [backupResult, setBackupResult] = useState(null);
  const [backupError, setBackupError] = useState("");

  const [restoreConfirm, setRestoreConfirm] = useState(null);
  const [restoreBusy, setRestoreBusy] = useState(false);
  const [restoreResult, setRestoreResult] = useState(null);
  const [restoreError, setRestoreError] = useState("");

  const [exportBusy, setExportBusy] = useState(false);

  const [deleteConfirm, setDeleteConfirm] = useState(null);

  async function fetchDbInfo() {
    setDbLoading(true);
    try {
      const res = await fetch("/api/database/info");
      setDbInfo(await res.json());
    } catch (err) {
      console.error("Failed to fetch DB info:", err);
    } finally {
      setDbLoading(false);
    }
  }

  async function fetchBackups() {
    setBackupsLoading(true);
    try {
      const res = await fetch("/api/database/backups");
      setBackups(await res.json());
    } catch (err) {
      console.error("Failed to fetch backups:", err);
    } finally {
      setBackupsLoading(false);
    }
  }

  function refresh() {
    fetchDbInfo();
    fetchBackups();
  }

  useEffect(() => { fetchDbInfo(); fetchBackups(); }, []);

  async function handleBackup() {
    if (!destination.trim()) return;
    setBackupBusy(true);
    setBackupResult(null);
    setBackupError("");
    try {
      const res = await fetch("/api/database/backup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          label: backupLabel.trim() || null,
          storage_type: storageType,
          destination: destination.trim(),
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setBackupResult(data);
      setBackupLabel("");
      fetchBackups();
      fetchDbInfo();
    } catch (err) {
      setBackupError(err.message);
    } finally {
      setBackupBusy(false);
    }
  }

  async function handleRestore(backup) {
    setRestoreConfirm(null);
    setRestoreBusy(true);
    setRestoreResult(null);
    setRestoreError("");
    try {
      const res = await fetch("/api/database/restore", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ backup_id: backup.id }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setRestoreResult(data);
      fetchDbInfo();
      fetchBackups();
    } catch (err) {
      setRestoreError(err.message);
    } finally {
      setRestoreBusy(false);
    }
  }

  async function handleExport(source, backupId) {
    setExportBusy(true);
    try {
      const params = new URLSearchParams({ source });
      if (backupId != null) params.set("backup_id", backupId);
      const res = await fetch(`/api/database/export-json?${params}`);
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      const blob = await res.blob();
      const cd = res.headers.get("Content-Disposition") || "";
      const match = cd.match(/filename="?([^"]+)"?/);
      const filename = match ? match[1] : "harness_db_export.json";
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (err) {
      alert("Export failed: " + err.message);
    } finally {
      setExportBusy(false);
    }
  }

  async function handleDeleteBackup(backup, deleteFile) {
    setDeleteConfirm(null);
    try {
      const params = new URLSearchParams();
      if (deleteFile) params.set("delete_file", "true");
      const res = await fetch(`/api/database/backups/${backup.id}?${params}`, { method: "DELETE" });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      fetchBackups();
    } catch (err) {
      alert("Delete failed: " + err.message);
    }
  }

  function formatBytes(bytes) {
    if (bytes == null) return "—";
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  }

  function formatTimestamp(ts) {
    if (!ts) return "—";
    try {
      const d = new Date(ts);
      return d.toLocaleString();
    } catch { return ts; }
  }

  const labelStyle = { display: "block", fontSize: "11px", fontWeight: 600, color: "var(--nv-text-dim)", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "6px" };

  if (dbLoading && !dbInfo) {
    return (
      <div className="card" style={{ padding: "60px", textAlign: "center" }}>
        <span className="spinner spinner-lg" style={{ display: "block", margin: "0 auto 16px" }}></span>
        <div style={{ color: "var(--nv-text-muted)", fontSize: "14px" }}>Loading database information…</div>
      </div>
    );
  }

  return (
    <>
      {/* Database Status */}
      <div className="card" style={{ padding: "24px", marginBottom: "20px" }}>
        <div className="section-title" style={{ marginBottom: "16px" }}>Database Status</div>
        {dbInfo && (
          <>
            <div style={{
              padding: "12px 16px", borderRadius: "8px", marginBottom: "20px",
              background: "rgba(118,185,0,0.04)", border: "1px solid rgba(118,185,0,0.15)",
              display: "flex", alignItems: "center", gap: "12px", flexWrap: "wrap",
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", minWidth: 0, flex: 1 }}>
                <IconHardDrive />
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontSize: "11px", fontWeight: 600, color: "var(--nv-text-dim)", textTransform: "uppercase", letterSpacing: "0.04em" }}>Active Database File</div>
                  <div className="mono" style={{ fontSize: "13px", color: "var(--nv-green)", fontWeight: 600, wordBreak: "break-all", marginTop: "2px" }}>{dbInfo.db_path}</div>
                </div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: "11px", fontWeight: 600, color: "var(--nv-text-dim)", textTransform: "uppercase", letterSpacing: "0.04em" }}>File Size</div>
                <div style={{ fontSize: "14px", color: "#fff", fontWeight: 600, marginTop: "2px" }}>{formatBytes(dbInfo.size_bytes)}</div>
              </div>
            </div>
            <div style={labelStyle}>Table Row Counts</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: "8px" }}>
              {Object.entries(dbInfo.table_counts || {}).map(([table, count]) => (
                <div key={table} style={{
                  padding: "8px 12px", borderRadius: "8px",
                  background: "rgba(255,255,255,0.02)", border: "1px solid var(--nv-border)",
                  display: "flex", justifyContent: "space-between", alignItems: "center",
                }}>
                  <span className="mono" style={{ fontSize: "11px", color: "var(--nv-text-muted)" }}>{table}</span>
                  <span style={{ fontSize: "13px", fontWeight: 600, color: count > 0 ? "var(--nv-green)" : "var(--nv-text-dim)" }}>{count}</span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      {/* Backup & Restore */}
      <div className="card" style={{ padding: "24px", marginBottom: "20px" }}>
        <div className="section-title" style={{ marginBottom: "6px" }}>Backup & Restore</div>
        <div style={{ fontSize: "12px", color: "var(--nv-text-dim)", lineHeight: "1.6", marginBottom: "20px" }}>
          Create a snapshot of the current database. Backups can be saved to a local directory or an S3 bucket.
          You can restore from any backup to replace the current database (a safety backup is created automatically).
        </div>

        {/* Create Backup Form */}
        <div style={{
          padding: "16px", borderRadius: "10px", marginBottom: "20px",
          background: "rgba(255,255,255,0.015)", border: "1px solid var(--nv-border)",
        }}>
          <div style={{ fontSize: "13px", fontWeight: 600, color: "#fff", marginBottom: "14px" }}>Create Backup</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "14px", marginBottom: "14px" }}>
            <div>
              <label style={labelStyle}>Storage Type</label>
              <div style={{ display: "flex", gap: "6px" }}>
                <button className="btn btn-sm" onClick={() => setStorageType("local")} style={{
                  flex: 1, justifyContent: "center",
                  background: storageType === "local" ? "rgba(118,185,0,0.15)" : "transparent",
                  color: storageType === "local" ? "var(--nv-green)" : "var(--nv-text-dim)",
                  border: `1px solid ${storageType === "local" ? "rgba(118,185,0,0.3)" : "var(--nv-border)"}`,
                }}>Local</button>
                <button className="btn btn-sm" onClick={() => setStorageType("s3")} style={{
                  flex: 1, justifyContent: "center",
                  background: storageType === "s3" ? "rgba(100,180,255,0.15)" : "transparent",
                  color: storageType === "s3" ? "#64b4ff" : "var(--nv-text-dim)",
                  border: `1px solid ${storageType === "s3" ? "rgba(100,180,255,0.3)" : "var(--nv-border)"}`,
                }}>S3</button>
              </div>
            </div>
            <div>
              <label style={labelStyle}>{storageType === "s3" ? "S3 URI" : "Destination Directory"}</label>
              <input className="input" style={{ width: "100%" }} value={destination}
                onChange={e => setDestination(e.target.value)}
                placeholder={storageType === "s3" ? "s3://my-bucket/backups" : "/path/to/backup/dir"} />
            </div>
            <div>
              <label style={labelStyle}>Label (optional)</label>
              <input className="input" style={{ width: "100%" }} value={backupLabel}
                onChange={e => setBackupLabel(e.target.value)}
                placeholder="e.g. pre-migration" />
            </div>
          </div>
          <button className="btn btn-primary" onClick={handleBackup}
            disabled={backupBusy || !destination.trim()}
            style={{ justifyContent: "center" }}>
            {backupBusy ? <><span className="spinner" style={{ marginRight: "6px" }}></span>Backing up…</> : <><IconDownload /> Create Backup</>}
          </button>
        </div>

        {/* Backup result / error */}
        {backupResult && (
          <div style={{ marginBottom: "16px", padding: "12px 16px", borderRadius: "8px", background: "rgba(118,185,0,0.05)", border: "1px solid rgba(118,185,0,0.2)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <IconCheck />
              <span style={{ fontSize: "13px", fontWeight: 600, color: "var(--nv-green)" }}>Backup created successfully</span>
            </div>
            <div className="mono" style={{ fontSize: "11px", color: "var(--nv-text-muted)", marginTop: "6px" }}>{backupResult.path}</div>
          </div>
        )}
        {backupError && (
          <div style={{ marginBottom: "16px", padding: "12px 16px", borderRadius: "8px", background: "rgba(255,50,50,0.08)", border: "1px solid rgba(255,50,50,0.2)" }}>
            <div style={{ fontSize: "13px", fontWeight: 600, color: "#ff5050", marginBottom: "4px" }}>Backup Failed</div>
            <div className="mono" style={{ fontSize: "12px", color: "#ff5050" }}>{backupError}</div>
          </div>
        )}

        {/* Restore result / error */}
        {restoreResult && (
          <div style={{ marginBottom: "16px", padding: "12px 16px", borderRadius: "8px", background: "rgba(118,185,0,0.05)", border: "1px solid rgba(118,185,0,0.2)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <IconCheck />
              <span style={{ fontSize: "13px", fontWeight: 600, color: "var(--nv-green)" }}>{restoreResult.message}</span>
            </div>
            <div className="mono" style={{ fontSize: "11px", color: "var(--nv-text-muted)", marginTop: "6px" }}>
              Safety backup: {restoreResult.safety_backup}
            </div>
          </div>
        )}
        {restoreError && (
          <div style={{ marginBottom: "16px", padding: "12px 16px", borderRadius: "8px", background: "rgba(255,50,50,0.08)", border: "1px solid rgba(255,50,50,0.2)" }}>
            <div style={{ fontSize: "13px", fontWeight: 600, color: "#ff5050", marginBottom: "4px" }}>Restore Failed</div>
            <div className="mono" style={{ fontSize: "12px", color: "#ff5050" }}>{restoreError}</div>
          </div>
        )}

        {/* Backups Table */}
        <div style={{ fontSize: "13px", fontWeight: 600, color: "#fff", marginBottom: "12px" }}>
          Backup History ({backups.length})
        </div>
        {backupsLoading && backups.length === 0 ? (
          <div style={{ padding: "30px", textAlign: "center" }}>
            <span className="spinner"></span>
          </div>
        ) : backups.length === 0 ? (
          <div style={{ padding: "30px", textAlign: "center", color: "var(--nv-text-dim)", fontSize: "13px" }}>
            No backups yet. Create your first backup above.
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table className="data-table" style={{ width: "100%" }}>
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Label</th>
                  <th>Storage</th>
                  <th>Path</th>
                  <th>Size</th>
                  <th style={{ textAlign: "right" }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {backups.map(b => (
                  <tr key={b.id}>
                    <td style={{ whiteSpace: "nowrap" }}>{formatTimestamp(b.timestamp)}</td>
                    <td>{b.label || <span style={{ color: "var(--nv-text-dim)" }}>—</span>}</td>
                    <td>
                      <span style={{
                        padding: "2px 8px", borderRadius: "4px", fontSize: "10px", fontWeight: 600,
                        textTransform: "uppercase", letterSpacing: "0.05em",
                        background: b.storage_type === "s3" ? "rgba(100,180,255,0.1)" : "rgba(118,185,0,0.1)",
                        color: b.storage_type === "s3" ? "#64b4ff" : "var(--nv-green)",
                      }}>{b.storage_type}</span>
                    </td>
                    <td><span className="mono" style={{ fontSize: "11px", color: "var(--nv-text-muted)", wordBreak: "break-all" }}>{b.path}</span></td>
                    <td style={{ whiteSpace: "nowrap" }}>{formatBytes(b.size_bytes)}</td>
                    <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                      <div style={{ display: "flex", gap: "4px", justifyContent: "flex-end" }}>
                        <button className="btn btn-sm" title="Restore from this backup"
                          disabled={restoreBusy}
                          onClick={() => setRestoreConfirm(b)}
                          style={{ fontSize: "10px", padding: "2px 8px" }}>
                          <IconUpload /> Restore
                        </button>
                        {b.storage_type === "local" && (
                          <button className="btn btn-sm" title="Export as JSON"
                            disabled={exportBusy}
                            onClick={() => handleExport("backup", b.id)}
                            style={{ fontSize: "10px", padding: "2px 8px" }}>
                            <IconFileText /> JSON
                          </button>
                        )}
                        <button className="btn btn-sm" title="Delete backup"
                          onClick={() => setDeleteConfirm(b)}
                          style={{ fontSize: "10px", padding: "2px 8px", color: "#ff5050" }}>
                          <IconTrash />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Export */}
      <div className="card" style={{ padding: "24px", marginBottom: "20px" }}>
        <div className="section-title" style={{ marginBottom: "6px" }}>Export</div>
        <div style={{ fontSize: "12px", color: "var(--nv-text-dim)", lineHeight: "1.6", marginBottom: "16px" }}>
          Export all raw data from the current database as a JSON file. This includes every table (runs, datasets,
          presets, runners, schedules, jobs, alerts, and more). You can also export from any local backup using
          the JSON button in the backup table above.
        </div>
        <button className="btn btn-primary" onClick={() => handleExport("current")}
          disabled={exportBusy}
          style={{ justifyContent: "center" }}>
          {exportBusy ? <><span className="spinner" style={{ marginRight: "6px" }}></span>Exporting…</> : <><IconDownload /> Export Current Database to JSON</>}
        </button>
      </div>

      {/* Restore Confirmation Modal */}
      {restoreConfirm && (
        <div className="modal-overlay" onClick={() => setRestoreConfirm(null)}>
          <div className="modal-content" style={{ maxWidth: "500px" }} onClick={e => e.stopPropagation()}>
            <div className="modal-head">
              <h2 style={{ fontSize: "16px", fontWeight: 700, color: "#fff" }}>Restore Database</h2>
              <button className="btn btn-ghost btn-icon" onClick={() => setRestoreConfirm(null)} style={{ borderRadius: "50%" }}><IconX /></button>
            </div>
            <div style={{ padding: "24px" }}>
              <div style={{ padding: "10px 14px", borderRadius: "8px", background: "rgba(255,184,77,0.08)", border: "1px solid rgba(255,184,77,0.25)", color: "#ffb84d", fontSize: "12px", marginBottom: "16px", lineHeight: "1.6" }}>
                <strong>Warning:</strong> This will overwrite the current database. A safety backup of the current state will be created automatically before restoring.
              </div>
              <div style={{ fontSize: "13px", color: "var(--nv-text-muted)", lineHeight: "1.8", marginBottom: "16px" }}>
                <div>Restoring from:</div>
                <div className="mono" style={{ color: "#fff", fontSize: "12px", wordBreak: "break-all", marginTop: "4px" }}>{restoreConfirm.path}</div>
                {restoreConfirm.label && (
                  <div style={{ marginTop: "6px" }}>Label: <span style={{ color: "#fff" }}>{restoreConfirm.label}</span></div>
                )}
                <div style={{ marginTop: "6px" }}>Created: <span style={{ color: "#fff" }}>{formatTimestamp(restoreConfirm.timestamp)}</span></div>
              </div>
            </div>
            <div className="modal-foot">
              <button className="btn btn-secondary" onClick={() => setRestoreConfirm(null)}>Cancel</button>
              <button className="btn" onClick={() => handleRestore(restoreConfirm)}
                style={{ flex: 1, justifyContent: "center", background: "rgba(255,184,77,0.15)", color: "#ffb84d", border: "1px solid rgba(255,184,77,0.3)", fontWeight: 600 }}>
                Restore Database
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {deleteConfirm && (
        <div className="modal-overlay" onClick={() => setDeleteConfirm(null)}>
          <div className="modal-content" style={{ maxWidth: "440px" }} onClick={e => e.stopPropagation()}>
            <div className="modal-head">
              <h2 style={{ fontSize: "16px", fontWeight: 700, color: "#fff" }}>Delete Backup</h2>
              <button className="btn btn-ghost btn-icon" onClick={() => setDeleteConfirm(null)} style={{ borderRadius: "50%" }}><IconX /></button>
            </div>
            <div style={{ padding: "24px" }}>
              <div style={{ fontSize: "13px", color: "var(--nv-text-muted)", lineHeight: "1.8", marginBottom: "16px" }}>
                <div className="mono" style={{ color: "#fff", fontSize: "12px", wordBreak: "break-all" }}>{deleteConfirm.path}</div>
                {deleteConfirm.label && <div style={{ marginTop: "6px" }}>Label: {deleteConfirm.label}</div>}
              </div>
              {deleteConfirm.storage_type === "local" && (
                <div style={{ fontSize: "12px", color: "var(--nv-text-dim)", lineHeight: "1.6", marginBottom: "10px" }}>
                  Choose whether to also delete the backup file from disk, or only remove the record from the portal.
                </div>
              )}
            </div>
            <div className="modal-foot" style={{ gap: "8px" }}>
              <button className="btn btn-secondary" onClick={() => setDeleteConfirm(null)}>Cancel</button>
              <button className="btn" onClick={() => handleDeleteBackup(deleteConfirm, false)}
                style={{ justifyContent: "center", background: "rgba(255,50,50,0.1)", color: "#ff5050", border: "1px solid rgba(255,50,50,0.25)", fontWeight: 600 }}>
                Remove Record Only
              </button>
              {deleteConfirm.storage_type === "local" && (
                <button className="btn" onClick={() => handleDeleteBackup(deleteConfirm, true)}
                  style={{ justifyContent: "center", background: "rgba(255,50,50,0.15)", color: "#ff5050", border: "1px solid rgba(255,50,50,0.35)", fontWeight: 600 }}>
                  Delete Record + File
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
