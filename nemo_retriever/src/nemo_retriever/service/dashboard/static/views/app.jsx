/* App — root component with hash-based routing.
 *
 * Two route shapes are supported:
 *   #overview / #jobs / #vdb         → top-level views
 *   #job/<job_id>                    → drill-down to JobDetailView
 *
 * Routing intentionally lives here (single source of truth) rather
 * than in each view so deep-linking works (paste a #job/<id> URL and
 * the detail page renders on first paint).
 */

function parseHash() {
  const raw = window.location.hash.replace(/^#/, '');
  if (raw.startsWith('job/')) {
    const jobId = raw.slice(4);
    if (jobId) return { view: 'job_detail', jobId };
  }
  if (['overview', 'jobs', 'vdb'].includes(raw)) {
    return { view: raw, jobId: null };
  }
  return { view: 'overview', jobId: null };
}

function App() {
  const [route, setRoute] = React.useState(parseHash);

  React.useEffect(() => {
    const onHash = () => setRoute(parseHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const navigate = (v) => {
    if (typeof v === 'string') {
      window.location.hash = v;
      setRoute({ view: v, jobId: null });
    } else if (v && v.view) {
      const hash = v.view === 'job_detail' && v.jobId ? `job/${v.jobId}` : v.view;
      window.location.hash = hash;
      setRoute(v);
    }
  };

  const openJob = (jobId) => navigate({ view: 'job_detail', jobId });
  const backToJobs = () => navigate('jobs');

  let content = null;
  if (route.view === 'overview' && typeof OverviewView !== 'undefined') {
    content = React.createElement(OverviewView);
  } else if (route.view === 'jobs' && typeof JobsView !== 'undefined') {
    content = React.createElement(JobsView, { onOpenJob: openJob });
  } else if (route.view === 'job_detail' && typeof JobDetailView !== 'undefined' && route.jobId) {
    content = React.createElement(JobDetailView, { jobId: route.jobId, onBack: backToJobs });
  } else if (route.view === 'vdb' && typeof VdbView !== 'undefined') {
    content = React.createElement(VdbView);
  } else {
    content = React.createElement('div', { className: 'empty-state' }, 'Loading view…');
  }

  // Map the detail route back onto the "jobs" sidebar entry so the
  // active state stays consistent while drilling down.
  const sidebarView = route.view === 'job_detail' ? 'jobs' : route.view;
  return React.createElement(Layout, {
    view: sidebarView,
    onNavigate: navigate,
    breadcrumb: route.view === 'job_detail'
      ? { label: `Job ${route.jobId.substring(0, 12)}…`, parent: 'Jobs' }
      : null,
  }, content);
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(React.createElement(App));
