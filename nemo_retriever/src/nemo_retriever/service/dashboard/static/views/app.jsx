/* App — root component with hash-based routing */

function App() {
  const [view, setView] = React.useState(() => {
    const hash = window.location.hash.replace('#', '');
    return ['overview', 'jobs', 'vdb'].includes(hash) ? hash : 'overview';
  });

  React.useEffect(() => {
    const onHash = () => {
      const hash = window.location.hash.replace('#', '');
      if (['overview', 'jobs', 'vdb'].includes(hash)) setView(hash);
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const navigate = (v) => {
    window.location.hash = v;
    setView(v);
  };

  let content = null;
  if (view === 'overview' && typeof OverviewView !== 'undefined') {
    content = React.createElement(OverviewView);
  } else if (view === 'jobs' && typeof JobsView !== 'undefined') {
    content = React.createElement(JobsView);
  } else if (view === 'vdb' && typeof VdbView !== 'undefined') {
    content = React.createElement(VdbView);
  } else {
    content = React.createElement('div', { className: 'empty-state' }, 'Loading view…');
  }

  return React.createElement(Layout, { view, onNavigate: navigate }, content);
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(React.createElement(App));
