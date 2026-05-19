/* Layout — sidebar navigation, header, view container */

const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', icon: '◉' },
  { id: 'jobs',     label: 'Job Tracker', icon: '▶' },
  { id: 'vdb',      label: 'VDB Explorer', icon: '⬡' },
];

const VIEW_TITLES = {
  overview: 'Cluster Overview',
  jobs: 'Job Tracker',
  vdb: 'VDB Explorer',
};

function Layout({ view, onNavigate, children }) {
  return React.createElement('div', { className: 'app-layout' },
    React.createElement('aside', { className: 'sidebar' },
      React.createElement('div', { className: 'sidebar-logo' },
        React.createElement('img', {
          src: '/v1/dashboard/static/nvidia-logo.svg',
          alt: 'NVIDIA',
        }),
        React.createElement('span', null, 'Dashboard'),
      ),
      React.createElement('nav', { className: 'sidebar-nav' },
        NAV_ITEMS.map(item =>
          React.createElement('a', {
            key: item.id,
            className: `nav-item${view === item.id ? ' active' : ''}`,
            href: `#${item.id}`,
            onClick: (e) => { e.preventDefault(); onNavigate(item.id); },
          },
            React.createElement('span', null, item.icon),
            React.createElement('span', null, item.label),
          )
        )
      ),
    ),
    React.createElement('div', { className: 'main-area' },
      React.createElement('header', { className: 'header' },
        VIEW_TITLES[view] || 'Dashboard'
      ),
      React.createElement('main', { className: 'content' }, children),
    ),
  );
}
