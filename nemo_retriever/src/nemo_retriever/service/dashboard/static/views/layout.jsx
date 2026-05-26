/* Layout — sidebar navigation, header (with optional breadcrumb), view container.
 *
 * The header displays either ``VIEW_TITLES[view]`` (top-level routes)
 * or a breadcrumb ``parent / label`` when drilling down into a
 * sub-view (e.g. Jobs / Job <id>). The sidebar always reflects the
 * top-level view so drill-down pages don't lose their nav anchor.
 */

const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', icon: '◉' },
  { id: 'jobs',     label: 'Jobs', icon: '▶' },
  { id: 'vdb',      label: 'VDB Explorer', icon: '⬡' },
];

const VIEW_TITLES = {
  overview: 'Cluster Overview',
  jobs: 'Jobs',
  vdb: 'VDB Explorer',
};

function Layout({ view, onNavigate, breadcrumb, children }) {
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
        breadcrumb
          ? React.createElement(React.Fragment, null,
              React.createElement('a', {
                href: `#${view}`,
                onClick: (e) => { e.preventDefault(); onNavigate(view); },
                style: { color: 'var(--nv-text-muted)', textDecoration: 'none' },
              }, breadcrumb.parent),
              React.createElement('span', { style: { margin: '0 8px', color: 'var(--nv-text-muted)' } }, '/'),
              React.createElement('span', null, breadcrumb.label),
            )
          : (VIEW_TITLES[view] || 'Dashboard')
      ),
      React.createElement('main', { className: 'content' }, children),
    ),
  );
}
