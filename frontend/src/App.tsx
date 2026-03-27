import React, { useState, createContext, useContext } from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import Overview from './pages/Overview';
import UserDetail from './pages/UserDetail';
import GraphAnalysis from './pages/GraphAnalysis';
import BatchPredict from './pages/BatchPredict';

const SidebarCtx = createContext({ expanded: true, toggle: () => {} });

const navItems = [
  { to: '/', label: '風險總覽', short: '總', end: true },
  { to: '/batch', label: '批次預測', short: '批' },
];

function Sidebar() {
  const { expanded, toggle } = useContext(SidebarCtx);

  return (
    <nav
      className={`fixed left-0 top-0 z-30 flex h-screen flex-col border-r border-slate-800/30 bg-sidebar-bg transition-[width] duration-200 ease-in-out ${
        expanded ? 'w-52' : 'w-[60px]'
      }`}
    >
      {/* Header — click to toggle */}
      <button
        onClick={toggle}
        className={`flex items-center border-b border-slate-700/40 py-4 text-left transition-colors hover:bg-sidebar-hover ${
          expanded ? 'gap-2.5 px-3.5' : 'justify-center px-0'
        }`}
        title={expanded ? '收合側邊欄' : '展開側邊欄'}
      >
        <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg bg-brand-600 text-xs font-bold text-white">
          A4
        </div>
        {expanded && (
          <span className="truncate text-sm font-semibold tracking-wide text-sidebar-text-active">
            A4T Dashboard
          </span>
        )}
      </button>

      {/* Nav links */}
      <div className="mt-3 flex flex-col gap-1 px-2">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            title={expanded ? undefined : item.label}
            className={({ isActive }) =>
              `group flex items-center rounded-lg text-sm transition-colors duration-150 ${
                expanded
                  ? 'px-3 py-2.5'
                  : 'mx-auto h-9 w-9 justify-center p-0'
              } ${
                isActive
                  ? 'bg-sidebar-active font-medium text-sidebar-text-active'
                  : 'text-sidebar-text hover:bg-sidebar-hover hover:text-sidebar-text-active'
              }`
            }
          >
            {expanded ? (
              <span className="truncate">{item.label}</span>
            ) : (
              <span className="text-xs font-semibold">{item.short}</span>
            )}
          </NavLink>
        ))}
      </div>

      {/* Footer */}
      <div className="mt-auto border-t border-slate-700/40 px-3 py-3">
        <p
          className={`text-center text-[11px] text-slate-500 transition-opacity duration-150 ${
            expanded ? '' : 'px-0'
          }`}
        >
          {expanded ? 'Fraud Detection System' : 'FDS'}
        </p>
      </div>
    </nav>
  );
}

const App: React.FC = () => {
  const [expanded, setExpanded] = useState(true);

  return (
    <SidebarCtx.Provider value={{ expanded, toggle: () => setExpanded((v) => !v) }}>
      <BrowserRouter>
        <div className="flex min-h-screen">
          <Sidebar />
          <main
            className="flex-1 overflow-y-auto bg-surface-50 transition-[margin-left] duration-200 ease-in-out"
            style={{ marginLeft: expanded ? '13rem' : '60px' }}
          >
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/users/:userId" element={<UserDetail />} />
              <Route path="/graph/:userId" element={<GraphAnalysis />} />
              <Route path="/batch" element={<BatchPredict />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </SidebarCtx.Provider>
  );
};

export default App;
