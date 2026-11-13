import { HashRouter, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { AuthProvider, useAuth } from './auth/AuthContext.jsx';
import { Layout } from './components/Layout.jsx';
import { Spinner } from './components/ui.jsx';
import { LoginPage } from './pages/LoginPage.jsx';
import { ProjectsPage } from './pages/ProjectsPage.jsx';
import { ProjectPage } from './pages/ProjectPage.jsx';
import { ProgressPage } from './pages/ProgressPage.jsx';
import { ViewerPage } from './pages/ViewerPage.jsx';

/** Authorization: every page except /login needs a valid token. 401s from the API also land here (see AuthContext). */
export function RequireAuth({ children }) {
  const { user, checking } = useAuth();
  const loc = useLocation();
  if (checking) return <div className="center-screen"><Spinner /></div>;
  if (!user) return <Navigate to="/login" replace state={{ from: loc.pathname + loc.search }} />;
  return children;
}

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<RequireAuth><Layout /></RequireAuth>}>
        <Route path="/" element={<Navigate to="/projects" replace />} />
        <Route path="/projects" element={<ProjectsPage />} />
        <Route path="/projects/:id" element={<ProjectPage />} />
        <Route path="/projects/:id/progress/:taskId" element={<ProgressPage />} />
        <Route path="/projects/:id/viewer" element={<ViewerPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <HashRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </HashRouter>
  );
}
