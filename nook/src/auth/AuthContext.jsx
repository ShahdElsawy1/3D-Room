import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { auth as authApi } from '../api/endpoints.js';
import { clearToken, getToken, setToken, setUnauthorizedHandler } from '../api/client.js';

const Ctx = createContext(null);
export const useAuth = () => useContext(Ctx);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  // "checking" while we ask /auth/me whether a stored token is still valid
  const [checking, setChecking] = useState(() => Boolean(getToken()));

  const logout = useCallback(() => { clearToken(); setUser(null); }, []);

  useEffect(() => { setUnauthorizedHandler(logout); return () => setUnauthorizedHandler(null); }, [logout]);

  useEffect(() => {
    if (!getToken()) return undefined;
    let cancelled = false;
    authApi.me()
      .then((u) => { if (!cancelled) setUser(u); })
      .catch((e) => { if (!cancelled && e.status === 401) logout(); })
      .finally(() => { if (!cancelled) setChecking(false); });
    return () => { cancelled = true; };
  }, [logout]);

  const accept = useCallback((tokenOut) => { setToken(tokenOut.access_token); setUser(tokenOut.user); return tokenOut.user; }, []);
  const login = useCallback(async (email, password) => accept(await authApi.login(email, password)), [accept]);
  const signup = useCallback(async (email, password) => accept(await authApi.signup(email, password)), [accept]);

  const value = useMemo(() => ({ user, checking, login, signup, logout }), [user, checking, login, signup, logout]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
