# Nook: video → browsable 3D room (frontend)

React + three.js + plain CSS frontend for the Cellula **"From Video to a Browsable 3D Room"** project (Person C).
It talks to the FastAPI backend in `ShahdElsawy1/3D-Room` (`backend/`). Nothing else is needed.

Colours: **green** = go / primary, **blue** = information / progress, **yellow** = attention / selection (and the object mask).

## What works

| Spec item                                                                        | Where                                                                                 |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Sign up / log in, authorization (JWT)                                            | `src/pages/LoginPage.jsx`, `src/auth/AuthContext.jsx`, `RequireAuth` in `src/App.jsx` |
| Projects list (yours only)                                                       | `src/pages/ProjectsPage.jsx`                                                          |
| Upload with submit (real upload progress)                                        | `src/pages/ProjectPage.jsx`                                                           |
| Progress page, **real** backend stages                                           | `src/pages/ProgressPage.jsx`, `StageBar` in `src/components/ui.jsx`                   |
| Viewer: Gaussian splat, orbit + free-fly (WASD + mouse), speed slider            | `src/viewer/*` (uses **Spark**, an open-source three.js splat renderer)               |
| Fallback: sparse point cloud in three.js, marked as a known limitation           | `RoomEngine._loadPoints`, banner in `SplatViewer.jsx`                                 |
| Week 2 hooks: click at screen (x, y), load a camera pose, load a camera path     | `RoomEngine.pickAt / setCameraPose / flyToCamera / playCameraPath`                    |
| Edit object mode (steps 1–6, tracking fallback, accept/regenerate, before/after) | `src/edit/*`, `src/pages/ViewerPage.jsx`                                              |

## Quick start

```bash
npm install
cp .env.example .env        # set VITE_API_URL (default http://localhost:8000)
npm run dev                 # http://localhost:5173
```

The backend must be running and must allow your frontend origin (see `docs/BACKEND_CONNECTION.md`).
Full steps: **[docs/RUN_STEPS.md](docs/RUN_STEPS.md)**.

## Tests

```bash
npm run test:unit    # 88 tests, no backend needed
npm run test:live    # 14 tests against the real backend
npm run test:e2e     # 31 checks in real Chromium (WebGL2) against the real backend
```
