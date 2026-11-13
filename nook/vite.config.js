import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Dev server on 5173 is already in the backend's default CORS list.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, host: true },
  build: { chunkSizeWarningLimit: 1500 },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./tests/setup.js'],
    include: ['tests/**/*.test.{js,jsx}'],
  },
});
