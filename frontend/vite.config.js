import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server proxies /api (including the SSE stream) to the backend, so the
// browser always sees one origin.  VITE_API_TARGET lets ./run.sh dev point the
// proxy at a non-default PORT.
const target = process.env.VITE_API_TARGET || 'http://127.0.0.1:8765'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target, changeOrigin: false },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
