import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // THE LINK TO THE BACKEND: every /api request (made in src/lib/api.js) is sent to the
    // FastAPI server in dashboard.py on port 8000, so the browser sees one site and we don't need CORS
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
