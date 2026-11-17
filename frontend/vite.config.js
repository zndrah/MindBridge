import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    fs: {
      strict: false,
      allow: ['..'],
    },
    proxy: {
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/ready': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/chat': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/safety': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/aiml': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/emergency-resources': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
