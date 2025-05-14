import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/

export default defineConfig({
  plugins: [react()],
  server: {
    host: true          // <-- This is important to expose to your local network
  },
  define: {
    'import.meta.env.DEV': process.env.NODE_ENV === 'development'
  }
});