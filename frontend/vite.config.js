import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite' // <-- Essential v4 Compiler Plugin!

export default defineConfig({
  plugins: [
    react(),
    tailwindcss() // <-- Adds Tailwind directly into the Vite build engine
  ],
})