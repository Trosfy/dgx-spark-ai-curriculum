import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],

  // Build configuration
  build: {
    target: 'esnext',
    outDir: 'dist',
    sourcemap: false,
    minify: 'esbuild',
    rollupOptions: {
      output: {
        manualChunks: {
          // Separate transformers.js into its own chunk
          transformers: ['@huggingface/transformers'],
        },
      },
    },
  },

  // Development server configuration
  server: {
    port: 5173,
    headers: {
      // Required headers for SharedArrayBuffer (needed for WebGPU/WASM threading)
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },

  // Preview server (for testing production builds locally)
  preview: {
    port: 4173,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },

  // Optimize dependencies
  optimizeDeps: {
    exclude: ['@huggingface/transformers'],
  },

  // Worker configuration for ONNX Runtime
  worker: {
    format: 'es',
  },
});
