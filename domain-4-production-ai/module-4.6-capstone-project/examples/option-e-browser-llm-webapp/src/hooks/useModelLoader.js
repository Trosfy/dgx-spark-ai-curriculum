import { useState, useEffect, useCallback, useRef } from 'react';
import { pipeline, env } from '@huggingface/transformers';

/**
 * Model Configuration
 *
 * Configure via environment variable or update the fallback URL.
 * Options:
 * 1. Hugging Face Hub: "your-username/troscha-matcha-onnx"
 * 2. S3 bucket: "https://your-bucket.s3.region.amazonaws.com/"
 * 3. Any CORS-enabled URL serving the ONNX files
 *
 * For local development, set VITE_MODEL_URL in .env:
 *   VITE_MODEL_URL=https://your-bucket.s3.us-east-1.amazonaws.com/
 */
const MODEL_URL = import.meta.env.VITE_MODEL_URL || 'YOUR_MODEL_URL_HERE';

// Validate MODEL_URL is configured
if (MODEL_URL === 'YOUR_MODEL_URL_HERE') {
  console.warn(
    '[useModelLoader] MODEL_URL not configured! Set VITE_MODEL_URL in .env or update useModelLoader.js'
  );
}

/**
 * Configure Transformers.js environment
 */
env.allowLocalModels = false;
env.useBrowserCache = true;

/**
 * Detect the best available backend
 */
async function detectBackend() {
  // Check for WebGPU support
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        return { device: 'webgpu', name: 'WebGPU' };
      }
    } catch (e) {
      console.warn('WebGPU detection failed:', e);
    }
  }

  // Check for WebGL support
  if (typeof document !== 'undefined') {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      if (gl) {
        return { device: 'webgl', name: 'WebGL' };
      }
    } catch (e) {
      console.warn('WebGL detection failed:', e);
    }
  }

  // Fallback to WASM
  return { device: 'wasm', name: 'WASM (CPU)' };
}

/**
 * useModelLoader Hook
 *
 * Manages the loading and lifecycle of the Transformers.js text generation pipeline.
 * Provides progress tracking, error handling, and backend detection.
 *
 * @returns {Object} Model state and controls
 */
export default function useModelLoader() {
  // State
  const [generator, setGenerator] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [loadingStage, setLoadingStage] = useState('');
  const [error, setError] = useState(null);
  const [backendInfo, setBackendInfo] = useState(null);

  // Track if we've already started loading
  const loadingRef = useRef(false);

  /**
   * Progress callback for model loading
   */
  const handleProgress = useCallback((progress) => {
    if (progress.status === 'initiate') {
      setLoadingStage(`Downloading ${progress.file}...`);
    } else if (progress.status === 'download') {
      // Calculate percentage if we have loaded and total
      if (progress.loaded && progress.total) {
        const pct = progress.loaded / progress.total;
        setLoadingProgress(pct);
      }
    } else if (progress.status === 'progress') {
      // Some versions use 'progress' instead of 'download'
      if (typeof progress.progress === 'number') {
        setLoadingProgress(progress.progress / 100);
      }
    } else if (progress.status === 'done') {
      setLoadingStage(`Loaded ${progress.file}`);
    } else if (progress.status === 'ready') {
      setLoadingStage('Model ready!');
      setLoadingProgress(1);
    }
  }, []);

  /**
   * Load the model
   */
  const loadModel = useCallback(async () => {
    // Prevent duplicate loading
    if (loadingRef.current) return;
    loadingRef.current = true;

    setIsLoading(true);
    setError(null);
    setLoadingProgress(0);
    setLoadingStage('Detecting hardware capabilities...');

    try {
      // Detect best backend
      const backend = await detectBackend();
      setBackendInfo(backend.name);
      setLoadingStage(`Using ${backend.name} backend. Loading model...`);

      console.log(`Loading model with ${backend.name} backend...`);

      // Create the text generation pipeline
      const pipe = await pipeline('text-generation', MODEL_URL, {
        device: backend.device,
        dtype: 'q4', // INT4 quantized model
        progress_callback: handleProgress,
      });

      setGenerator(() => pipe);
      setLoadingStage('Model ready!');
      setLoadingProgress(1);
      console.log('Model loaded successfully!');
    } catch (err) {
      console.error('Failed to load model:', err);
      setError(err.message || 'Failed to load model');
      loadingRef.current = false;
    } finally {
      setIsLoading(false);
    }
  }, [handleProgress]);

  // Auto-load on mount
  useEffect(() => {
    loadModel();

    // Cleanup
    return () => {
      if (generator) {
        // Dispose of the pipeline if possible
        try {
          generator.dispose?.();
        } catch (e) {
          console.warn('Error disposing pipeline:', e);
        }
      }
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    generator,
    isLoading,
    loadingProgress,
    loadingStage,
    error,
    backendInfo,
    loadModel,
  };
}
