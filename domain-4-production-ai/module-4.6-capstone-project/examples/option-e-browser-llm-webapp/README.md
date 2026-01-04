# Troscha Matcha Guide - Browser LLM Chatbot

A browser-deployed AI chatbot specialized in Troscha's premium matcha products. Runs entirely in the browser using WebGPU/WASM - no server required!

## Features

- **100% Client-Side**: All AI inference runs locally in your browser
- **Zero Cost**: No API fees, no server hosting costs after deployment
- **Privacy First**: No data leaves your device
- **WebGPU Accelerated**: Fast inference on modern GPUs
- **WASM Fallback**: Works on any modern browser

## Prerequisites

- Node.js 18+ and npm
- A trained and converted model (ONNX INT4 format)
- Model hosted at a CORS-enabled URL

## Quick Start

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Configure your model URL:**
   Edit `src/hooks/useModelLoader.js` and update `MODEL_URL`:
   ```javascript
   const MODEL_URL = 'https://your-bucket.s3.amazonaws.com/model/';
   // Or: 'your-username/troscha-matcha-onnx' for Hugging Face Hub
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```

4. **Open in browser:**
   Navigate to `http://localhost:5173`

## Project Structure

```
├── public/
│   └── matcha-icon.svg      # Favicon
├── src/
│   ├── components/
│   │   └── MatchaChatbot.jsx  # Main chatbot component
│   ├── hooks/
│   │   └── useModelLoader.js  # Model loading logic
│   ├── styles/
│   │   └── chatbot.css        # Styling
│   ├── App.jsx                # Root component
│   └── main.jsx               # Entry point
├── index.html
├── package.json
├── vite.config.js
└── vercel.json                # Deployment config
```

## Deployment

### Vercel (Recommended)

1. Push to GitHub
2. Connect to Vercel
3. Deploy - headers are auto-configured via `vercel.json`

### Netlify

Create `netlify.toml`:
```toml
[[headers]]
  for = "/*"
  [headers.values]
    Cross-Origin-Opener-Policy = "same-origin"
    Cross-Origin-Embedder-Policy = "require-corp"
```

### GitHub Pages

Add to your deploy script:
```yaml
- name: Add Security Headers
  run: |
    echo '/*
      Cross-Origin-Opener-Policy: same-origin
      Cross-Origin-Embedder-Policy: require-corp' > dist/_headers
```

## Model Requirements

Your model must be:
- **Format**: ONNX with INT4 quantization
- **Size**: <500 MB recommended for reasonable load times
- **Hosted**: At a CORS-enabled URL (S3, Hugging Face Hub, etc.)

Required files:
- `model.onnx` or `model_quantized.onnx`
- `tokenizer.json`
- `tokenizer_config.json`
- `config.json`
- `special_tokens_map.json`

## Browser Compatibility

| Browser | WebGPU | WASM |
|---------|--------|------|
| Chrome 113+ | ✅ | ✅ |
| Edge 113+ | ✅ | ✅ |
| Firefox 121+ | ⚠️ Flag | ✅ |
| Safari 17+ | ⚠️ Limited | ✅ |

## Customization

### Changing the System Prompt

Edit `SYSTEM_PROMPT` in `src/components/MatchaChatbot.jsx`:
```javascript
const SYSTEM_PROMPT = `Your custom system prompt here...`;
```

### Changing the Theme

Update CSS variables in `src/styles/chatbot.css`:
```css
:root {
  --matcha-primary: #4a7c59;  /* Main color */
  --matcha-dark: #2d5a3d;     /* Darker shade */
  /* ... */
}
```

### Adding More Sample Questions

Edit `SAMPLE_QUESTIONS` in `src/components/MatchaChatbot.jsx`:
```javascript
const SAMPLE_QUESTIONS = [
  "Your question 1",
  "Your question 2",
  // ...
];
```

## Performance Tips

1. **Use WebGPU**: Ensure users have WebGPU-capable browsers
2. **Pre-cache Model**: Service workers can cache model files
3. **Lazy Loading**: Model loads on demand, not blocking initial render
4. **Streaming**: Consider implementing token streaming for UX

## Troubleshooting

### Model fails to load

1. Check browser console for errors
2. Verify CORS headers on model host
3. Ensure model files are accessible
4. Try a different browser

### Slow generation

1. Check if WebGPU is being used (not WASM fallback)
2. Ensure GPU drivers are up to date
3. Close other GPU-intensive applications

### SharedArrayBuffer error

Ensure your server sends these headers:
```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

## License

MIT

## Credits

- Built with [React](https://react.dev/) and [Vite](https://vitejs.dev/)
- AI inference by [Transformers.js](https://huggingface.co/docs/transformers.js)
- Fine-tuned on [NVIDIA DGX Spark](https://www.nvidia.com/dgx-spark/)
- Part of the DGX Spark AI Curriculum Capstone Project
