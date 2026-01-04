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
   // Recommended: CloudFront distribution URL
   const MODEL_URL = 'https://d1234567890abc.cloudfront.net/';
   // Alternative: Hugging Face Hub
   // const MODEL_URL = 'your-username/troscha-matcha-onnx';
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

## Deployment (AWS S3 + CloudFront)

Deploy the entire app (React + model) as a single static site:

### Step 1: Build the App
```bash
npm run build
# Output: dist/ folder
```

### Step 2: Create S3 Bucket
```bash
aws s3 mb s3://troscha-matcha-demo --region us-east-1
aws s3 website s3://troscha-matcha-demo --index-document index.html
```

### Step 3: Upload Everything
```bash
# Upload React app
aws s3 sync ./dist s3://troscha-matcha-demo/ --acl public-read

# Upload model files to /model/ subfolder
aws s3 sync ./models/troscha-browser s3://troscha-matcha-demo/model/ --acl public-read
```

### Step 4: Create CloudFront Distribution
1. Go to AWS CloudFront Console → Create Distribution
2. Origin domain: `troscha-matcha-demo.s3.us-east-1.amazonaws.com`
3. Default root object: `index.html`
4. Create Response Headers Policy with:
   - `Cross-Origin-Opener-Policy: same-origin`
   - `Cross-Origin-Embedder-Policy: require-corp`
5. Wait for deployment (~5-10 minutes)

### Step 5: Update MODEL_URL
```javascript
// Use relative path (same bucket)
const MODEL_URL = '/model/';
// Or full CloudFront URL
const MODEL_URL = 'https://d1234567890abc.cloudfront.net/model/';
```

**Your site:** `https://d1234567890abc.cloudfront.net/`

### Benefits
- Single deployment for app + model
- Global edge caching (~500MB model cached at edge)
- Free SSL certificate
- User downloads model once, runs locally forever

## Model Requirements

Your model must be:
- **Format**: ONNX with INT4 quantization
- **Size**: <500 MB recommended for reasonable load times
- **Hosted**: AWS S3 + CloudFront (recommended) or Hugging Face Hub

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
