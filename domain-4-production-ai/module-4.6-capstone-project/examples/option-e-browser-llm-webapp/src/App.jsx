import React from 'react';
import MatchaChatbot from './components/MatchaChatbot';
import './styles/chatbot.css';

/**
 * Main Application Component
 *
 * This is the root component for the Matcha Expert browser chatbot.
 * It provides the main layout and renders the chatbot interface.
 */
function App() {
  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">🍵</span>
            <h1>Matcha Expert</h1>
          </div>
          <p className="tagline">AI-powered matcha knowledge, running entirely in your browser</p>
        </div>
      </header>

      <main className="app-main">
        <MatchaChatbot />
      </main>

      <footer className="app-footer">
        <div className="footer-content">
          <p>
            <strong>Privacy First:</strong> This AI runs 100% locally in your browser.
            No data is sent to any server.
          </p>
          <p className="tech-info">
            Powered by Transformers.js | Fine-tuned on NVIDIA DGX Spark |
            <a
              href="https://github.com/your-username/matcha-expert"
              target="_blank"
              rel="noopener noreferrer"
            >
              View Source
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
