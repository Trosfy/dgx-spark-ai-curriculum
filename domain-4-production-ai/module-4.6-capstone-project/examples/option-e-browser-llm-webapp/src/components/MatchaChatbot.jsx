import React, { useState, useRef, useEffect } from 'react';
import useModelLoader from '../hooks/useModelLoader';

/**
 * System prompt for the Matcha Expert chatbot
 */
const SYSTEM_PROMPT = `You are a matcha tea expert with deep knowledge of Japanese tea culture, preparation methods, health benefits, and culinary applications. You provide accurate, helpful information about:
- Matcha grades (ceremonial, premium, culinary) and quality indicators
- Traditional preparation with chasen (bamboo whisk) and chawan (tea bowl)
- Health benefits including L-theanine, EGCG, and antioxidants
- Japanese tea ceremony (chado/sado) traditions
- Matcha-based recipes and culinary techniques
- Sourcing and storage recommendations

Be concise, accurate, and helpful. If you're not certain about something, say so.`;

/**
 * Sample questions to help users get started
 */
const SAMPLE_QUESTIONS = [
  "What's the difference between ceremonial and culinary grade matcha?",
  "How do I make traditional matcha with a bamboo whisk?",
  "What are the health benefits of matcha compared to regular green tea?",
  "How should I store matcha to keep it fresh?",
];

/**
 * MatchaChatbot Component
 *
 * A complete chat interface for the browser-deployed matcha expert LLM.
 * Handles model loading, message generation, and conversation display.
 */
function MatchaChatbot() {
  // State
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);

  // Refs
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Model loader hook
  const {
    generator,
    isLoading,
    loadingProgress,
    loadingStage,
    error,
    backendInfo,
    loadModel,
  } = useModelLoader();

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input when model is ready
  useEffect(() => {
    if (generator && !isLoading) {
      inputRef.current?.focus();
    }
  }, [generator, isLoading]);

  /**
   * Generate a response from the model
   */
  const generateResponse = async (userMessage) => {
    if (!generator) return;

    setIsGenerating(true);

    // Build the conversation history for the model
    const conversationHistory = [
      { role: 'system', content: SYSTEM_PROMPT },
      ...messages.map((m) => ({
        role: m.role,
        content: m.content,
      })),
      { role: 'user', content: userMessage },
    ];

    try {
      // Generate response using Transformers.js pipeline
      const result = await generator(conversationHistory, {
        max_new_tokens: 256,
        temperature: 0.7,
        top_p: 0.9,
        do_sample: true,
        return_full_text: false,
      });

      // Extract the generated text
      const generatedText = result[0]?.generated_text || 'Sorry, I could not generate a response.';

      // Add assistant message
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          role: 'assistant',
          content: generatedText.trim(),
          timestamp: new Date(),
        },
      ]);
    } catch (err) {
      console.error('Generation error:', err);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          role: 'assistant',
          content: 'Sorry, an error occurred while generating a response. Please try again.',
          timestamp: new Date(),
          isError: true,
        },
      ]);
    } finally {
      setIsGenerating(false);
    }
  };

  /**
   * Handle form submission
   */
  const handleSubmit = async (e) => {
    e.preventDefault();
    const message = inputValue.trim();
    if (!message || isGenerating || !generator) return;

    // Add user message
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        role: 'user',
        content: message,
        timestamp: new Date(),
      },
    ]);

    // Clear input
    setInputValue('');

    // Generate response
    await generateResponse(message);
  };

  /**
   * Handle sample question click
   */
  const handleSampleQuestion = (question) => {
    setInputValue(question);
    inputRef.current?.focus();
  };

  /**
   * Clear conversation
   */
  const handleClearChat = () => {
    setMessages([]);
    inputRef.current?.focus();
  };

  // Render loading state
  if (isLoading || !generator) {
    return (
      <div className="chatbot-container">
        <div className="loading-panel">
          <div className="loading-icon">🍵</div>
          <h2>Loading Matcha Expert</h2>

          {error ? (
            <div className="error-message">
              <p>Failed to load model: {error}</p>
              <button onClick={loadModel} className="retry-button">
                Retry
              </button>
            </div>
          ) : (
            <>
              <p className="loading-stage">{loadingStage || 'Initializing...'}</p>

              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${Math.round(loadingProgress * 100)}%` }}
                />
              </div>

              <p className="progress-text">{Math.round(loadingProgress * 100)}%</p>

              <div className="loading-info">
                <p>
                  <strong>First load:</strong> ~500 MB download (cached for future visits)
                </p>
                {backendInfo && (
                  <p>
                    <strong>Backend:</strong> {backendInfo}
                  </p>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    );
  }

  // Render chat interface
  return (
    <div className="chatbot-container">
      {/* Chat header */}
      <div className="chat-header">
        <div className="status-indicator">
          <span className="status-dot online"></span>
          <span>Model ready ({backendInfo})</span>
        </div>
        <button onClick={handleClearChat} className="clear-button" title="Clear conversation">
          Clear Chat
        </button>
      </div>

      {/* Messages area */}
      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="welcome-panel">
            <h2>Welcome to Matcha Expert!</h2>
            <p>Ask me anything about matcha tea, Japanese tea culture, preparation methods, or recipes.</p>

            <div className="sample-questions">
              <p className="sample-label">Try asking:</p>
              {SAMPLE_QUESTIONS.map((question, index) => (
                <button
                  key={index}
                  onClick={() => handleSampleQuestion(question)}
                  className="sample-question"
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <div
                key={message.id}
                className={`message ${message.role} ${message.isError ? 'error' : ''}`}
              >
                <div className="message-avatar">
                  {message.role === 'user' ? '👤' : '🍵'}
                </div>
                <div className="message-content">
                  <div className="message-text">{message.content}</div>
                  <div className="message-time">
                    {message.timestamp.toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </div>
                </div>
              </div>
            ))}

            {isGenerating && (
              <div className="message assistant generating">
                <div className="message-avatar">🍵</div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input area */}
      <form onSubmit={handleSubmit} className="input-form">
        <input
          ref={inputRef}
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ask about matcha..."
          disabled={isGenerating}
          className="message-input"
          autoComplete="off"
        />
        <button
          type="submit"
          disabled={!inputValue.trim() || isGenerating}
          className="send-button"
        >
          {isGenerating ? '...' : 'Send'}
        </button>
      </form>
    </div>
  );
}

export default MatchaChatbot;
