import { useEffect, useRef, useState } from 'react'
import './App.css'

function App() {
    const [message, setMessage] = useState('')
    const [messages, setMessages] = useState([
      {
        id: crypto.randomUUID(),
        type: 'assistant',
        text: 'Hello. I can help you understand healthcare, find health services in Victoria, and plan how to get there. What would you like help with?',
      },
    ])
    const [loading, setLoading] = useState(false)
    const sessionId = useRef(crypto.randomUUID())

    const inputRef = useRef(null)

    useEffect(() => {
      inputRef.current?.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
      })
    }, [messages, loading])


    function handleExample(question) {
      setMessage(question)
    }

    async function askQuestion() {
      const question = message.trim()

      if (!question || loading) {
        return
      }

      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          type: 'user',
          text: question,
        },
      ])

      setMessage('')
      setLoading(true)

      try {
        const response = await fetch('/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: question,
            session_id: sessionId.current,
          }),
        })

        if (!response.ok) {
          throw new Error('Request failed')
        }

        const data = await response.json()

        setMessages((current) => [
          ...current,
          {
            id: crypto.randomUUID(),
            type: 'assistant',
            text: data.response,
            link: data.link,
            images: data.images || [],
          },
        ])
      } catch {
        setMessages((current) => [
          ...current,
          {
            id: crypto.randomUUID(),
            type: 'assistant',
            text: "Sorry, I couldn't process your request. Please try again.",
          },
        ])
      } finally {
        setLoading(false)
        inputRef.current?.focus()
      }
    }

    function handleKeyDown(event) {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault()
        askQuestion()
      }
    }

    return (
    <>
      <header>
        <div className="header-content">
          <h1>Health Service Navigator</h1>
          <div className="subtitle">
            AI-powered navigation for health services in Victoria
          </div>
        </div>
      </header>

      <main className="container">
        <div className="safety">
          <strong>Health service navigation only.</strong>{' '}
          Health Service Navigator uses AI to help you understand the healthcare
          system, find health services in Victoria, and navigate access pathways
          such as Medicare, referrals and appointments. It does not provide
          medical advice, diagnosis, treatment, medication advice, or
          interpretation of symptoms or clinical results.
        </div>

        <div className="examples">
          <p>Try asking:</p>

          <button
            className="example-button"
            onClick={() => handleExample('Find a GP near Glenroy')}
          >
            Find a health service
          </button>

          <button
            className="example-button"
            onClick={() =>
              handleExample(
                'How do I get to the Royal Children’s Hospital by public transport?'
              )
            }
          >
            Plan my journey
          </button>

          <button
            className="example-button"
            onClick={() => handleExample('How do I get a Medicare card?')}
          >
            Understand Medicare
          </button>

          <button
            className="example-button"
            onClick={() =>
              handleExample('Do I need a referral to see a specialist?')
            }
          >
            Understand referrals
          </button>
        </div>

        <div className="chat" >
          {messages.map((item) => (
            <div key={item.id} className={`message ${item.type}`}>
              <span>
                {item.text}

                {item.link && (
                  <a
                    href={item.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="healthdirect-link"
                  >
                    Open Healthdirect Service Finder
                  </a>
                )}

                {item.images?.map((src, imageIndex) => {
                  const labels = [
                    'Step 1 — Search for a service',
                    'Step 2 — Refine your results',
                  ]

                  const label = labels[imageIndex] || 'Healthdirect guide'

                  return (
                    <div key={src} className="healthdirect-guide-section">
                      <div className="healthdirect-guide-label">
                        {label}
                      </div>
                      <img
                        src={src}
                        alt={label}
                        className="healthdirect-guide"
                      />
                    </div>
                  )
                })}
              </span>
            </div>
          ))}
          {loading && (
            <div className="message assistant">
              <span>Thinking...</span>
            </div>
          )}
        </div>

        <div className="input-area">
          <textarea
            ref={inputRef}
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about health services in Victoria..."
          />

          <button
            id="sendButton"
            onClick={askQuestion}
            disabled={loading}
          >
            Send
          </button>
        </div>
      </main>

      <footer>
        Health Service Navigator • AI-generated information may contain errors •
        Verify important information with the relevant health service or official
        source
      </footer>
    </>
  )
}

export default App