import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './styles.css'

class AppBoundary extends React.Component {
  state = { error: null }
  static getDerivedStateFromError(error) { return { error } }
  render() {
    if (this.state.error) return <main className="boot" role="alert"><strong>The app could not render.</strong><p>{this.state.error.message}</p><button onClick={() => window.location.reload()}>Reload app</button></main>
    return this.props.children
  }
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AppBoundary><App /></AppBoundary>
  </React.StrictMode>,
)
