import { Component, StrictMode, type ErrorInfo, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import "katex/dist/katex.min.css";

import App from "./App";
import "./styles.css";

interface AppErrorBoundaryState {
  error: Error | null;
}

class AppErrorBoundary extends Component<{ children: ReactNode }, AppErrorBoundaryState> {
  state: AppErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): AppErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("MFQ Studio failed to render", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    const chinese = navigator.language.toLowerCase().startsWith("zh");
    return (
      <main className="fatal-error" role="alert">
        <section>
          <p className="eyebrow">MFQ STUDIO</p>
          <h1>{chinese ? "界面遇到错误" : "The interface hit an error"}</h1>
          <p>
            {chinese
              ? "模型服务仍在运行。重新载入界面通常可以恢复；下面的信息可用于定位问题。"
              : "The model service is still running. Reloading the interface usually recovers it; the detail below helps diagnose the problem."}
          </p>
          <pre>{this.state.error.message || this.state.error.name}</pre>
          <button onClick={() => window.location.reload()} type="button">
            {chinese ? "重新载入" : "Reload"}
          </button>
        </section>
      </main>
    );
  }
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AppErrorBoundary>
      <App />
    </AppErrorBoundary>
  </StrictMode>,
);
