import { Component, StrictMode } from "react";
import { createRoot } from "react-dom/client";
import type { ReactNode } from "react";

import App from "./App";
import "./styles.css";
// Imported after styles.css so the consolidated reply-body rules win on source order.
import "./narrative-output.css";

/**
 * Top-level error boundary that prevents any render-phase throw from
 * unmounting the entire React root (which would leave a blank page).
 * Modelled after SimulationRunErrorBoundary in App.tsx.
 */
class AppErrorBoundary extends Component<{ children: ReactNode }, { errorMessage?: string }> {
  state: { errorMessage?: string } = {};

  static getDerivedStateFromError(error: unknown): { errorMessage: string } {
    return {
      errorMessage: error instanceof Error ? error.message : "앱 초기화 중 예기치 않은 오류가 발생했습니다."
    };
  }

  render() {
    if (this.state.errorMessage) {
      return (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            height: "100vh",
            gap: "12px",
            fontFamily: "sans-serif",
            padding: "24px",
            textAlign: "center"
          }}
        >
          <strong style={{ fontSize: "1.1rem" }}>앱을 표시하지 못했습니다.</strong>
          <span style={{ color: "#888", maxWidth: "480px" }}>{this.state.errorMessage}</span>
          <button
            style={{ marginTop: "8px", padding: "8px 20px", cursor: "pointer" }}
            onClick={() => window.location.reload()}
          >
            새로 고침
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

const root = document.getElementById("root");

if (!root) {
  throw new Error("Root element not found");
}

createRoot(root).render(
  <StrictMode>
    <AppErrorBoundary>
      <App />
    </AppErrorBoundary>
  </StrictMode>
);
