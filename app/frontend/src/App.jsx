import { Header }     from "./components/Header";
import { DropZone }   from "./components/DropZone";
import { ResultCard } from "./components/ResultCard";
import { usePredict } from "./hooks/usePredict";
import "./App.css";

export default function App() {
  const { result, preview, loading, error, predict, reset } = usePredict();

  return (
    <div className="app">
      <Header />

      <main className="main">
        <DropZone
          onFile={predict}
          preview={preview}
          loading={loading}
          onReset={reset}
        />

        {error && (
          <div className="error-box">
            ⚠️ {error}
          </div>
        )}

        {result && <ResultCard result={result} />}
      </main>

      <footer className="footer">
        PlantGuard AI © 2025 — Дипломная работа
      </footer>
    </div>
  );
}