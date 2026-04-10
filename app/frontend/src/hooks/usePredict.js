import { useState, useCallback } from "react";
import { predictDisease } from "../api/api";

export function usePredict() {
  const [result, setResult]   = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const predict = useCallback(async (file) => {
    // Показываем превью сразу
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(file);

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await predictDisease(file);
      setResult(data);
    } catch (err) {
      setError(err.message ?? "Неизвестная ошибка");
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setPreview(null);
    setError(null);
    setLoading(false);
  }, []);

  return { result, preview, loading, error, predict, reset };
}