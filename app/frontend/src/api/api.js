const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function predictDisease(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `Server error: ${res.status}`);
  }
  return res.json();
}

export async function checkHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error("Server unavailable");
  return res.json();
}
