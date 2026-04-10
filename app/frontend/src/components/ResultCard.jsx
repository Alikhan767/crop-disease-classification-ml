import { DiseaseInfo } from "./DiseaseInfo";
import { TopKChart } from "./TopKChart";
import styles from "./ResultCard.module.css";

const SEVERITY_LABEL = {
  none:     "Здоров",
  moderate: "Умеренно",
  high:     "Серьёзно",
  unknown:  "Неизвестно",
};

export function ResultCard({ result }) {
  const {
    plant, disease, is_healthy, confidence,
    severity, description, treatment,
    color, top_k, time_ms,
  } = result;

  return (
    <div className={styles.card}>

      {/* Шапка */}
      <div className={styles.header} style={{ borderColor: color }}>
        <span className={styles.icon}>{is_healthy ? "✅" : "🔬"}</span>
        <div className={styles.meta}>
          <h2 className={styles.plant}>{plant}</h2>
          <h3 className={styles.disease} style={{ color }}>{disease}</h3>
        </div>
        <span className={styles.badge} style={{ background: color }}>
          {SEVERITY_LABEL[severity] ?? severity}
        </span>
      </div>

      {/* Confidence bar */}
      <div className={styles.confSection}>
        <div className={styles.confLabel}>
          <span>Уверенность</span>
          <strong>{(confidence * 100).toFixed(1)}%</strong>
        </div>
        <div className={styles.confBar}>
          <div
            className={styles.confFill}
            style={{
              width: `${(confidence * 100).toFixed(1)}%`,
              background: color,
            }}
          />
        </div>
      </div>

      {/* Описание и лечение */}
      <DiseaseInfo description={description} treatment={treatment} />

      {/* Топ предсказания */}
      <TopKChart topK={top_k} accentColor={color} />

      {/* Время инференса */}
      <p className={styles.time}>⚡ {time_ms} мс</p>
    </div>
  );
}