import styles from "./TopKChart.module.css";

export function TopKChart({ topK, accentColor }) {
  return (
    <div className={styles.wrapper}>
      <h4 className={styles.title}>Топ предсказания</h4>
      <div className={styles.list}>
        {topK.map((p, i) => (
          <div key={i} className={styles.row}>
            <span className={styles.rank}>#{i + 1}</span>
            <span className={styles.name}>
              {p.plant} — {p.disease}
            </span>
            <div className={styles.barWrap}>
              <div
                className={styles.bar}
                style={{
                  width: `${(p.confidence * 100).toFixed(1)}%`,
                  background: accentColor,
                  opacity: 1 - i * 0.25,
                }}
              />
            </div>
            <span className={styles.pct}>
              {(p.confidence * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}