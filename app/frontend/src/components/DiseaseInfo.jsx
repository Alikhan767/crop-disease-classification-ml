import styles from "./DiseaseInfo.module.css";

export function DiseaseInfo({ description, treatment }) {
  return (
    <div className={styles.grid}>
      <div className={styles.box}>
        <h4 className={styles.label}>Описание</h4>
        <p className={styles.text}>{description}</p>
      </div>
      <div className={styles.box}>
        <h4 className={styles.label}>Лечение</h4>
        <p className={styles.text}>{treatment}</p>
      </div>
    </div>
  );
}