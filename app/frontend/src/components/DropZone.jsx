import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import styles from "./DropZone.module.css";

const SUPPORTED_PLANTS = [
  "🍎 Apple", "🍅 Tomato", "🌽 Corn", "🍇 Grape",
  "🥔 Potato", "🍑 Peach", "🫑 Pepper", "🍓 Strawberry",
];

export function DropZone({ onFile, preview, loading, onReset }) {
  const onDrop = useCallback(
    (accepted) => {
      if (accepted[0]) onFile(accepted[0]);
    },
    [onFile]
  );

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    maxFiles: 1,
    noClick: false,
    disabled: loading,
  });

  if (preview) {
    return (
      <div className={styles.previewWrap}>
        <img src={preview} alt="leaf preview" className={styles.previewImg} />
        {loading && (
          <div className={styles.overlay}>
            <span className={styles.spinner} />
            <p>Анализируем...</p>
          </div>
        )}
        {!loading && (
          <button className={styles.resetBtn} onClick={onReset}>
            Загрузить другое фото
          </button>
        )}
      </div>
    );
  }

  return (
    <div className={styles.wrapper}>
      <div
        {...getRootProps()}
        className={`${styles.dropzone} ${isDragActive ? styles.active : ""}`}
      >
        <input {...getInputProps()} />
        <svg className={styles.icon} viewBox="0 0 48 48" fill="none">
          <rect width="48" height="48" rx="14" fill="#e8f5ef"/>
          <path d="M24 14v14M17 21l7-7 7 7" stroke="#2d9e75" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M14 34h20" stroke="#2d9e75" strokeWidth="2" strokeLinecap="round"/>
        </svg>
        <h2 className={styles.heading}>Загрузите фото листа</h2>
        <p className={styles.hint}>Перетащите или нажмите для выбора файла</p>
        <p className={styles.formats}>JPG, PNG, WEBP — до 10 МБ</p>
        <button
          className={styles.uploadBtn}
          onClick={(e) => { e.stopPropagation(); open(); }}
        >
          Выбрать фото
        </button>
      </div>

      <div className={styles.plants}>
        <p className={styles.plantsLabel}>Поддерживаемые культуры:</p>
        <div className={styles.tags}>
          {SUPPORTED_PLANTS.map((p) => (
            <span key={p} className={styles.tag}>{p}</span>
          ))}
        </div>
      </div>
    </div>
  );
}