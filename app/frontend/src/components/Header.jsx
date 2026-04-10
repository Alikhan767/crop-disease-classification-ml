import styles from "./Header.module.css";

export function Header() {
  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <div className={styles.logo}>
          <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
            <rect width="36" height="36" rx="10" fill="#1a4731"/>
            <path
              d="M18 6C14 6 10 9 10 14c0 3 1.5 5.5 4 7l-1 4h10l-1-4c2.5-1.5 4-4 4-7 0-5-4-8-8-8z"
              fill="#2d9e75"
            />
            <path d="M15 27h6M16.5 29.5h3" stroke="#fff" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
          <span className={styles.title}>PlantGuard AI</span>
        </div>
        <p className={styles.subtitle}>
          Определение болезней растений по фотографии листа
        </p>
      </div>
    </header>
  );
}