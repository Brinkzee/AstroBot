import asyncio
from app.db.session import engine
from scripts.wsl_helper import ensure_mysql_ready
from sqlalchemy import text

async def run():
    ensure_mysql_ready()
    async with engine.begin() as conn:
        try:
            await conn.exec_driver_sql("ALTER TABLE low_confidence_questions ADD COLUMN retrieved_chunks JSON NULL COMMENT 'retrieved_chunks';")
        except Exception as e:
            print(e)
        
        try:
            await conn.exec_driver_sql("ALTER TABLE low_confidence_questions ADD COLUMN matched_review_id BIGINT UNSIGNED NULL;")
        except Exception as e:
            print(e)
        
        try:
            await conn.exec_driver_sql("CREATE TABLE IF NOT EXISTS eval_runs (id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT, triggered_by ENUM('定时','手动') NOT NULL DEFAULT '定时', dataset_size INT UNSIGNED NOT NULL, metrics JSON NOT NULL, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (id), KEY idx_created_at (created_at)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;")
        except Exception as e:
            print(e)

        try:
            await conn.exec_driver_sql("CREATE TABLE IF NOT EXISTS review_queue (id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT, normalized_question VARCHAR(512) NOT NULL, ai_suggested_answer TEXT NULL, occurrence_count INT UNSIGNED NOT NULL DEFAULT 1, review_status ENUM('待审','通过','驳回') NOT NULL DEFAULT '待审', approved_answer TEXT NULL, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, PRIMARY KEY (id), KEY idx_review_status (review_status)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;")
        except Exception as e:
            print(e)

if __name__ == "__main__":
    asyncio.run(run())
