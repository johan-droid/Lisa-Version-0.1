import sqlite3


def inspect_db(path):
    print(f"=== {path} ===")
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    tables = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    for table in tables:
        table_name = table[0]
        print(f"Table: {table_name}")
        try:
            columns = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
            print("Columns:", [col[1] for col in columns])
            rows = cursor.execute(f"SELECT * FROM {table_name} LIMIT 10").fetchall()
            print("Rows:")
            for r in rows:
                print(r)
        except Exception as e:
            print(f"Error reading {table_name}: {e}")
    conn.close()


inspect_db("data/lisa_notepad.db")
inspect_db("data/personal.db")
