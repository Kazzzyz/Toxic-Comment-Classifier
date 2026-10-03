import sqlite3
import json
from datetime import datetime
from contextlib import closing


#cursor.execute('CREATE TABLE comments(id INTEGER PRIMARY KEY, text TEXT, date TEXT, time TEXT,predicted_labels TEXT)')


def save_submission(text, predicted_labels):
    now = datetime.now()  

    with closing(sqlite3.connect("comments.db")) as connection:
        connection.execute("""
            INSERT INTO comments (text, date, time, predicted_labels)
            VALUES (?, ?, ?, ?)
        """, (
            text,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            json.dumps(predicted_labels)
        ))

        connection.commit()