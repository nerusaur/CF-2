-- database/schema.sql

CREATE TABLE videos (
    video_id    TEXT PRIMARY KEY,
    label       TEXT CHECK(label IN ('Educational','Neutral','Overstimulating')),
    final_score REAL,
    last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    checked_by  TEXT
);

CREATE TABLE segments (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id       TEXT REFERENCES videos(video_id),
    offset_seconds INTEGER,
    length_seconds INTEGER,
    fcr            REAL,   -- Frame-Change Rate
    csv            REAL,   -- Color Saturation Variance
    att            REAL,   -- Audio Tempo Transitions
    score          REAL
);

CREATE TABLE users (
    user_id         TEXT PRIMARY KEY,
    parent_settings TEXT   -- JSON: thresholds, alerts, screen-time limits
);

CREATE TABLE logs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id       TEXT REFERENCES videos(video_id),
    user_id        TEXT REFERENCES users(user_id),
    action         TEXT,   -- 'allowed', 'blocked', 'blurred'
    timestamp      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reason_details TEXT
);