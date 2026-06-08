"""v11 → v12: 移除 VersionPhase 的 `tagging` 值（自动打标步骤已删除）。

VersionPhase 从 6 → 5：

    curating → preprocessing → editing → regularizing → ready

phase 列是 TEXT（_v8 加的），无需改 schema；只把存量数据里仍停在
`tagging` 的 version 平移到下一个必经 phase `editing`（旧 tagging 可跳过，
跳过即进 editing —— 与历史语义一致，用户不丢进度）。

幂等：没有 phase='tagging' 行时 UPDATE 影响 0 行，无副作用。
"""
from __future__ import annotations

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute(
        "UPDATE versions SET phase = 'editing' WHERE phase = 'tagging'"
    )
