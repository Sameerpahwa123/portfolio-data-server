"""Configuration from .env. Tokens are never logged, printed or repr'd."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AccountConfig:
    label: str
    token: str = field(repr=False)
    query_id: str


@dataclass(frozen=True)
class ManualAccount:
    """An account with no broker feed (e.g. Capital.com): balances typed in by hand, never reconciled."""

    label: str
    title: str
    deposited: float
    current_value: float
    as_of: str
    currency: str = "USD"
    note: str = ""


@dataclass(frozen=True)
class Config:
    accounts: tuple[AccountConfig, ...]
    base_currency: str = "USD"
    db_path: Path = PROJECT_ROOT / "portfolio.db"
    raw_dir: Path = PROJECT_ROOT / "raw"
    manual: tuple[ManualAccount, ...] = ()

    def account(self, label: str) -> AccountConfig:
        for a in self.accounts:
            if a.label == label:
                return a
        raise ValueError(f"Unknown account {label!r}; configured: {', '.join(a.label for a in self.accounts)}")

    def redact(self, text: str) -> str:
        for a in self.accounts:
            if a.token:
                text = text.replace(a.token, "***")
        return text


def _path(value: str | None, default: str) -> Path:
    p = Path(value or default).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_config(env_file: Path | None = None) -> Config:
    env_file = env_file or PROJECT_ROOT / ".env"
    vals = {k: (v or "").strip() for k, v in dotenv_values(env_file).items()} if env_file.exists() else {}
    vals.update({k: v.strip() for k, v in os.environ.items() if k.startswith("IBKR_") or k in ("BASE_CURRENCY", "DB_PATH")})
    accounts = []
    for n in range(1, 10):
        p = f"IBKR_ACC{n}_"
        token, qid = vals.get(p + "TOKEN", ""), vals.get(p + "QUERY_ID", "")
        if token and qid:
            accounts.append(AccountConfig(vals.get(p + "LABEL") or f"acc{n}", token, qid))
    base = (vals.get("BASE_CURRENCY") or "USD").upper()
    return Config(tuple(accounts), base, _path(vals.get("DB_PATH"), "./portfolio.db"), _path(vals.get("RAW_DIR"), "./raw"),
                  load_manual(PROJECT_ROOT / "manual_accounts.json", base))


def load_manual(path: Path, base: str = "USD") -> tuple[ManualAccount, ...]:
    if not path.exists():
        return ()
    out = []
    for i, a in enumerate(json.loads(path.read_text())):
        try:
            m = ManualAccount(label=str(a["label"]), title=str(a.get("title") or a["label"]).upper(),
                              deposited=float(a["deposited"]), current_value=float(a["current_value"]),
                              as_of=str(a["as_of"]), currency=str(a.get("currency") or base).upper(),
                              note=str(a.get("note") or ""))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{path.name} entry {i}: needs label, deposited, current_value, as_of ({e})") from None
        if m.currency != base:
            raise ValueError(f"{path.name} {m.label}: currency {m.currency} != BASE_CURRENCY {base} (convert it first)")
        out.append(m)
    return tuple(out)
