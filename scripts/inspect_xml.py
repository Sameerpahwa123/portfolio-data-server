"""Phase 1 discovery: describe what a Flex XML file actually contains.

    .venv/bin/python scripts/inspect_xml.py raw/*.xml   (default: newest file per label in ./raw)
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

from lxml import etree

REQUIRED = ("CashTransaction", "ChangeInNAV", "Trade")


def inspect(path: Path) -> list[str]:
    """Aggregate over every FlexStatement in the file (IBKR returns one statement per day for fd/td requests)."""
    root = etree.parse(str(path), etree.XMLParser(huge_tree=True, resolve_entities=False, no_network=True)).getroot()
    print(f"\n=== {path.name}  (root <{root.tag}> {dict(root.attrib)})")
    stmts = list(root.iter("FlexStatement"))
    accts = sorted({s.get("accountId") for s in stmts})
    froms = [s.get("fromDate", "") for s in stmts]
    tos = [s.get("toDate", "") for s in stmts]
    print(f"accountId={','.join(accts)} statements={len(stmts)} first fromDate={froms[0] if froms else None} "
          f"last toDate={tos[-1] if tos else None} period={stmts[0].get('period') if stmts else None!r}")
    counts: Counter[str] = Counter()
    attrs: dict[str, set[str]] = defaultdict(set)
    values: dict[str, Counter[str]] = defaultdict(Counter)
    acct_info: dict[str, str] = {}
    for stmt in stmts:
        for el in stmt.iterdescendants():
            if not isinstance(el.tag, str):
                continue
            counts[el.tag] += 1
            attrs[el.tag].update(el.attrib.keys())
            if el.tag == "CashTransaction":
                values["CashTransaction.type"][el.get("type", "")] += 1
                values["CashTransaction.currency"][el.get("currency", "")] += 1
                values["CashTransaction.levelOfDetail"][el.get("levelOfDetail", "")] += 1
            elif el.tag == "Trade":
                values["Trade.assetCategory"][el.get("assetCategory", "")] += 1
                values["Trade.levelOfDetail"][el.get("levelOfDetail", "")] += 1
            elif el.tag in ("ChangeInNAV", "EquitySummaryByReportDateInBase", "OpenPosition", "CashReportCurrency",
                            "FIFOPerformanceSummaryUnderlying"):
                values[f"{el.tag}.levelOfDetail"][el.get("levelOfDetail", "")] += 1
            elif el.tag == "AccountInformation" and not acct_info:
                acct_info = {k: el.get(k, "") for k in ("currency", "accountType", "dateOpened", "dateFunded", "dateClosed")}
    print("AccountInformation: " + ", ".join(f"{k}={v}" for k, v in acct_info.items()))
    print("\n-- element tags (rows across all statements) and attribute names")
    for tag in sorted(counts):
        names = sorted(attrs[tag])
        print(f"{tag} ({counts[tag]})" + (f": {', '.join(names)}" if names else "  [container]"))
    print("\n-- distinct values")
    for k in sorted(values):
        print(f"{k}: " + ", ".join(f"{v or '(empty)'}={n}" for v, n in values[k].most_common()))
    missing = [t for t in REQUIRED if counts[t] == 0]
    for t in missing:
        print(f"!! MISSING {t} rows")
    return [f"{','.join(accts)}:{t}" for t in missing]


def main() -> int:
    raw = Path(__file__).resolve().parent.parent / "raw"
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        newest: dict[str, Path] = {}
        for p in sorted(raw.glob("*.xml")):
            newest[p.name.split("_")[0]] = p
        paths = list(newest.values())
    missing = [m for p in paths for m in inspect(p)]
    print("\nSUMMARY: " + ("all required sections present" if not missing else "missing: " + ", ".join(missing)))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
