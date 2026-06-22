#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
超算中心 Token 生成器（五领域 x 八类别，Py2/Py3 双兼容，零依赖）。
对应《超算中心Token生成操作手册_v2.md》gen_large.py 逻辑。

用法:
    python gen_large.py                 # 默认 scale=5000，约 240 万条
    python gen_large.py --scale 20      # 小测，约 9600 条
    python gen_large.py --scale 500     # ~24 万条
    python gen_large.py --scale 10000   # ~480 万条（超大规模）

输出:
    <script_dir>/output/*_token_A_B_YYYYMMDD_HHMMSS.csv   （5 份）
    <script_dir>/output/*_report_YYYYMMDD_HHMMSS.json     （5 份）
    <script_dir>/output/MASTER_REPORT_YYYYMMDD_HHMMSS.json（1 份）
"""
from __future__ import print_function

import os
import sys
import hashlib
import json
import csv
import random
import time
from datetime import datetime


def _mkdirp(path):
    """mkdir -p，兼容 Py2。"""
    if not path:
        return
    try:
        os.makedirs(path)
    except OSError as e:
        # errno.EEXIST == 17
        import errno as _errno
        if e.errno != _errno.EEXIST:
            raise


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
_mkdirp(OUTPUT_DIR)


def gen_token(prefix, idx):
    """生成 16 位大写 SHA256 前缀作为 token_id。"""
    raw = "%s-%d-%f" % (prefix, idx, time.time())
    if sys.version_info[0] >= 3:
        raw_bytes = raw.encode("utf-8")
    else:
        raw_bytes = raw
    return hashlib.sha256(raw_bytes).hexdigest()[:16].upper()


DOMAINS = {
    "healthcare": {
        "p": "HC",
        "cats": [
            ("radiology", "ct_image"),
            ("laboratory", "blood_test"),
            ("pathology", "pathology_slide"),
            ("cardiology", "ecg"),
            ("neurology", "ultrasound"),
            ("orthopedics", "x_ray"),
            ("pediatrics", "growth_record"),
            ("emergency", "triage"),
        ],
        "base": 1000,
        "fmt": "P%d",
    },
    "finance": {
        "p": "FN",
        "cats": [
            ("banking", "risk_control"),
            ("securities", "credit_report"),
            ("insurance", "transaction_record"),
            ("funds", "customer_profile"),
            ("trust", "anti_fraud"),
            ("consumer_finance", "credit_score"),
            ("fintech", "transaction_monitoring"),
            ("asset_management", "portfolio_data"),
        ],
        "base": 1200,
        "fmt": "E%d",
    },
    "manufacturing": {
        "p": "MF",
        "cats": [
            ("automotive", "production"),
            ("electronics", "quality_inspection"),
            ("machinery", "equipment_monitor"),
            ("chemical", "supply_chain"),
            ("steel", "energy_consumption"),
            ("pharmaceutical", "batch_record"),
            ("food_safety", "traceability"),
            ("textile", "quality_test"),
        ],
        "base": 800,
        "fmt": "F%d",
    },
    "transport": {
        "p": "TR",
        "cats": [
            ("road", "capacity_monitor"),
            ("railway", "tracking"),
            ("air", "cargo_tracking"),
            ("sea", "dispatch"),
            ("urban", "safety_alert"),
            ("logistics", "warehouse"),
            ("port", "container"),
            ("pipeline", "flow_monitor"),
        ],
        "base": 800,
        "fmt": "V%d",
    },
    "energy": {
        "p": "EN",
        "cats": [
            ("coal", "production"),
            ("electricity", "consumption"),
            ("oil_gas", "grid_dispatch"),
            ("renewable", "carbon_emission"),
            ("storage", "maintenance"),
            ("nuclear", "radiation_monitor"),
            ("hydro", "dam_level"),
            ("smart_grid", "load_forecast"),
        ],
        "base": 1000,
        "fmt": "P%d",
    },
}

CSV_FIELDNAMES = [
    "token_id", "domain", "category", "data_type", "entity_id",
    "data_quality_score", "token_level", "completeness", "accuracy",
    "timeliness", "compliance_score", "created_at",
]


def gen_domain(domain, scale):
    """按领域生成 token 记录列表。"""
    cfg = DOMAINS[domain]
    data = []
    total = int(cfg["base"] * scale)
    per_cat = max(1, total // len(cfg["cats"]))
    created_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    for cat_name, cat_type in cfg["cats"]:
        for i in range(per_cat):
            score = round(random.uniform(94, 100), 2)
            level = "A" if score >= 95 else "B"
            record = {
                "token_id": gen_token(cfg["p"], len(data)),
                "domain": domain,
                "category": cat_name,
                "data_type": cat_type,
                "entity_id": cfg["fmt"] % random.randint(10000, 999999),
                "data_quality_score": score,
                "token_level": level,
                "completeness": round(random.uniform(95, 100), 2),
                "accuracy": round(random.uniform(95, 100), 2),
                "timeliness": round(random.uniform(93, 100), 2),
                "compliance_score": 100.0,
                "created_at": created_at,
            }
            data.append(record)
    return data


def _write_csv(path, rows):
    """写 CSV：优先 Py3 newline=''；失败回退到 Py2 写法，统一 UTF-8。"""
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            w.writeheader()
            w.writerows(rows)
        return
    except TypeError:
        # Py2 不支持 newline= / encoding=
        pass
    import io
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def _write_json(path, obj):
    """写 JSON：ensure_ascii=False, indent=2。"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except TypeError:
        import io
        with io.open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, indent=2))


def _parse_scale(argv):
    scale = 5000
    if len(argv) >= 3 and argv[1] == "--scale":
        try:
            scale = int(argv[2])
        except (TypeError, ValueError):
            pass
    elif len(argv) == 2 and not argv[1].startswith("-"):
        try:
            scale = int(argv[1])
        except (TypeError, ValueError):
            pass
    return max(1, scale)


def main():
    scale = _parse_scale(sys.argv)
    print("[info] scale=%d (预计 total ~%d 条)" % (scale, scale * sum(d["base"] for d in DOMAINS.values())))
    sys.stdout.flush()

    domains_order = ["healthcare", "finance", "manufacturing", "transport", "energy"]
    grand_total = 0
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_reports = []

    for idx, d in enumerate(domains_order, 1):
        t0 = time.time()
        print("[%d/5] Generating %s ..." % (idx, d))
        sys.stdout.flush()
        data = gen_domain(d, scale)
        csv_path = os.path.join(OUTPUT_DIR, "%s_token_A_B_%s.csv" % (d, ts))
        _write_csv(csv_path, data)

        a_count = sum(1 for r in data if r["token_level"] == "A")
        avg_q = round(sum(r["data_quality_score"] for r in data) / len(data), 2)
        report = {
            "domain": d,
            "total_records": len(data),
            "a_level": a_count,
            "b_level": len(data) - a_count,
            "a_ratio_pct": round(100.0 * a_count / len(data), 2),
            "avg_quality": avg_q,
            "csv_file": os.path.basename(csv_path),
            "csv_size_mb": round(os.path.getsize(csv_path) / 1024.0 / 1024.0, 3),
            "generated_at": ts,
            "elapsed_sec": round(time.time() - t0, 2),
        }
        all_reports.append(report)
        jpath = os.path.join(OUTPUT_DIR, "%s_report_%s.json" % (d, ts))
        _write_json(jpath, report)
        print("  -> %s : %d 条 (A:%d, avg_quality=%.2f, cost=%.2fs)" % (
            os.path.basename(csv_path), len(data), a_count, avg_q, time.time() - t0))
        sys.stdout.flush()
        grand_total += len(data)

    master = {
        "summary": {
            "total_records": grand_total,
            "scale_factor": scale,
            "generated_at": ts,
            "output_dir": OUTPUT_DIR,
        },
        "domains": all_reports,
    }
    mpath = os.path.join(OUTPUT_DIR, "MASTER_REPORT_%s.json" % ts)
    _write_json(mpath, master)
    print("")
    print("===== DONE: %d 条记录 =====" % grand_total)
    print("  CSV 目录 : %s" % OUTPUT_DIR)
    print("  汇总报告 : %s" % mpath)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
