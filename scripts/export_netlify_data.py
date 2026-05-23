"""Export Netlify-friendly JSON assets from the existing Python artifacts.

This script produces compact snapshots under:
    netlify/functions/_generated/

It is intended to run from the project root.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = PROJECT_ROOT / "netlify" / "functions" / "_generated"


def _ensure_dir() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def _load_jsonl_index(filename: str) -> Dict[str, dict]:
    data_dir = PROJECT_ROOT / "data"
    index: Dict[str, dict] = {}
    with (data_dir / filename).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            index[str(row["user_id"])] = row
    return index


def _select_representative_users(
    overview: dict,
    gnn_user_ids: list[str],
    model_service: Any,
    limit: int = 5,
) -> list[str]:
    high_risk = overview.get("high_risk_list", [])
    high_risk_ids = [str(row["user_id"]) for row in high_risk if "user_id" in row]
    if gnn_user_ids:
        scored = []
        for uid in gnn_user_ids:
            try:
                scored.append((float(model_service.predict_single(uid).risk_score), uid))
            except Exception:
                scored.append((0.0, uid))
        scored.sort(reverse=True)
        selected = [uid for _, uid in scored[:limit]]
        if len(selected) < limit:
            for uid in high_risk_ids:
                if uid not in selected:
                    selected.append(uid)
                if len(selected) >= limit:
                    break
        return selected[:limit]

    return high_risk_ids[:limit]


def _parse_hour(ts: str) -> int:
    if len(ts) < 13:
        return 0
    try:
        return int(ts[11:13])
    except ValueError:
        return 0


def _is_night_hour(hour: int) -> bool:
    return hour >= 22 or hour < 6


def _compute_tx_stats(user_id: str, model_service: Any) -> tuple[int, float, int]:
    twd_txs = model_service._twd_by_user.get(user_id, [])
    crypto_txs = model_service._crypto_by_user.get(user_id, [])
    trading_txs = model_service._trading_by_user.get(user_id, [])
    swap_txs = model_service._swap_by_user.get(user_id, [])

    all_txs = twd_txs + crypto_txs + trading_txs + swap_txs
    tx_volume = len(all_txs)
    if tx_volume == 0:
        return 0, 0.0, 0

    night_count = 0
    counterparties: set[str] = set()
    for tx in all_txs:
        ts = tx.get("created_at") or tx.get("updated_at") or tx.get("timestamp") or tx.get("time") or ""
        if _is_night_hour(_parse_hour(str(ts))):
            night_count += 1
        rel = tx.get("relation_user_id")
        if rel is not None:
            counterparties.add(str(rel))

    return tx_volume, night_count / tx_volume, len(counterparties)


def _compute_heatmap_data(all_txs: list[dict]) -> list[dict]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for tx in all_txs:
        ts = tx.get("created_at") or tx.get("updated_at") or tx.get("timestamp") or tx.get("time") or ""
        ts_str = str(ts)
        if len(ts_str) < 13:
            continue
        try:
            parsed = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            hour = int(ts_str[11:13])
            day = parsed.weekday()
            counts[(hour, day)] += 1
        except ValueError:
            continue
    return [{"hour": hour, "day": day, "count": count} for (hour, day), count in counts.items()]


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def main() -> None:
    _ensure_dir()

    # Keep the repo's virtualenv site-packages visible when this script runs
    # from a clean Python interpreter.
    venv_site_packages = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
    if venv_site_packages.exists():
        import sys

        sys.path.insert(0, str(venv_site_packages))
    sys.path.insert(0, str(PROJECT_ROOT))

    from backend.app.services.case_service import CaseService
    from backend.app.services.explainer_service import ExplainerService
    from backend.app.services.graph_service import GraphService
    from backend.app.services.model_service import ModelService
    from src.shared_features import _compute_user_features

    model_service = ModelService.get_instance()
    graph_service = GraphService.get_instance()
    explainer_service = ExplainerService.get_instance()
    case_service = CaseService.get_instance()
    explainer_service._ensure_loaded()
    raw_gnn = getattr(explainer_service, "_sage_data", None) or {}
    gnn_user_ids = [str(u) for u in raw_gnn.get("gnn_explanations", {}).keys()]

    overview = _read_json(PROJECT_ROOT / "outputs" / "overview.json")
    representative_user_ids = _select_representative_users(overview, gnn_user_ids, model_service, limit=5)
    representative_set = set(representative_user_ids)
    feature_stats = _read_json(PROJECT_ROOT / "outputs" / "feature_stats.json")
    case_status_seed = _read_json(PROJECT_ROOT / "outputs" / "case_status.json")
    audit_log_seed = _read_json(PROJECT_ROOT / "outputs" / "audit_log.json")
    case_status_map = {
        case.user_id: case.status
        for case in case_service.get_all_cases()
    }

    # Meta / small lookups
    lgbm_results = _load_pickle(PROJECT_ROOT / "outputs" / "lgbm_v13_no_graph_results.pkl")
    feature_names = list(lgbm_results.get("feature_names", []))
    meta = {
        "feature_names": feature_names,
        "high_risk_threshold": 0.35,
        "representative_user_ids": representative_user_ids,
    }

    # Build a compact per-user snapshot for the frontend.
    user_summaries: list[dict] = []
    user_ids = representative_user_ids
    for idx, user_id in enumerate(user_ids, start=1):
        print(f"  exporting representative user {idx}/{len(user_ids)}: {user_id}")

        prediction = model_service.predict_single(user_id)
        feat_dict = _compute_user_features(
            user_id=user_id,
            user_info=model_service._user_info[user_id],
            twd_txs=model_service._twd_by_user.get(user_id, []),
            crypto_txs=model_service._crypto_by_user.get(user_id, []),
            trading_txs=model_service._trading_by_user.get(user_id, []),
            swap_txs=model_service._swap_by_user.get(user_id, []),
        )

        top_features = prediction.top_features[:10]
        feature_values = {
            feature: float(feat_dict.get(feature, 0.0))
            for feature in top_features
        }

        all_txs = (
            model_service._twd_by_user.get(user_id, [])
            + model_service._crypto_by_user.get(user_id, [])
            + model_service._trading_by_user.get(user_id, [])
            + model_service._swap_by_user.get(user_id, [])
        )
        tx_volume, night_tx_ratio, counterparty_count = _compute_tx_stats(user_id, model_service)
        heatmap_data = _compute_heatmap_data(all_txs)

        case_status = case_status_map.get(user_id, "pending")

        user_summaries.append(
            {
                "user_id": user_id,
                "risk_score": float(prediction.risk_score),
                "risk_level": prediction.risk_level,
                "shap_values": {
                    feature: float(prediction.shap_values.get(feature, 0.0))
                    for feature in top_features
                },
                "top_features": top_features,
                "feature_values": feature_values,
                "tx_volume": tx_volume,
                "night_tx_ratio": night_tx_ratio,
                "counterparty_count": counterparty_count,
                "case_status": case_status,
                "fraud_label": model_service._fraud_labels.get(user_id),
                "heatmap_data": heatmap_data,
                "graphDegree": 0,  # filled later
                "cryptoTotalAmount": 0.0,  # filled later
                "isBlacklist": prediction.is_blacklist,
            }
        )

    # Build graph snapshots for each representative user and each hop.
    graph_service._ensure_graph()
    graph = graph_service._G
    assert graph is not None
    representative_graphs: Dict[str, dict] = {}
    for user_id in representative_user_ids:
        user_summary = next(row for row in user_summaries if row["user_id"] == user_id)
        user_summary["graphDegree"] = int(graph.degree(user_id)) if user_id in graph else 0
        crypto_txs = model_service._crypto_by_user.get(user_id, [])
        crypto_total = sum(
            tx.get("ori_samount", 0) * tx.get("twd_srate", 1) * 1e-16
            for tx in crypto_txs
        )
        user_summary["cryptoTotalAmount"] = float(crypto_total)

        per_hop: Dict[str, dict] = {}
        for hop in range(1, 6):
            subgraph = graph_service.get_subgraph(
                root_user_id=user_id,
                hop_depth=hop,
                case_status_map=case_status_map,
            )
            per_hop[str(hop)] = {
                "nodes": subgraph.nodes,
                "edges": subgraph.edges,
                "hasGnnExplanation": explainer_service.has_explanation(user_id),
                "gnnNodeMaskTop10": explainer_service.get_node_mask_top10(user_id)
                if explainer_service.has_explanation(user_id)
                else [],
            }
        representative_graphs[user_id] = per_hop

    # Normalise GNN explanations into edge-key lookups + top-10 node mask.
    gnn_payload: Dict[str, dict] = {}
    raw_explanations = raw_gnn.get("gnn_explanations", {})
    feature_names_gnn = raw_gnn.get("feature_names", [])
    user_ids_gnn = [str(u) for u in raw_gnn.get("user_ids", [])]
    for uid, exp_data in raw_explanations.items():
        uid_str = str(uid)
        if uid_str not in representative_set:
            continue
        if exp_data.get("error"):
            continue
        raw_edge_mask = exp_data.get("edge_mask")
        edge_mask: Dict[str, float] = {}
        if raw_edge_mask is not None:
            edge_order = [f"{u}|{v}" for u, v in graph.edges()]
            for i, val in enumerate(raw_edge_mask):
                if float(val) > 0.1 and i < len(edge_order):
                    edge_mask[edge_order[i]] = round(float(val), 4)

        raw_node_mask = exp_data.get("node_mask")
        node_mask_top10: list[tuple[str, float]] = []
        if raw_node_mask is not None:
            import numpy as np

            node_importances = None
            if isinstance(raw_node_mask, np.ndarray):
                if raw_node_mask.ndim == 2:
                    try:
                        user_idx = user_ids_gnn.index(uid_str)
                        node_importances = raw_node_mask[user_idx]
                    except ValueError:
                        node_importances = raw_node_mask[0]
                elif raw_node_mask.ndim == 1:
                    node_importances = raw_node_mask

            if node_importances is not None:
                sorted_idx = sorted(
                    range(len(node_importances)),
                    key=lambda i: abs(node_importances[i]),
                    reverse=True,
                )[:10]
                node_mask_top10 = [
                    (
                        feature_names_gnn[i] if i < len(feature_names_gnn) else f"feature_{i}",
                        round(float(node_importances[i]), 6),
                    )
                    for i in sorted_idx
                ]

        gnn_payload[uid_str] = {
            "user_id": uid_str,
            "edge_mask": edge_mask,
            "node_mask_top10": node_mask_top10,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    # Persist generated snapshots.
    _write_jsonl(GENERATED_DIR / "users.jsonl", user_summaries)
    _write_json(GENERATED_DIR / "graph.json", representative_graphs)
    _write_json(GENERATED_DIR / "overview.json", overview)
    _write_json(GENERATED_DIR / "feature_stats.json", feature_stats)
    _write_json(GENERATED_DIR / "meta.json", meta)
    _write_json(GENERATED_DIR / "case_status.json", case_status_seed)
    _write_json(GENERATED_DIR / "audit_log.json", audit_log_seed)
    _write_json(GENERATED_DIR / "gnn_explanations.json", gnn_payload)

    print(f"Exported Netlify assets to {GENERATED_DIR}")
    print(f"  users: {len(user_summaries)}")
    print(f"  gnn explanations: {len(gnn_payload)}")
    print(f"  representative users: {', '.join(representative_user_ids)}")


if __name__ == "__main__":
    main()
