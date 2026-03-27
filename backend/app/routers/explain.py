"""
Explain router — GNNExplainer endpoints (delegated to graph.py).

This module is kept for backward compatibility.
The main GNN explanation endpoints are in graph.py.
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()

# All explain endpoints are now in graph.py (/api/explain/gnn/{user_id})
# This empty router is kept so main.py import doesn't break.
