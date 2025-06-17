"""Truss FastAPI application package.

Exposes the FastAPI ``app`` instance so external tooling (e.g. uvicorn
or tests) can simply run ``uvicorn truss.api:app``.
"""
from __future__ import annotations


from .main import app  # noqa: F401  (re-export for uvicorn) 
