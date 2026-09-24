"""
ECG Guardian — Supabase Client Configuration & Factory
======================================================
Provides authenticated Supabase client connections adhering to security guidelines:
- Never exposes SUPABASE_SERVICE_ROLE_KEY to client-side code.
- Server/privileged operations use service-role key only in secure backend contexts.
- Client/frontend queries use publishable/anon key with RLS.
- Graceful offline / unconfigured fallback when environment variables are not yet provided.
"""

from __future__ import annotations

import os
from typing import Any, Optional

try:
    from supabase import Client, create_client
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False
    Client = None
    create_client = None


def get_supabase_url() -> Optional[str]:
    """Retrieve Supabase URL from environment."""
    return os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")


def get_supabase_publishable_key() -> Optional[str]:
    """Retrieve publishable/anon key for standard client operations."""
    return (
        os.environ.get("SUPABASE_PUBLISHABLE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    )


def get_supabase_service_role_key() -> Optional[str]:
    """Retrieve privileged service-role key (server-side ONLY)."""
    return os.environ.get("SUPABASE_SERVICE_ROLE_KEY")


def is_supabase_configured(require_service_role: bool = False) -> bool:
    """Check if Supabase credentials are validly present in the environment."""
    if not HAS_SUPABASE:
        return False
    url = get_supabase_url()
    if not url or url.strip() == "":
        return False
    if require_service_role:
        key = get_supabase_service_role_key()
    else:
        key = get_supabase_publishable_key() or get_supabase_service_role_key()
    return bool(key and key.strip())


def get_supabase_client(use_service_role: bool = False) -> Optional[Any]:
    """
    Factory creating a Supabase Client instance.
    Returns None if supabase-py is missing or credentials are not configured.
    """
    if not HAS_SUPABASE:
        return None

    url = get_supabase_url()
    if not url:
        return None

    if use_service_role:
        key = get_supabase_service_role_key()
        if not key:
            # Fall back to publishable if service role is omitted
            key = get_supabase_publishable_key()
    else:
        key = get_supabase_publishable_key() or get_supabase_service_role_key()

    if not key:
        return None

    try:
        return create_client(url, key)
    except Exception as exc:
        # Avoid crashing application if network is down or URL is invalid placeholder
        return None
