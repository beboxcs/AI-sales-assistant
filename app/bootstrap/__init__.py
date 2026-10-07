"""Local operator bootstrap contracts for the backend."""

from app.bootstrap.local_tenant import (
    BootstrapDisposition,
    CompoundBootstrapResult,
    CompoundBootstrapSpec,
    InventoryScopeBootstrapResult,
    InventoryScopeBootstrapSpec,
    LocalTenantBootstrapConflictError,
    LocalTenantBootstrapDisabledError,
    LocalTenantBootstrapError,
    LocalTenantBootstrapInfrastructureError,
    LocalTenantBootstrapResult,
    LocalTenantBootstrapValidationError,
    bootstrap_local_tenant,
)

__all__ = [
    "BootstrapDisposition",
    "CompoundBootstrapResult",
    "CompoundBootstrapSpec",
    "InventoryScopeBootstrapResult",
    "InventoryScopeBootstrapSpec",
    "LocalTenantBootstrapConflictError",
    "LocalTenantBootstrapDisabledError",
    "LocalTenantBootstrapError",
    "LocalTenantBootstrapInfrastructureError",
    "LocalTenantBootstrapResult",
    "LocalTenantBootstrapValidationError",
    "bootstrap_local_tenant",
]
