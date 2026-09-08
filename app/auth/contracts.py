from dataclasses import dataclass


@dataclass(frozen=True)
class TrustedTenantContext:
    company_id: int
    credential_id: int
