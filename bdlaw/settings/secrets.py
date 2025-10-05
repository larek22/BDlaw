"""Keyring-backed secret storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import keyring


@dataclass
class SecretStore:
    service_name: str = "bdlaw"

    def set(self, key: str, value: str) -> None:
        keyring.set_password(self.service_name, key, value)

    def get(self, key: str) -> Optional[str]:
        return keyring.get_password(self.service_name, key)

    def delete(self, key: str) -> None:
        keyring.delete_password(self.service_name, key)


__all__ = ["SecretStore"]
