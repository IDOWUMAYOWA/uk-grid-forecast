from collections.abc import Iterator

import pytest

from gridcast.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Settings are cached; reset between tests so env overrides take effect."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
