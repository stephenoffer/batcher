"""`io.formats.saas` — SaaS connectors built on the HTTP source foundation.

Each module registers its source (and, for Google Sheets, its sink) on import: GitHub
(`github`), Salesforce Bulk API 2.0 (`salesforce`), Google Sheets (`sheets`), SharePoint and
OneDrive through Microsoft Graph (`msgraph`), and the Airbyte protocol bridge (`airbyte`).
All but the Google Application Default Credentials path need nothing beyond the standard
library and pyarrow. None has been verified against the live service yet; see
``tests/PENDING_VERIFICATION.md``.
"""

from __future__ import annotations

from batcher.io.formats.saas.airbyte import AirbyteSource
from batcher.io.formats.saas.github import GitHubSource
from batcher.io.formats.saas.msgraph import SharePointSource
from batcher.io.formats.saas.salesforce import SalesforceSource
from batcher.io.formats.saas.sheets import GoogleSheetsSink, GoogleSheetsSource

__all__ = [
    "AirbyteSource",
    "GitHubSource",
    "GoogleSheetsSink",
    "GoogleSheetsSource",
    "SalesforceSource",
    "SharePointSource",
]
