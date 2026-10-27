"""Test settings that must be in place before the app is imported.

The API starts a background thread that fetches from AEMO and Open-Meteo.
Tests must never do that: it is slow, it needs the network, and it would write
to the real database. The switch is read at import time, so it is set here.
"""

import os

os.environ["REFRESH_ENABLED"] = "0"
