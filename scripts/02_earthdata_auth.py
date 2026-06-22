#!/usr/bin/env python3
"""
02_earthdata_auth.py — set up and test a NASA Earthdata Login via `earthaccess`.

NOT needed for the EOGS reproduction milestone. This is for Paper 1 (GEDI /
ICESat-2) and Paper 2 (HLS), so we set it up early per HANDOFF §8.

First create a free Earthdata account: https://urs.earthdata.nasa.gov/users/new

Auth options (earthaccess tries these in order):
  1. Environment variables EARTHDATA_USERNAME / EARTHDATA_PASSWORD
  2. A ~/.netrc entry for urs.earthdata.nasa.gov
  3. Interactive prompt (this script will prompt if nothing else is found)

Run:  python scripts/02_earthdata_auth.py
"""
import sys


def main() -> int:
    try:
        import earthaccess
    except ImportError:
        print("earthaccess not installed. Run: pip install earthaccess")
        return 1

    print("Authenticating to NASA Earthdata ...")
    # persist_to_netrc writes ~/.netrc so future sessions are non-interactive.
    try:
        auth = earthaccess.login(strategy="interactive", persist=True)
    except Exception as e:  # noqa: BLE001
        print(f"Login failed: {e}")
        return 1

    if not getattr(auth, "authenticated", False):
        print("Not authenticated. Check your Earthdata credentials.")
        return 1

    print("Authenticated to Earthdata.")

    # Smoke test: a tiny GEDI L2A search (no download) to prove access works.
    try:
        results = earthaccess.search_data(
            short_name="GEDI02_A",
            bounding_box=(-81.7, 30.3, -81.6, 30.4),  # ~Jacksonville FL (DFC2019 JAX area)
            count=1,
        )
        print(f"GEDI L2A search OK — {len(results)} granule(s) found for the test bbox.")
    except Exception as e:  # noqa: BLE001
        print(f"(Auth worked, but the test search failed: {e})")

    print("Done. Credentials persisted to ~/.netrc for future sessions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
