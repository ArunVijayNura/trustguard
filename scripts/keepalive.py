"""Keep the hosted reviewer console awake on Streamlit Community Cloud.

Free Streamlit apps sleep after a period without visitors. This opens the app
in a headless browser; if it is asleep it clicks the wake button, otherwise
the visit itself resets the sleep timer. Run on a schedule by
.github/workflows/keepalive.yml.
"""

import sys

from playwright.sync_api import sync_playwright

APP_URL = "https://trustguard.streamlit.app/"
WAKE_BUTTON = "Yes, get this app back up!"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            page.goto(APP_URL, timeout=90_000, wait_until="domcontentloaded")
            page.wait_for_timeout(8_000)
            button = page.get_by_role("button", name=WAKE_BUTTON)
            if button.count() > 0:
                button.first.click()
                print(f"WOKE UP  {APP_URL}")
                page.wait_for_timeout(90_000)  # give the app time to boot
            else:
                print(f"AWAKE    {APP_URL}")
                page.wait_for_timeout(20_000)  # keep a session open briefly
        except Exception as err:
            print(f"FAILED   {APP_URL}: {err}")
            sys.exit(1)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
