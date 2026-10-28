# Streamlit Community Cloud

Repository: https://github.com/notevensmart/Sydneywater_chatbot

Create an app in https://share.streamlit.io with branch `main`, entrypoint `app.py`, and Python **3.12**. Suggested subdomain: `sydney-water-student-demo` (availability is checked by Streamlit; it is not reserved yet).

In Advanced settings → Secrets, add `APP_ENV` as a TOML multiline literal string containing the exact contents of your local `.env`. Do not commit `.env` or secrets.toml. At startup the app provisions a private local `.env` from that secret only when no file exists. To change cloud configuration, update APP_ENV and reboot the app so the deployment filesystem is recreated; confirm the new configuration before use.

Model selection and credentials come exclusively from `.env`: GROQ_MODEL/GROQ_API_KEY for Groq, or explicit LLM_PROVIDER=google with LLM_MODEL and GOOGLE_API_KEY/GEMINI_API_KEY for Google. Missing configuration fails closed; there is no OS-key or alternate-model fallback.

The committed SQLite corpus contains only extracted public Sydney Water documentation. It retains the existing 30-day source-review expiry; refresh and review the corpus before expiry, then publish the updated database. A stable URL does not guarantee continuous uptime or provider quota. This student demo still has the retrieval/test failures described in data/SOURCE_IMPORT_REPORT.md.
