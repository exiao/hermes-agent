# Session storage and recovery

The bridge uses the protocol version bundled with its pinned Baileys release. Retry requests retrieve the original message from the bounded in-memory message cache; a missing entry returns no message. Messages older than the cache or a process restart are unavailable for retry.

Credentials and encryption keys use SQLite in the session directory. On first open, legacy JSON is imported transactionally; the original key files are retained. After migration SQLite is authoritative. Only credentials and LID mappings are mirrored to JSON for the existing Python pairing and identity readers. Restriction events received from WhatsApp are logged locally without polling or sending diagnostics.

Before migration, stop the bridge and privately back up the whole session directory. Do not run old and new bridge versions against the same directory concurrently. Never delete credentials to fix a database error. Once the new bridge has connected and updated keys, old JSON key files are stale: reverting code alone is not a safe auth rollback.

For an official-app-only observation interval, set the existing `WHATSAPP_ENABLED=false` in the Hermes environment and restart the gateway through its service manager. Verify no bridge process remains. This disables bot replies as well as outbound sends. Keep the flag disabled until the operator explicitly resumes; no timer or automatic relinking is installed.
