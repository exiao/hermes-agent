/**
 * WhatsApp anti-ban middleware for the Hermes bridge.
 *
 * WhatsApp's automated-behavior detection flags accounts whose outbound
 * traffic looks bot-shaped: evenly-spaced sends, no typing/presence, high
 * per-recipient volume, and byte-identical bodies fanned out to many chats.
 * This module makes broadcast-style sends look more human without changing
 * message content in any user-visible way.
 *
 * It is self-contained (no third-party "anti-ban" packages — those have a
 * history of credential-stealing supply-chain attacks, e.g. the April 2026
 * `lotusbail` incident). Everything here is plain JS + Node stdlib.
 *
 * DESIGN
 *   - Pure, dependency-free, unit-testable in isolation (mirrors the
 *     outbound_ids.js / bridge_helpers.js pattern).
 *   - DEFAULT OFF. With WHATSAPP_ANTIBAN unset/false, createAntiban()
 *     returns a no-op passthrough so existing deployments are unaffected.
 *   - Interactive replies stay snappy; only sends explicitly marked
 *     `broadcast: true` (the alerts wrapper's fan-out path) pay the full
 *     jitter + rate-cap cost. A high reply ratio *lowers* ban risk, so we
 *     never slow the reply path down.
 *
 * KNOBS (env, all optional; only read when WHATSAPP_ANTIBAN is truthy):
 *   WHATSAPP_ANTIBAN                      master switch (default off)
 *   WHATSAPP_ANTIBAN_JITTER_MIN_MS        broadcast pre-send delay floor   (default 3000)
 *   WHATSAPP_ANTIBAN_JITTER_MAX_MS        broadcast pre-send delay ceiling (default 15000)
 *   WHATSAPP_ANTIBAN_TYPING               send composing presence before a send (default on)
 *   WHATSAPP_ANTIBAN_TYPING_MS_PER_CHAR   typing dwell per char            (default 35)
 *   WHATSAPP_ANTIBAN_TYPING_MAX_MS        typing dwell cap                 (default 6000)
 *   WHATSAPP_ANTIBAN_MAX_PER_RECIPIENT_HR per-chat broadcast cap/hour, 0=off (default 6)
 *   WHATSAPP_ANTIBAN_MAX_PER_HOUR         global broadcast cap/hour, 0=off (default 60)
 *   WHATSAPP_ANTIBAN_VARY_BODY            append invisible variation so no
 *                                         two broadcast bodies are byte-identical
 *                                         (default OFF — weak measure; real copy
 *                                         variation belongs in the alert generator)
 */

const HOUR_MS = 60 * 60 * 1000;

function envFlag(env, name, defaultOn) {
  const raw = env[name];
  if (raw === undefined || raw === '') return defaultOn;
  return ['1', 'true', 'yes', 'on'].includes(String(raw).toLowerCase());
}

function envInt(env, name, defaultVal) {
  const n = parseInt(env[name], 10);
  return Number.isFinite(n) ? n : defaultVal;
}

/**
 * Standard-normal sample via Box-Muller. Returns a value ~N(0,1).
 */
function standardNormal(rng) {
  let u = 0;
  let v = 0;
  // Avoid log(0).
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/**
 * Gaussian-distributed delay in [minMs, maxMs].
 *
 * WhatsApp flags UNIFORM/fixed delays (a giveaway of scripted sends). A
 * gaussian centered on the midpoint with the range spanning ~3 sigma
 * clusters around a human-plausible middle while still varying, and is
 * clamped so it never exceeds the caller's bounds.
 */
function gaussianDelay(minMs, maxMs, rng = Math.random) {
  if (!Number.isFinite(minMs)) minMs = 0;
  if (!Number.isFinite(maxMs) || maxMs < minMs) maxMs = minMs;
  if (maxMs === minMs) return minMs;
  const mean = (minMs + maxMs) / 2;
  const stddev = (maxMs - minMs) / 6; // ~99.7% of mass inside [min,max]
  const raw = mean + standardNormal(rng) * stddev;
  return Math.max(minMs, Math.min(maxMs, Math.round(raw)));
}

/**
 * Fixed-window-free sliding rate check.
 *
 * Tracks send timestamps per key and prunes anything older than one hour.
 * `check(key)` returns { allowed, retryAfterMs }. On allowed sends the
 * caller must call `record(key)` so the window advances. Global limit is
 * tracked under a reserved GLOBAL bucket.
 */
function createRateLimiter({ perRecipientHour, perHour }) {
  const GLOBAL = '\u0000global';
  const buckets = new Map(); // key -> number[] (send epoch ms, ascending)

  function prune(arr, now) {
    const cutoff = now - HOUR_MS;
    let i = 0;
    while (i < arr.length && arr[i] <= cutoff) i += 1;
    if (i > 0) arr.splice(0, i);
    return arr;
  }

  function bucket(key) {
    let arr = buckets.get(key);
    if (!arr) {
      arr = [];
      buckets.set(key, arr);
    }
    return arr;
  }

  function retryAfter(arr, limit, now) {
    // Oldest send in-window will expire at arr[len-limit] + HOUR_MS.
    const idx = arr.length - limit;
    if (idx < 0) return 0;
    return Math.max(0, arr[idx] + HOUR_MS - now);
  }

  function check(key, now = Date.now()) {
    let wait = 0;
    if (perRecipientHour > 0) {
      const arr = prune(bucket(key), now);
      if (arr.length >= perRecipientHour) {
        wait = Math.max(wait, retryAfter(arr, perRecipientHour, now));
      }
    }
    if (perHour > 0) {
      const arr = prune(bucket(GLOBAL), now);
      if (arr.length >= perHour) {
        wait = Math.max(wait, retryAfter(arr, perHour, now));
      }
    }
    return { allowed: wait === 0, retryAfterMs: wait };
  }

  function record(key, now = Date.now()) {
    if (perRecipientHour > 0) bucket(key).push(now);
    if (perHour > 0) bucket(GLOBAL).push(now);
  }

  return { check, record };
}

/**
 * Append an invisible, per-send-varying suffix so no two broadcast bodies
 * are byte-identical. Uses a small rotating count of zero-width spaces
 * (U+200B), which render as nothing in every WhatsApp client. This is a
 * WEAK measure and off by default: genuine copy variation should come from
 * the alert generator upstream, not the transport. Never applied to
 * interactive replies.
 */
function makeBodyVarier() {
  let counter = 0;
  return function varyBody(text) {
    const t = String(text == null ? '' : text);
    if (!t) return t;
    counter = (counter % 7) + 1; // 1..7 zero-width spaces, cycling
    return t + '\u200B'.repeat(counter);
  };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, Math.max(0, ms || 0)));
}

/**
 * Build the anti-ban middleware.
 *
 * Returns an object:
 *   enabled            boolean
 *   varyBody(text)     -> possibly-varied text (identity when disabled or VARY_BODY off)
 *   beforeSend(opts)   -> Promise<void>, applies typing presence + gaussian
 *                         jitter + rate-cap wait for a broadcast send.
 *
 * beforeSend(opts):
 *   chatId       recipient JID (rate-limit key)
 *   broadcast    when false/undefined, this is an interactive send: no
 *                jitter, no rate cap. Typing presence still applies (cheap,
 *                human-like, helps reply-shaped traffic too).
 *   textLength   message length, drives typing dwell
 *   sock         Baileys socket (for sendPresenceUpdate); optional
 *   sleepFn/rng  injectable for tests
 */
function createAntiban({ env = process.env } = {}) {
  const enabled = envFlag(env, 'WHATSAPP_ANTIBAN', false);

  if (!enabled) {
    return {
      enabled: false,
      varyBody: (text) => text,
      beforeSend: async () => {},
    };
  }

  const jitterMin = Math.max(0, envInt(env, 'WHATSAPP_ANTIBAN_JITTER_MIN_MS', 3000));
  const jitterMax = Math.max(jitterMin, envInt(env, 'WHATSAPP_ANTIBAN_JITTER_MAX_MS', 15000));
  const typingOn = envFlag(env, 'WHATSAPP_ANTIBAN_TYPING', true);
  const typingPerChar = Math.max(0, envInt(env, 'WHATSAPP_ANTIBAN_TYPING_MS_PER_CHAR', 35));
  const typingMax = Math.max(0, envInt(env, 'WHATSAPP_ANTIBAN_TYPING_MAX_MS', 6000));
  const perRecipientHour = Math.max(0, envInt(env, 'WHATSAPP_ANTIBAN_MAX_PER_RECIPIENT_HR', 6));
  const perHour = Math.max(0, envInt(env, 'WHATSAPP_ANTIBAN_MAX_PER_HOUR', 60));
  const varyOn = envFlag(env, 'WHATSAPP_ANTIBAN_VARY_BODY', false);

  const limiter = createRateLimiter({ perRecipientHour, perHour });
  const bodyVarier = makeBodyVarier();

  function varyBody(text) {
    return varyOn ? bodyVarier(text) : text;
  }

  function typingDwell(textLength) {
    const len = Number.isFinite(textLength) ? textLength : 0;
    return Math.min(typingMax, len * typingPerChar);
  }

  async function beforeSend(opts = {}) {
    const {
      chatId,
      broadcast = false,
      textLength = 0,
      sock = null,
      sleepFn = sleep,
      rng = Math.random,
      now = Date.now(),
    } = opts;

    // Rate cap: broadcasts only. Wait out the window rather than dropping —
    // the alerts wrapper sends sequentially so a short wait paces the batch.
    if (broadcast && (perRecipientHour > 0 || perHour > 0)) {
      const { retryAfterMs } = limiter.check(chatId, now);
      if (retryAfterMs > 0) {
        await sleepFn(retryAfterMs);
      }
    }

    // Typing presence: cheap and human-like. Applied to any send when on.
    if (typingOn && sock && chatId) {
      try {
        await sock.sendPresenceUpdate('composing', chatId);
        await sleepFn(typingDwell(textLength));
        await sock.sendPresenceUpdate('paused', chatId);
      } catch {
        // Presence is best-effort; never block the send on it.
      }
    }

    // Gaussian jitter: broadcasts only, so interactive replies stay instant.
    if (broadcast) {
      await sleepFn(gaussianDelay(jitterMin, jitterMax, rng));
    }

    if (broadcast) {
      limiter.record(chatId, now);
    }
  }

  return {
    enabled: true,
    varyBody,
    beforeSend,
    // Exposed for tests / diagnostics.
    _config: { jitterMin, jitterMax, typingOn, typingPerChar, typingMax, perRecipientHour, perHour, varyOn },
  };
}

export {
  createAntiban,
  createRateLimiter,
  gaussianDelay,
  standardNormal,
  makeBodyVarier,
};
