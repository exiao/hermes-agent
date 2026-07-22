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
 *   - DEFAULT ON. createAntiban() is enabled unless WHATSAPP_ANTIBAN is
 *     explicitly 0/false.
 *   - TWO pacing profiles:
 *       reply (broadcast:false) — MINIMAL + UNCAPPED. A brief "typing…"
 *         indicator + tiny gaussian jitter (default 0.1-0.5s). No rate cap.
 *         Designed to run OUTSIDE the serialised send queue so several
 *         concurrent conversations pace in parallel and stay real-time. An LLM
 *         reply already arrives after variable generation latency (itself
 *         human-shaped); this just flashes a typing state and avoids a
 *         byte-perfect zero-latency send. Set the floor to 0 for near-instant.
 *       broadcast (broadcast:true) — HEAVY. Big gaussian jitter (default 3-15s),
 *         typing dwell, and per-recipient + global sliding-window rate caps.
 *         The ban-prone fan-out path. Runs INSIDE the send queue.
 *
 * KNOBS (env, all optional):
 *   WHATSAPP_ANTIBAN                      master switch (default ON; 0 disables)
 *   -- reply path --
 *   WHATSAPP_ANTIBAN_REPLY_JITTER_MIN_MS  reply typing/delay floor   (default 100)
 *   WHATSAPP_ANTIBAN_REPLY_JITTER_MAX_MS  reply typing/delay ceiling (default 500)
 *   -- broadcast path --
 *   WHATSAPP_ANTIBAN_JITTER_MIN_MS        broadcast pre-send delay floor   (default 3000)
 *   WHATSAPP_ANTIBAN_JITTER_MAX_MS        broadcast pre-send delay ceiling (default 15000)
 *   -- shared --
 *   WHATSAPP_ANTIBAN_TYPING               send composing presence before a send (default on)
 *   WHATSAPP_ANTIBAN_TYPING_MS_PER_CHAR   broadcast typing dwell per char  (default 35)
 *   WHATSAPP_ANTIBAN_TYPING_MAX_MS        broadcast typing dwell cap       (default 6000)
 *   WHATSAPP_ANTIBAN_MAX_PER_RECIPIENT_HR per-chat BROADCAST cap/hour, 0=off (default 240)
 *   WHATSAPP_ANTIBAN_MAX_PER_HOUR         global BROADCAST cap/hour, 0=off (default 60)
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
  // Default ON. The middleware only ever paces sends explicitly flagged
  // `broadcast: true`; interactive replies are never touched (see beforeSend),
  // so enabling by default adds zero latency to normal conversation and is
  // safe for any WhatsApp deployment — it stays inert until something sends a
  // broadcast. Set WHATSAPP_ANTIBAN=0/false to hard-disable.
  const enabled = envFlag(env, 'WHATSAPP_ANTIBAN', true);

  if (!enabled) {
    return {
      enabled: false,
      varyBody: (text) => text,
      beforeSend: async () => {},
    };
  }

  const jitterMin = Math.max(0, envInt(env, 'WHATSAPP_ANTIBAN_JITTER_MIN_MS', 3000));
  const jitterMax = Math.max(jitterMin, envInt(env, 'WHATSAPP_ANTIBAN_JITTER_MAX_MS', 15000));
  // Reply pacing is deliberately MINIMAL: an LLM reply already arrives after
  // variable generation time (2-8s), which is itself a human-shaped signal, so
  // the reply path only needs to (a) flash a brief "typing…" indicator and
  // (b) avoid a byte-perfect zero-latency send. A ~0.1-0.5s dwell does both
  // without being perceptibly slow. NO rate cap on replies. Set floor to 0 for
  // an essentially instant reply that still shows a momentary typing state.
  const replyJitterMin = Math.max(0, envInt(env, 'WHATSAPP_ANTIBAN_REPLY_JITTER_MIN_MS', 100));
  const replyJitterMax = Math.max(replyJitterMin, envInt(env, 'WHATSAPP_ANTIBAN_REPLY_JITTER_MAX_MS', 500));
  const typingOn = envFlag(env, 'WHATSAPP_ANTIBAN_TYPING', true);
  const typingPerChar = Math.max(0, envInt(env, 'WHATSAPP_ANTIBAN_TYPING_MS_PER_CHAR', 35));
  const typingMax = Math.max(0, envInt(env, 'WHATSAPP_ANTIBAN_TYPING_MAX_MS', 6000));
  // Per-recipient cap only ever applies to broadcasts; replies are uncapped so a
  // real-time back-and-forth is never throttled. Default set high for headroom.
  const perRecipientHour = Math.max(0, envInt(env, 'WHATSAPP_ANTIBAN_MAX_PER_RECIPIENT_HR', 240));
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

  // Show a real "typing…" indicator for `dwellMs`, then let the caller send
  // (the actual message clears the composing state). Best-effort: presence
  // failures never block the send.
  async function showTyping(sock, chatId, dwellMs, sleepFn) {
    if (typingOn && sock && chatId) {
      let dwelled = false;
      try {
        await sock.sendPresenceUpdate('composing', chatId);
        await sleepFn(dwellMs);
        dwelled = true;
        await sock.sendPresenceUpdate('paused', chatId);
        return;
      } catch {
        // Presence update failed. If the dwell already completed (only the
        // trailing 'paused' update threw), don't sleep it a second time —
        // that would double the delay (up to +typingMax per send). Only fall
        // through to a plain sleep when the dwell never ran.
        if (dwelled) return;
      }
    }
    await sleepFn(dwellMs);
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

    if (broadcast) {
      // Broadcast fan-out: the ban-prone path. Heavy pacing + rate caps.
      // Runs sequentially (the alerts wrapper sends one at a time), so the
      // rate window stays accurate.
      //
      // Track the EFFECTIVE send time. The message is not actually put on the
      // wire until every pacing delay below (rate-cap wait, typing dwell,
      // gaussian jitter) has elapsed, so recording at the original `now` would
      // stamp the send in the past. On an over-cap broadcast that waits out the
      // window (e.g. 30s) that error makes the sliding window expire the entry
      // ~30s early, so the limiter drifts more permissive than configured.
      // Advance `sendAt` by each delay we impose and record at that time.
      let sendAt = now;
      if (perRecipientHour > 0 || perHour > 0) {
        const { retryAfterMs } = limiter.check(chatId, now);
        if (retryAfterMs > 0) {
          await sleepFn(retryAfterMs);
          sendAt += retryAfterMs;
        }
      }
      const dwellMs = typingDwell(textLength);
      await showTyping(sock, chatId, dwellMs, sleepFn);
      sendAt += dwellMs;
      // Gaussian jitter (uniform/fixed delays are the scripted-send tell).
      const jitterMs = gaussianDelay(jitterMin, jitterMax, rng);
      await sleepFn(jitterMs);
      sendAt += jitterMs;
      limiter.record(chatId, sendAt);
      return;
    }

    // Reply path: light + uncapped so a real-time convo (incl. several people
    // at once) stays snappy. The jitter doubles as the "typing" duration, so
    // fast replies still show a brief composing indicator and never land with
    // robotic zero latency. Meant to run OUTSIDE the serialised send queue so
    // concurrent chats pace in parallel.
    const dwell = gaussianDelay(replyJitterMin, replyJitterMax, rng);
    await showTyping(sock, chatId, dwell, sleepFn);
  }

  return {
    enabled: true,
    varyBody,
    beforeSend,
    // Exposed for tests / diagnostics.
    _config: {
      jitterMin, jitterMax, replyJitterMin, replyJitterMax,
      typingOn, typingPerChar, typingMax, perRecipientHour, perHour, varyOn,
    },
  };
}

export {
  createAntiban,
  createRateLimiter,
  gaussianDelay,
  standardNormal,
  makeBodyVarier,
};
