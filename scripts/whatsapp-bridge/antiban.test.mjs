/**
 * Tests for the WhatsApp anti-ban middleware (antiban.js).
 *
 * These exercise the pure logic directly — no live WhatsApp socket, no
 * express. Time and randomness are injected so the tests are deterministic
 * and fast (no real sleeps).
 *
 * Run: node scripts/whatsapp-bridge/antiban.test.mjs
 */

import { strict as assert } from 'node:assert';
import {
  createAntiban,
  createRateLimiter,
  gaussianDelay,
  standardNormal,
  makeBodyVarier,
} from './antiban.js';

// ------------------------------------------------------------------
// gaussianDelay: bounded, varied, non-uniform, centered
// ------------------------------------------------------------------
{
  const min = 3000;
  const max = 15000;
  const samples = Array.from({ length: 5000 }, () => gaussianDelay(min, max));

  assert.ok(samples.every(v => v >= min && v <= max), 'all samples within [min,max]');

  const distinct = new Set(samples);
  assert.ok(distinct.size > 100, 'samples vary widely (not a fixed delay)');

  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const mid = (min + max) / 2;
  // Mean should cluster near the midpoint (gaussian), within 5% of range.
  assert.ok(Math.abs(mean - mid) < (max - min) * 0.05, `mean ${mean} near midpoint ${mid}`);

  // Degenerate range collapses cleanly.
  assert.strictEqual(gaussianDelay(500, 500), 500, 'equal bounds return the bound');
  assert.strictEqual(gaussianDelay(1000, 200), 1000, 'max<min collapses to min');
  console.log('  ✓ gaussianDelay bounded, varied, centered');
}

// -- standardNormal is roughly N(0,1) -------------------------------
{
  const n = 20000;
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < n; i += 1) {
    const x = standardNormal(Math.random);
    sum += x;
    sumSq += x * x;
  }
  const mean = sum / n;
  const variance = sumSq / n - mean * mean;
  assert.ok(Math.abs(mean) < 0.05, `mean ~0 (got ${mean.toFixed(3)})`);
  assert.ok(Math.abs(variance - 1) < 0.1, `variance ~1 (got ${variance.toFixed(3)})`);
  console.log('  ✓ standardNormal ~ N(0,1)');
}

// ------------------------------------------------------------------
// createRateLimiter: per-recipient + global sliding window
// ------------------------------------------------------------------
{
  const limiter = createRateLimiter({ perRecipientHour: 2, perHour: 100 });
  const chat = 'a@s.whatsapp.net';
  const t0 = 1_000_000;

  assert.ok(limiter.check(chat, t0).allowed, '1st allowed');
  limiter.record(chat, t0);
  assert.ok(limiter.check(chat, t0 + 1000).allowed, '2nd allowed');
  limiter.record(chat, t0 + 1000);

  const blocked = limiter.check(chat, t0 + 2000);
  assert.ok(!blocked.allowed, '3rd blocked (per-recipient cap=2)');
  assert.ok(blocked.retryAfterMs > 0, 'retryAfterMs positive');
  // Oldest send at t0 expires at t0 + 1h; retry should point roughly there.
  assert.ok(Math.abs(blocked.retryAfterMs - (t0 + 3_600_000 - (t0 + 2000))) < 5, 'retry ~ window expiry');

  // A different recipient is independent.
  assert.ok(limiter.check('b@s.whatsapp.net', t0 + 2000).allowed, 'other recipient unaffected');

  // After the window slides past the oldest send, the recipient frees up.
  assert.ok(limiter.check(chat, t0 + 3_600_001).allowed, 'freed after 1st ages out');
  console.log('  ✓ rate limiter per-recipient sliding window');
}

{
  // Global cap independent of per-recipient.
  const limiter = createRateLimiter({ perRecipientHour: 0, perHour: 3 });
  const t0 = 5_000_000;
  for (let i = 0; i < 3; i += 1) {
    const c = `u${i}@s.whatsapp.net`;
    assert.ok(limiter.check(c, t0 + i).allowed, `global send ${i} allowed`);
    limiter.record(c, t0 + i);
  }
  const blocked = limiter.check('u9@s.whatsapp.net', t0 + 10);
  assert.ok(!blocked.allowed, 'global cap blocks the 4th across all recipients');
  console.log('  ✓ rate limiter global cap');
}

// ------------------------------------------------------------------
// makeBodyVarier: appends invisible, changing suffix
// ------------------------------------------------------------------
{
  const vary = makeBodyVarier();
  const base = 'AAPL is up 5.2% to $200.50 today.';
  const out1 = vary(base);
  const out2 = vary(base);
  assert.notStrictEqual(out1, out2, 'consecutive varied bodies differ');
  // Visible text (zero-width stripped) is unchanged.
  assert.strictEqual(out1.replace(/\u200B/g, ''), base, 'visible text preserved');
  assert.strictEqual(out2.replace(/\u200B/g, ''), base, 'visible text preserved (2)');
  assert.strictEqual(vary(''), '', 'empty stays empty');
  console.log('  ✓ body varier invisible + changing');
}

// ------------------------------------------------------------------
// createAntiban: DEFAULT ON. Replies get LIGHT pacing (typing + small jitter,
// no rate cap); broadcasts get heavy pacing.
// ------------------------------------------------------------------
{
  // Env unset -> enabled by default.
  const ab = createAntiban({ env: {} });
  assert.strictEqual(ab.enabled, true, 'enabled by default when env unset');

  // Reply: shows typing + one small jitter sleep in the REPLY band (0.6-3s),
  // and never touches the rate limiter.
  const presence = [];
  const sock = { sendPresenceUpdate: async (s) => { presence.push(s); } };
  const slept = [];
  await ab.beforeSend({
    chatId: 'x@s.whatsapp.net', broadcast: false, sock,
    sleepFn: async (ms) => { slept.push(ms); }, rng: () => 0.5,
  });
  assert.deepStrictEqual(presence, ['composing', 'paused'], 'reply shows typing');
  assert.strictEqual(slept.length, 1, 'reply has exactly one (typing) delay');
  assert.ok(slept[0] >= 100 && slept[0] <= 500, `reply jitter ${slept[0]} in reply band 100-500`);
  console.log('  ✓ default on, replies get brief typing + tiny jitter');
}

// -- explicit hard-disable via env ----------------------------------
{
  const ab = createAntiban({ env: { WHATSAPP_ANTIBAN: '0' } });
  assert.strictEqual(ab.enabled, false, 'WHATSAPP_ANTIBAN=0 disables');
  assert.strictEqual(ab.varyBody('hello'), 'hello', 'varyBody identity when disabled');
  let sockCalled = false;
  const sock = { sendPresenceUpdate: async () => { sockCalled = true; } };
  const slept = [];
  await ab.beforeSend({ chatId: 'x@s.whatsapp.net', broadcast: false, sock, sleepFn: async (ms) => { slept.push(ms); } });
  await ab.beforeSend({ chatId: 'x@s.whatsapp.net', broadcast: true, sock, sleepFn: async (ms) => { slept.push(ms); } });
  assert.strictEqual(sockCalled, false, 'no presence when hard-disabled (reply or broadcast)');
  assert.strictEqual(slept.length, 0, 'no delay at all when hard-disabled');
  console.log('  ✓ WHATSAPP_ANTIBAN=0 hard-disables both paths');
}

// ------------------------------------------------------------------
// createAntiban ENABLED: reply pacing is lighter than broadcast pacing
// ------------------------------------------------------------------
{
  const ab = createAntiban({
    env: {
      WHATSAPP_ANTIBAN: '1',
      WHATSAPP_ANTIBAN_REPLY_JITTER_MIN_MS: '600',
      WHATSAPP_ANTIBAN_REPLY_JITTER_MAX_MS: '3000',
      WHATSAPP_ANTIBAN_JITTER_MIN_MS: '3000',
      WHATSAPP_ANTIBAN_JITTER_MAX_MS: '15000',
      WHATSAPP_ANTIBAN_TYPING: 'on',
      WHATSAPP_ANTIBAN_MAX_PER_RECIPIENT_HR: '240',
      WHATSAPP_ANTIBAN_MAX_PER_HOUR: '60',
    },
  });
  assert.strictEqual(ab.enabled, true, 'enabled via env');

  const slept = [];
  const sleepFn = async (ms) => { slept.push(ms); };
  const presence = [];
  const sock = { sendPresenceUpdate: async (state) => { presence.push(state); } };

  // Reply: typing + ONE small jitter in the reply band. No rate-cap sleep.
  slept.length = 0; presence.length = 0;
  await ab.beforeSend({ chatId: 'r@s.whatsapp.net', broadcast: false, textLength: 40, sock, sleepFn, rng: () => 0.5 });
  assert.deepStrictEqual(presence, ['composing', 'paused'], 'reply shows typing');
  assert.strictEqual(slept.length, 1, 'reply: one delay (typing jitter only, no cap)');
  assert.ok(slept[0] >= 600 && slept[0] <= 3000, `reply jitter ${slept[0]} in reply band`);

  // Broadcast: typing dwell + gaussian jitter (2 sleeps), jitter in broadcast band.
  slept.length = 0; presence.length = 0;
  await ab.beforeSend({
    chatId: 'r@s.whatsapp.net', broadcast: true, textLength: 40, sock, sleepFn,
    rng: () => 0.5,
  });
  assert.deepStrictEqual(presence, ['composing', 'paused'], 'broadcast shows typing');
  assert.strictEqual(slept.length, 2, 'broadcast sleeps twice (typing + jitter)');
  const jitter = slept[1];
  assert.ok(jitter >= 3000 && jitter <= 15000, `jitter ${jitter} within configured band`);
  console.log('  ✓ enabled: reply light-paced, broadcast heavy-paced');
}

// -- enabled broadcast honours the rate cap (waits, does not drop) ---
{
  const ab = createAntiban({
    env: {
      WHATSAPP_ANTIBAN: '1',
      WHATSAPP_ANTIBAN_TYPING: 'off',
      WHATSAPP_ANTIBAN_JITTER_MIN_MS: '0',
      WHATSAPP_ANTIBAN_JITTER_MAX_MS: '0',
      WHATSAPP_ANTIBAN_MAX_PER_RECIPIENT_HR: '1',
      WHATSAPP_ANTIBAN_MAX_PER_HOUR: '0',
    },
  });
  const slept = [];
  const sleepFn = async (ms) => { slept.push(ms); };
  const chat = 'c@s.whatsapp.net';
  const base = 2_000_000;

  // 1st broadcast: allowed, records.
  await ab.beforeSend({ chatId: chat, broadcast: true, sleepFn, now: base });
  // 2nd broadcast within the hour: must wait out the window (a positive sleep).
  slept.length = 0;
  await ab.beforeSend({ chatId: chat, broadcast: true, sleepFn, now: base + 1000 });
  assert.ok(slept.some(ms => ms > 0), 'over-cap broadcast waits for the window');
  console.log('  ✓ enabled: rate cap paces (waits, no drop)');
}

// ------------------------------------------------------------------
// Reply sequence: read receipt → typing → send (mirrors bridge.js /send).
// The read receipt itself lives in bridge.js (needs the Baileys message key),
// so this asserts the ORDER of the human sequence against a mock socket the
// same way the handler drives it.
// ------------------------------------------------------------------
{
  const ab = createAntiban({
    env: {
      WHATSAPP_ANTIBAN: '1',
      WHATSAPP_ANTIBAN_REPLY_JITTER_MIN_MS: '100',
      WHATSAPP_ANTIBAN_REPLY_JITTER_MAX_MS: '100',
    },
  });
  const events = [];
  const quotedKey = { id: 'INBOUND1', remoteJid: 'u@s.whatsapp.net', fromMe: false };
  const sock = {
    readMessages: async (keys) => { events.push(`read:${keys[0].id}`); },
    sendPresenceUpdate: async (state) => { events.push(`presence:${state}`); },
  };

  // Replicate the /send reply path: read the quoted msg, then beforeSend, then send.
  await sock.readMessages([quotedKey]);
  await ab.beforeSend({ chatId: 'u@s.whatsapp.net', broadcast: false, textLength: 20, sock });
  events.push('send');

  assert.deepStrictEqual(
    events,
    ['read:INBOUND1', 'presence:composing', 'presence:paused', 'send'],
    'human sequence: read → typing → send, in order',
  );
  console.log('  ✓ reply sequence: read receipt → typing → send');
}

// ------------------------------------------------------------------
// Regression: an over-cap broadcast that WAITS out the window records the
// send at the EFFECTIVE (post-wait) time, not the stale `now` captured at the
// start of beforeSend(). If it recorded at the old `now`, the sliding window
// would expire the entry early and the very next broadcast (attempted at the
// real moment the prior send left the wire) would be under-throttled.
// ------------------------------------------------------------------
{
  const ab = createAntiban({
    env: {
      WHATSAPP_ANTIBAN: '1',
      WHATSAPP_ANTIBAN_TYPING: 'off',
      WHATSAPP_ANTIBAN_JITTER_MIN_MS: '0',
      WHATSAPP_ANTIBAN_JITTER_MAX_MS: '0',
      WHATSAPP_ANTIBAN_MAX_PER_RECIPIENT_HR: '1',
      WHATSAPP_ANTIBAN_MAX_PER_HOUR: '0',
    },
  });
  const HOUR = 60 * 60 * 1000;
  const chat = 'c@s.whatsapp.net';
  const base = 5_000_000;
  const slept = [];
  const sleepFn = async (ms) => { slept.push(ms); };

  // Send 1 at t=base: allowed, no wait, records at base.
  await ab.beforeSend({ chatId: chat, broadcast: true, sleepFn, now: base });

  // Send 2 at t=base+1000: over cap, must wait out the window (~HOUR-1000).
  slept.length = 0;
  await ab.beforeSend({ chatId: chat, broadcast: true, sleepFn, now: base + 1000 });
  const wait2 = slept.reduce((a, b) => a + b, 0);
  assert.ok(wait2 > 0, 'send 2 waits out the window');

  // The effective moment send 2 left the wire = base+1000 + wait2. Send 3 is
  // attempted at that instant. Because send 2 was recorded at its effective
  // time (~base+HOUR), the window has NOT freed up, so send 3 must wait a full
  // window again. Under the bug (record at stale now=base+1000) send 2's entry
  // ages out almost immediately and send 3 would wait only ~1000ms.
  const send2Effective = base + 1000 + wait2;
  slept.length = 0;
  await ab.beforeSend({ chatId: chat, broadcast: true, sleepFn, now: send2Effective });
  const wait3 = slept.reduce((a, b) => a + b, 0);
  assert.ok(
    wait3 > HOUR / 2,
    `send 3 must wait a fresh window (got ${wait3}ms; the bug yields ~1000ms)`,
  );
  console.log('  ✓ over-cap broadcast records at the effective post-wait send time');
}

// ------------------------------------------------------------------
// Regression: when the typing dwell already ran and only the trailing
// 'paused' presence update fails, showTyping must NOT sleep the dwell a
// second time (that would double the delay, up to +typingMax per send).
// ------------------------------------------------------------------
{
  const ab = createAntiban({
    env: {
      WHATSAPP_ANTIBAN: '1',
      WHATSAPP_ANTIBAN_TYPING: '1',
      WHATSAPP_ANTIBAN_TYPING_MS_PER_CHAR: '100',
      WHATSAPP_ANTIBAN_TYPING_MAX_MS: '6000',
      WHATSAPP_ANTIBAN_JITTER_MIN_MS: '0',
      WHATSAPP_ANTIBAN_JITTER_MAX_MS: '0',
      WHATSAPP_ANTIBAN_MAX_PER_RECIPIENT_HR: '0',
      WHATSAPP_ANTIBAN_MAX_PER_HOUR: '0',
    },
  });
  const slept = [];
  const sleepFn = async (ms) => { slept.push(ms); };
  const sock = {
    sendPresenceUpdate: async (state) => {
      if (state === 'paused') throw new Error('presence paused failed');
    },
  };
  // textLength 20 → dwell = 20 * 100 = 2000ms, once.
  await ab.beforeSend({ chatId: 'x@s.whatsapp.net', broadcast: true, textLength: 20, sock, sleepFn });
  const dwellCount = slept.filter(ms => ms === 2000).length;
  assert.strictEqual(dwellCount, 1, `dwell must run exactly once even when 'paused' fails (got ${dwellCount} x 2000ms in ${JSON.stringify(slept)})`);
  console.log("  ✓ typing dwell runs once when only 'paused' presence update fails");
}

console.log('\n✅ All antiban tests passed.');
