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
// createAntiban: DEFAULT ON, but a pure passthrough for interactive replies
// ------------------------------------------------------------------
{
  // Env unset -> enabled by default.
  const ab = createAntiban({ env: {} });
  assert.strictEqual(ab.enabled, true, 'enabled by default when env unset');

  // Interactive (non-broadcast) send: nothing happens — no presence, no sleep.
  let sockCalled = false;
  const sock = { sendPresenceUpdate: async () => { sockCalled = true; } };
  const slept = [];
  await ab.beforeSend({
    chatId: 'x@s.whatsapp.net', broadcast: false, sock,
    sleepFn: async (ms) => { slept.push(ms); },
  });
  assert.strictEqual(sockCalled, false, 'no presence on interactive reply');
  assert.strictEqual(slept.length, 0, 'no delay added to interactive reply');
  console.log('  ✓ default on, interactive replies untouched');
}

// -- explicit hard-disable via env ----------------------------------
{
  const ab = createAntiban({ env: { WHATSAPP_ANTIBAN: '0' } });
  assert.strictEqual(ab.enabled, false, 'WHATSAPP_ANTIBAN=0 disables');
  assert.strictEqual(ab.varyBody('hello'), 'hello', 'varyBody identity when disabled');
  let sockCalled = false;
  const sock = { sendPresenceUpdate: async () => { sockCalled = true; } };
  await ab.beforeSend({ chatId: 'x@s.whatsapp.net', broadcast: true, sock });
  assert.strictEqual(sockCalled, false, 'no presence when hard-disabled even for broadcast');
  console.log('  ✓ WHATSAPP_ANTIBAN=0 hard-disables');
}

// ------------------------------------------------------------------
// createAntiban ENABLED: reply path stays instant, broadcast gets paced
// ------------------------------------------------------------------
{
  const ab = createAntiban({
    env: {
      WHATSAPP_ANTIBAN: '1',
      WHATSAPP_ANTIBAN_JITTER_MIN_MS: '3000',
      WHATSAPP_ANTIBAN_JITTER_MAX_MS: '15000',
      WHATSAPP_ANTIBAN_TYPING: 'on',
      WHATSAPP_ANTIBAN_MAX_PER_RECIPIENT_HR: '6',
      WHATSAPP_ANTIBAN_MAX_PER_HOUR: '60',
    },
  });
  assert.strictEqual(ab.enabled, true, 'enabled via env');

  const slept = [];
  const sleepFn = async (ms) => { slept.push(ms); };
  const presence = [];
  const sock = { sendPresenceUpdate: async (state) => { presence.push(state); } };

  // Interactive reply: NOTHING — no presence, no sleep at all.
  slept.length = 0; presence.length = 0;
  await ab.beforeSend({ chatId: 'r@s.whatsapp.net', broadcast: false, textLength: 40, sock, sleepFn });
  assert.deepStrictEqual(presence, [], 'reply shows no typing (broadcast-only)');
  assert.strictEqual(slept.length, 0, 'reply adds no delay');

  // Broadcast: typing dwell + gaussian jitter (2 sleeps), jitter in range.
  slept.length = 0; presence.length = 0;
  await ab.beforeSend({
    chatId: 'r@s.whatsapp.net', broadcast: true, textLength: 40, sock, sleepFn,
    rng: () => 0.5,
  });
  assert.deepStrictEqual(presence, ['composing', 'paused'], 'broadcast shows typing');
  assert.strictEqual(slept.length, 2, 'broadcast sleeps twice (typing + jitter)');
  const jitter = slept[1];
  assert.ok(jitter >= 3000 && jitter <= 15000, `jitter ${jitter} within configured band`);
  console.log('  ✓ enabled: reply instant, broadcast paced with jitter');
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

console.log('\n✅ All antiban tests passed.');
