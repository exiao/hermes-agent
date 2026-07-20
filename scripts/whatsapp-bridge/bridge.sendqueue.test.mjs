/**
 * Regression tests for the WhatsApp bridge send queue (#33360).
 *
 * The bridge must serialise all sock.sendMessage() calls through a
 * promise-based queue so that concurrent HTTP /send requests never
 * produce overlapping Baileys socket writes.  Overlapping writes are
 * the confirmed root cause of cross-chat contamination.
 *
 * These tests exercise the queue itself — they do NOT require a live
 * WhatsApp socket.
 */

import { strict as assert } from 'node:assert';
import {
  createGenerationTracker,
  createInFlightLookup,
  raceWithTimeout,
} from './bridge_helpers.js';

// ------------------------------------------------------------------
// 1.  Unit test for the queue primitives
// ------------------------------------------------------------------

// -- group metadata invalidation / coalescing -----------------------------
{
  const generations = createGenerationTracker();
  const beforeInvalidation = generations.token('group@g.us');
  generations.invalidate('group@g.us');
  assert.equal(
    generations.isCurrent('group@g.us', beforeInvalidation),
    false,
    'an in-flight snapshot cannot be stored after its group is invalidated',
  );

  const beforeReconnect = generations.token('group@g.us');
  generations.clear();
  assert.equal(
    generations.isCurrent('group@g.us', beforeReconnect),
    false,
    'an old-socket snapshot cannot be stored after reconnect clears the cache',
  );
  console.log('  ✓ metadata generations reject invalidated snapshots');
}

{
  const inFlight = createInFlightLookup();
  let resolveMetadata;
  let requests = 0;
  const create = () => {
    requests += 1;
    return new Promise(resolve => { resolveMetadata = resolve; });
  };
  const first = inFlight.getOrCreate('group@g.us', create);
  const second = inFlight.getOrCreate('group@g.us', create);
  assert.strictEqual(first, second, 'concurrent callers share one metadata request');
  assert.equal(requests, 1, 'only one metadata request is started');
  resolveMetadata({ id: 'group@g.us' });
  await first;
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(inFlight.get('group@g.us'), undefined, 'settled metadata requests are released');
  console.log('  ✓ metadata lookup coalesces in-flight requests');
}

{
  const generations = createGenerationTracker();
  const inFlight = createInFlightLookup();
  let releaseStalledLookup;
  const staleToken = generations.token('group@g.us');
  const stalledLookup = inFlight.getOrCreate(
    'group@g.us',
    () => new Promise(resolve => { releaseStalledLookup = resolve; }),
  );

  const result = await raceWithTimeout(stalledLookup, 5, () => {
    generations.invalidate('group@g.us');
    inFlight.clear('group@g.us');
  });
  assert.equal(result, undefined, 'timed-out metadata returns a cache miss');
  assert.equal(inFlight.get('group@g.us'), undefined, 'timed-out lookup is released for retry');
  assert.equal(
    generations.isCurrent('group@g.us', staleToken),
    false,
    'a late timed-out lookup can no longer populate the cache',
  );

  const replacement = inFlight.getOrCreate('group@g.us', async () => ({ id: 'fresh@g.us' }));
  assert.notStrictEqual(replacement, stalledLookup, 'the next warm fetch starts a replacement lookup');
  releaseStalledLookup({ id: 'stale@g.us' });
  await Promise.all([stalledLookup, replacement]);
  console.log('  ✓ timed-out metadata lookup is invalidated and retried');
}

/**
 * Replicate the queue logic from bridge.js so we can test it in
 * isolation without importing the full module (which would trigger
 * Baileys / express side effects).
 */
function createSendQueue() {
  let _sendQueue = Promise.resolve();

  function enqueueSend(fn) {
    const task = _sendQueue.then(() => fn(), () => fn());
    _sendQueue = task.catch(() => {});
    return task;
  }

  return { enqueueSend };
}

function createSendWithTimeout(enqueueSend, send, timeoutMs) {
  return () => {
    return enqueueSend(() => {
      let timer;
      const timeout = new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error('timeout')), timeoutMs);
      });
      return Promise.race([send(), timeout]).finally(() => clearTimeout(timer));
    });
  };
}

// -- serial ordering -------------------------------------------------
{
  const { enqueueSend } = createSendQueue();
  const order = [];

  const a = enqueueSend(async () => {
    await new Promise(r => setTimeout(r, 30));
    order.push('a');
    return 'A';
  });
  const b = enqueueSend(async () => {
    order.push('b');
    return 'B';
  });
  const c = enqueueSend(async () => {
    await new Promise(r => setTimeout(r, 10));
    order.push('c');
    return 'C';
  });

  const results = await Promise.all([a, b, c]);
  assert.deepStrictEqual(results, ['A', 'B', 'C'], 'all tasks resolve');
  assert.deepStrictEqual(order, ['a', 'b', 'c'], 'tasks execute in FIFO order');
  console.log('  ✓ serial ordering');
}

// -- error isolation (one rejection does not stall the queue) --------
{
  const { enqueueSend } = createSendQueue();
  const order = [];

  const bad = enqueueSend(async () => {
    order.push('bad');
    throw new Error('boom');
  });
  const good = enqueueSend(async () => {
    order.push('good');
    return 'ok';
  });

  await assert.rejects(() => bad, /boom/, 'bad task rejects');
  const g = await good;
  assert.strictEqual(g, 'ok', 'good task still resolves');
  assert.deepStrictEqual(order, ['bad', 'good'], 'good runs after bad');
  console.log('  ✓ error isolation');
}

// -- timeout still fires (wrapped inside enqueueSend) ----------------
{
  const { enqueueSend } = createSendQueue();
  const timedOut = enqueueSend(async () => {
    await new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), 20));
  });
  await assert.rejects(() => timedOut, /timeout/, 'inner timeout propagates');
  console.log('  ✓ timeout propagation');
}

// -- queued time does not consume the send timeout --------------------
{
  const { enqueueSend } = createSendQueue();
  const releaseFirst = enqueueSend(() => new Promise(resolve => setTimeout(resolve, 30)));
  const send = createSendWithTimeout(enqueueSend, async () => 'sent', 10);

  const result = await send();
  await releaseFirst;
  assert.equal(result, 'sent', 'timeout starts only after the queued send begins');
  console.log('  ✓ queue delay does not consume send timeout');
}

// -- concurrent enqueues maintain single-consumer semantics ----------
{
  const { enqueueSend } = createSendQueue();
  let concurrent = 0;
  let maxConcurrent = 0;

  async function tracked() {
    concurrent += 1;
    if (concurrent > maxConcurrent) maxConcurrent = concurrent;
    await new Promise(r => setTimeout(r, 5));
    concurrent -= 1;
  }

  await Promise.all(Array.from({ length: 20 }, () => enqueueSend(tracked)));
  assert.strictEqual(maxConcurrent, 1, 'never more than one in-flight');
  assert.strictEqual(concurrent, 0, 'all finished');
  console.log('  ✓ single-consumer concurrency');
}

console.log('\n✅ All send-queue tests passed.');
