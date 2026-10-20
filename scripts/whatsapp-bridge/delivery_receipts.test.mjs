import test from 'node:test';
import assert from 'node:assert/strict';

import { createDeliveryReceiptTracker } from './delivery_receipts.js';

test('keeps server ack undelivered and applies fast receipts after registration', () => {
  const tracker = createDeliveryReceiptTracker();
  tracker.updateStatus('m-ack', 2);
  assert.equal(tracker.get('m-ack'), null);
  tracker.register({ key: { id: 'm-ack' } });
  assert.deepEqual(tracker.get('m-ack'), { messageId: 'm-ack', status: 'server_ack', delivered: false });

  tracker.updateStatus('m-ack', 3);
  tracker.updateStatus('m-ack', 4);
  assert.deepEqual(tracker.get('m-ack'), { messageId: 'm-ack', status: 'read', delivered: true });

  tracker.updateReceipt('m-zero', { receiptTimestamp: 0 });
  assert.equal(tracker.get('m-zero'), null);
  tracker.updateReceipt('m-fast', { receiptTimestamp: 10, readTimestamp: 11 });
  assert.equal(tracker.get('m-fast'), null);
  tracker.register({ key: { id: 'm-fast' } });
  assert.deepEqual(tracker.get('m-fast'), { messageId: 'm-fast', status: 'read', delivered: true });
});

test('hides unknown and expired ids and bounds pending receipts', () => {
  let now = 0;
  const tracker = createDeliveryReceiptTracker({ maxSize: 2, ttlMs: 10, now: () => now });
  tracker.updateReceipt('unknown', { receiptTimestamp: 1 });
  assert.equal(tracker.get('unknown'), null);
  tracker.register('one');
  tracker.register('two');
  tracker.register('three');
  assert.equal(tracker.get('one'), null);
  assert.equal(tracker.get('three')?.status, 'sent');
  now = 11;
  assert.equal(tracker.get('two'), null);
});
