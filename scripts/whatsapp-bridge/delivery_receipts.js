const STATUS_BY_CODE = {
  2: { status: 'server_ack', delivered: false },
  3: { status: 'delivered', delivered: true },
  4: { status: 'read', delivered: true },
  5: { status: 'played', delivered: true },
};

const STATUS_RANK = { sent: 0, server_ack: 1, delivered: 2, read: 3, played: 4 };

export function createDeliveryReceiptTracker({
  maxSize = 1000,
  ttlMs = 24 * 60 * 60 * 1000,
  now = Date.now,
} = {}) {
  if (!Number.isInteger(maxSize) || maxSize < 1) throw new RangeError('maxSize must be positive');
  const entries = new Map();

  function prune() {
    const timestamp = now();
    for (const [id, entry] of entries) {
      if (entry.expiresAt <= timestamp) entries.delete(id);
    }
  }

  function cap() {
    while (entries.size > maxSize) entries.delete(entries.keys().next().value);
  }

  function setStatus(id, value) {
    if (!id || !value || !STATUS_RANK[value.status]) return;
    prune();
    const entry = entries.get(id) || { known: false, status: 'sent', delivered: false, expiresAt: now() + ttlMs };
    if (STATUS_RANK[value.status] < STATUS_RANK[entry.status]) return;
    entries.delete(id);
    entries.set(id, { ...entry, ...value });
    cap();
  }

  function register(sent) {
    const id = typeof sent === 'string' ? sent : sent?.key?.id;
    if (!id) return;
    prune();
    const pending = entries.get(id);
    entries.delete(id);
    entries.set(id, {
      known: true,
      status: pending?.status || 'sent',
      delivered: pending?.delivered || false,
      expiresAt: now() + ttlMs,
    });
    cap();
  }

  function updateStatus(id, code) {
    const value = STATUS_BY_CODE[Number(code)];
    if (value) setStatus(id, value);
  }

  function updateReceipt(id, receipt = {}) {
    const status = Number(receipt.playedTimestamp) > 0
      ? 'played'
      : Number(receipt.readTimestamp) > 0
        ? 'read'
        : Number(receipt.receiptTimestamp) > 0 ? 'delivered' : null;
    if (status) setStatus(id, { status, delivered: true });
  }

  function get(id) {
    prune();
    const entry = entries.get(id);
    return entry?.known
      ? { messageId: id, status: entry.status, delivered: entry.delivered }
      : null;
  }

  return { register, updateStatus, updateReceipt, get, size: () => (prune(), entries.size) };
}
