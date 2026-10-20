import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, readFileSync, existsSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import Database from 'better-sqlite3';
import { BufferJSON, initAuthCreds } from '@whiskeysockets/baileys';
import { useSqliteAuthState } from './auth_state.js';
import { matchesAllowedUser, parseAllowedUsers } from './allowlist.js';
const encode = value => JSON.stringify(value, BufferJSON.replacer);

test('legacy auth migrates, survives reopen, commits key batches atomically and repairs compatibility views', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'wa-auth-'));
  let store;
  try {
    const creds = initAuthCreds();
    writeFileSync(join(dir, 'creds.json'), encode(creds));
    writeFileSync(join(dir, 'session-user-1.json'), encode({ key: Buffer.from('original') }));
    writeFileSync(join(dir, 'lid-mapping-15550001111.json'), encode('999'));
    writeFileSync(join(dir, 'lid-mapping-999_reverse.json'), encode('15550001111'));
    store = useSqliteAuthState(dir);
    assert.deepEqual(store.state.creds.noiseKey.private, creds.noiseKey.private);
    assert.deepEqual((await store.state.keys.get('session', ['user:1']))['user:1'].key, Buffer.from('original'));
    assert.equal(matchesAllowedUser('999@lid', parseAllowedUsers('15550001111'), dir, store.getLidMapping), true);
    store.state.creds.accountSyncCounter = 123;
    store.saveCreds();
    assert.equal(JSON.parse(readFileSync(join(dir, 'creds.json'))).accountSyncCounter, 123);
    await store.state.keys.set({ session: { 'user:1': { key: Buffer.from('updated') }, second: { ok: true } }, 'app-state-sync-key': { key: { keyData: Buffer.from('poll') } } });
    // An invalid second value must roll back the first SQL write too.
    await assert.rejects(store.state.keys.set({ session: { 'user:1': { key: Buffer.from('wrong') }, bad: () => {} } }));
    assert.deepEqual((await store.state.keys.get('session', ['user:1']))['user:1'].key, Buffer.from('updated'));
    await store.state.keys.set({ session: { second: null }, 'lid-mapping': { '999_reverse': null } });
    assert.equal(existsSync(join(dir, 'lid-mapping-999_reverse.json')), false);
    store.close(); store = undefined;
    // Simulate stale files left after a crash, including original legacy keys.
    writeFileSync(join(dir, 'lid-mapping-999_reverse.json'), encode('15550001111'));
    writeFileSync(join(dir, 'creds.json'), 'bad stale JSON');
    store = useSqliteAuthState(dir);
    assert.equal(store.state.creds.accountSyncCounter, 123);
    assert.deepEqual((await store.state.keys.get('session', ['user:1', 'second']))['user:1'].key, Buffer.from('updated'));
    assert.equal((await store.state.keys.get('session', ['second'])).second, null);
    assert.deepEqual((await store.state.keys.get('app-state-sync-key', ['key'])).key.keyData, Buffer.from('poll'));
    assert.equal(store.getLidMapping('999', '_reverse'), null);
    assert.equal(existsSync(join(dir, 'lid-mapping-999_reverse.json')), false);
    assert.equal(statSync(join(dir, 'auth.sqlite')).mode & 0o777, 0o600);
    assert.equal(statSync(join(dir, 'creds.json')).mode & 0o777, 0o600);
    assert.equal(existsSync(join(dir, 'session-user-1.json')), true, 'legacy keys retained');
  } finally { store?.close(); rmSync(dir, { recursive: true, force: true }); }
});

test('corrupt or incomplete credentials fail closed without generating a replacement identity', () => {
  const dir = mkdtempSync(join(tmpdir(), 'wa-auth-fail-'));
  try {
    writeFileSync(join(dir, 'session-existing.json'), '{}');
    assert.throws(() => useSqliteAuthState(dir), /without credentials/);
    writeFileSync(join(dir, 'creds.json'), '{bad');
    assert.throws(() => useSqliteAuthState(dir));
    assert.equal(readFileSync(join(dir, 'creds.json'), 'utf8'), '{bad');
    writeFileSync(join(dir, 'creds.json'), '{}');
    assert.throws(() => useSqliteAuthState(dir), /Invalid WhatsApp credentials/);
    const creds = initAuthCreds();
    writeFileSync(join(dir, 'creds.json'), encode(creds));
    const store = useSqliteAuthState(dir); store.close();
    const db = new Database(join(dir, 'auth.sqlite'));
    db.prepare("DELETE FROM auth WHERE name = 'creds.json'").run(); db.close();
    assert.throws(() => useSqliteAuthState(dir), /Invalid WhatsApp credentials/);
    rmSync(join(dir, 'auth.sqlite'));
    assert.throws(() => useSqliteAuthState(dir), /missing after migration/);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
