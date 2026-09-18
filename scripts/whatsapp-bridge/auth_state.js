import Database from 'better-sqlite3';
import { BufferJSON, initAuthCreds, proto } from '@whiskeysockets/baileys';
import { mkdirSync, chmodSync, existsSync, readFileSync, readdirSync, writeFileSync, renameSync, unlinkSync } from 'node:fs';
import { join } from 'node:path';
import { randomBytes } from 'node:crypto';

const encode = value => JSON.stringify(value, BufferJSON.replacer);
const decode = value => JSON.parse(value, BufferJSON.reviver);
const filename = value => value.replace(/\//g, '__').replace(/:/g, '-');
const mirrored = name => name === 'creds.json' || /^lid-mapping-.*\.json$/.test(name);

function validateCreds(creds) {
  if (!creds || !Number.isInteger(creds.registrationId)
      || !creds.noiseKey?.private || !creds.noiseKey?.public
      || !creds.signedIdentityKey?.private || !creds.signedIdentityKey?.public
      || !creds.signedPreKey?.keyPair?.private || !creds.signedPreKey?.keyPair?.public) {
    throw new Error('Invalid WhatsApp credentials; restore the session backup, do not reinitialize');
  }
}

// One bridge owns this directory. SQLite commits key batches atomically; JSON
// files are compatibility views for Python pairing/identity readers, not auth.
export function useSqliteAuthState(sessionDir) {
  mkdirSync(sessionDir, { recursive: true, mode: 0o700 });
  chmodSync(sessionDir, 0o700);
  const dbPath = join(sessionDir, 'auth.sqlite');
  const marker = join(sessionDir, 'sqlite-migrated');
  if (existsSync(marker) && !existsSync(dbPath)) {
    throw new Error('WhatsApp auth database missing after migration; restore backup');
  }
  const db = new Database(dbPath);
  try {
    chmodSync(dbPath, 0o600);
    db.pragma('journal_mode = WAL');
    db.pragma('synchronous = FULL');
    db.exec('CREATE TABLE IF NOT EXISTS auth (name TEXT PRIMARY KEY, value TEXT NOT NULL); CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)');
    const read = db.prepare('SELECT value FROM auth WHERE name = ?');
    const put = db.prepare('INSERT OR REPLACE INTO auth (name, value) VALUES (?, ?)');
    const remove = db.prepare('DELETE FROM auth WHERE name = ?');
    const initialized = db.prepare("SELECT value FROM metadata WHERE name = 'initialized'").get();
    if (!initialized) {
      if (existsSync(marker)) throw new Error('WhatsApp auth database incomplete after migration');
      const files = readdirSync(sessionDir).filter(name => name.endsWith('.json'));
      if (files.length && !files.includes('creds.json')) {
        throw new Error('Legacy WhatsApp keys exist without credentials; restore backup');
      }
      const entries = files.map(name => [name, encode(decode(readFileSync(join(sessionDir, name), 'utf8')))]);
      const creds = entries.find(([name]) => name === 'creds.json');
      const initialCreds = creds ? decode(creds[1]) : initAuthCreds();
      validateCreds(initialCreds);
      db.transaction(() => {
        for (const [name, value] of entries) put.run(name, value);
        if (!creds) put.run('creds.json', encode(initialCreds));
        db.prepare("INSERT INTO metadata VALUES ('initialized', '1')").run();
      })();
    }
    const row = read.get('creds.json');
    const creds = row && decode(row.value);
    validateCreds(creds);
    writeFileSync(marker, 'SQLite is authoritative. Do not restore stale JSON keys.\n', { mode: 0o600 });

    function exportFile(name, value) {
      const target = join(sessionDir, name);
      if (value === undefined) {
        if (existsSync(target)) unlinkSync(target);
        return;
      }
      const temp = `${target}.${randomBytes(6).toString('hex')}.tmp`;
      try {
        writeFileSync(temp, value, { mode: 0o600, flag: 'wx' });
        renameSync(temp, target);
      } finally {
        if (existsSync(temp)) unlinkSync(temp);
      }
    }
    // Repair views after a crash between a DB commit and a compatibility export.
    const mirrors = new Map(db.prepare("SELECT name, value FROM auth WHERE name = 'creds.json' OR name LIKE 'lid-mapping-%.json'").all().map(r => [r.name, r.value]));
    for (const name of readdirSync(sessionDir).filter(mirrored)) {
      if (!mirrors.has(name)) exportFile(name, undefined);
    }
    for (const [name, value] of mirrors) exportFile(name, value);

    const getValue = name => {
      const row = read.get(name);
      return row ? decode(row.value) : null;
    };
    return {
      state: {
        creds,
        keys: {
          async get(type, ids) {
            return Object.fromEntries(ids.map(id => {
              let value = getValue(filename(`${type}-${id}.json`));
              if (type === 'app-state-sync-key' && value) value = proto.Message.AppStateSyncKeyData.fromObject(value);
              return [id, value];
            }));
          },
          async set(data) {
            // Serialize before committing so malformed values cannot partially write.
            const entries = Object.entries(data).flatMap(([type, values]) => Object.entries(values).map(([id, value]) => [filename(`${type}-${id}.json`), value == null ? null : encode(value)]));
            db.transaction(() => {
              for (const [name, value] of entries) {
                if (value === null) remove.run(name);
                else put.run(name, value);
              }
            })();
            for (const [name, value] of entries) if (mirrored(name)) exportFile(name, value ?? undefined);
          },
        },
      },
      saveCreds() {
        validateCreds(creds);
        const value = encode(creds);
        put.run('creds.json', value);
        exportFile('creds.json', value);
      },
      getLidMapping: (identifier, suffix = '') => getValue(filename(`lid-mapping-${identifier}${suffix}.json`)),
      close: () => db.close(),
    };
  } catch (error) {
    db.close();
    throw error;
  }
}
