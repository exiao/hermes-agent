// Exercises the real bridge close handler and HTTP health with a fake transport.
import { strict as assert } from 'node:assert';
import { mock, test } from 'node:test';
import { EventEmitter } from 'node:events';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import express from 'express';
import { get } from 'node:http';
import * as baileys from '@whiskeysockets/baileys';

const tick = () => new Promise(resolve => setImmediate(resolve));

test('bridge honors backoff, reconnects, and stops on forbidden credentials', async () => {
  const session = mkdtempSync(join(tmpdir(), 'bridge-connection-'));
  const sockets = [];
  let server;
  mock.module('express', { defaultExport: Object.assign(() => {
    const app = express();
    const listen = app.listen.bind(app);
    app.listen = (...args) => (server = listen(...args));
    return app;
  }, express) });
  const { default: defaultExport, ...namedExports } = baileys;
  mock.module('@whiskeysockets/baileys', { defaultExport, namedExports: {
    ...namedExports,
    useMultiFileAuthState: async () => ({ state: {}, saveCreds: async () => {} }),
    fetchLatestBaileysVersion: async () => ({ version: [2, 3000, 0] }),
    makeWASocket: () => {
      const socket = { ev: new EventEmitter() };
      sockets.push(socket);
      return socket;
    },
  } });
  const argv = process.argv;
  process.argv = [...argv.slice(0, 2), '--port', '0', '--session', session, '--mode', 'bot'];
  try {
    mock.timers.enable({ apis: ['setTimeout'] });
    await import('./bridge.js');
    for (let n = 0; n < 100 && sockets.length === 0; n++) { mock.timers.tick(0); await tick(); }
    assert.equal(sockets.length, 1);
    mock.method(Math, 'random', () => 1 - Number.EPSILON);
    const close = code => sockets.at(-1).ev.emit('connection.update', {
      connection: 'close', lastDisconnect: { error: Object.assign(new Error('test disconnect'), { isBoom: true, output: { statusCode: code } }) },
    });
    close(428);
    mock.timers.tick(3000);
    for (let n = 0; n < 100 && sockets.length < 2; n++) await tick();
    assert.equal(sockets.length, 2);
    close(428);
    mock.timers.tick(3000);
    await tick();
    assert.equal(sockets.length, 2, 'second failure must not retry after only three seconds');
    mock.timers.tick(3000);
    for (let n = 0; n < 100 && sockets.length < 3; n++) await tick();
    assert.equal(sockets.length, 3);
    sockets.at(-1).ev.emit('connection.update', { connection: 'open' });
    const health = () => new Promise((resolve, reject) => {
      get(`http://127.0.0.1:${server.address().port}/health`, { agent: false }, response => {
        let body = '';
        response.on('data', chunk => { body += chunk; });
        response.on('end', () => resolve(JSON.parse(body)));
      }).on('error', reject);
    });
    assert.equal((await health()).status, 'connected');
    close(403);
    mock.timers.tick(24 * 60 * 60 * 1000);
    await tick();
    assert.equal(sockets.length, 3, 'forbidden session must not reconnect');
    assert.equal((await health()).status, 'disconnected');
  } finally {
    mock.restoreAll();
    mock.timers.reset();
    process.argv = argv;
    if (server) {
      server.closeAllConnections();
      await new Promise(resolve => server.close(resolve));
    }
    rmSync(session, { recursive: true, force: true });
  }
});
