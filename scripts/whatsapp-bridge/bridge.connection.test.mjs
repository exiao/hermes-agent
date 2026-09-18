// Exercises the real bridge close handler and HTTP health with a fake transport.
import { strict as assert } from 'node:assert';
import { mock, test } from 'node:test';
import { EventEmitter } from 'node:events';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import express from 'express';
import { get, request } from 'node:http';
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
      const socket = {
        ev: new EventEmitter(),
        sendMessage: async () => ({ key: { id: 'receipt-test-id', fromMe: true, remoteJid: '15551234567@s.whatsapp.net' } }),
      };
      sockets.push(socket);
      return socket;
    },
  } });
  const argv = process.argv;
  const antiban = process.env.WHATSAPP_ANTIBAN;
  process.env.WHATSAPP_ANTIBAN = '0';
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
    const jsonRequest = (method, path, body) => new Promise((resolve, reject) => {
      const req = request({
        hostname: '127.0.0.1',
        port: server.address().port,
        path,
        method,
        headers: body ? { 'content-type': 'application/json' } : {},
      }, response => {
        let bodyText = '';
        response.on('data', chunk => { bodyText += chunk; });
        response.on('end', () => resolve({ status: response.statusCode, body: JSON.parse(bodyText) }));
      });
      req.on('error', reject);
      req.end(body ? JSON.stringify(body) : undefined);
    });
    const sent = await jsonRequest('POST', '/send', { chatId: '15551234567@s.whatsapp.net', message: 'receipt test' });
    assert.equal(sent.status, 200);
    assert.equal(sent.body.messageId, 'receipt-test-id');
    const socket = sockets.at(-1);
    socket.ev.emit('messages.update', [{ key: { id: 'receipt-test-id', fromMe: true }, update: { status: 2 } }]);
    await tick();
    assert.deepEqual((await jsonRequest('GET', '/message-status/receipt-test-id')).body, {
      messageId: 'receipt-test-id', status: 'server_ack', delivered: false,
    });
    socket.ev.emit('message-receipt.update', [{
      key: { id: 'receipt-test-id', fromMe: true }, receipt: { receiptTimestamp: 123 },
    }]);
    assert.deepEqual((await jsonRequest('GET', '/message-status/receipt-test-id')).body, {
      messageId: 'receipt-test-id', status: 'delivered', delivered: true,
    });
    socket.ev.emit('messages.update', [{ key: { id: 'inbound-id', fromMe: false }, update: { status: 4 } }]);
    socket.ev.emit('message-receipt.update', [{
      key: { id: 'inbound-id', fromMe: false }, receipt: { receiptTimestamp: 123 },
    }]);
    assert.equal((await jsonRequest('GET', '/message-status/inbound-id')).status, 404);
    close(403);
    mock.timers.tick(24 * 60 * 60 * 1000);
    await tick();
    assert.equal(sockets.length, 3, 'forbidden session must not reconnect');
    assert.equal((await health()).status, 'disconnected');
  } finally {
    mock.restoreAll();
    mock.timers.reset();
    process.argv = argv;
    if (antiban === undefined) delete process.env.WHATSAPP_ANTIBAN;
    else process.env.WHATSAPP_ANTIBAN = antiban;
    if (server) {
      server.closeAllConnections();
      await new Promise(resolve => server.close(resolve));
    }
    rmSync(session, { recursive: true, force: true });
  }
});
