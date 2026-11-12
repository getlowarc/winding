// The KTX2 container. Run: node test/ktx2.test.js
//
// Files are assembled here, byte by byte, the way the spec lays them out --
// identifier, header, index, level index, Data Format Descriptor, then the
// levels smallest first -- so every field the reader takes is one this test
// put there on purpose.

import assert from 'node:assert/strict';
import zlib from 'node:zlib';
import { readKTX2, KTX2_NONE, KTX2_BASISLZ, KTX2_ZSTD } from '../src/scene/ktx2.js';

let passed = 0;
function test(name, fn) { fn(); passed++; console.log(`  ok  ${name}`); }

const IDENTIFIER = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A];

/** A KTX2 file: levels largest first, as raw bytes; packed with zstd when `scheme` is 2. */
function ktx2({ vkFormat = 0, width, height, levels, scheme = KTX2_NONE, model = 166, transfer = 2, channel = 3, flags = 0, globalData = null }) {
  const stored = levels.map((data) => (scheme === KTX2_ZSTD ? new Uint8Array(zlib.zstdCompressSync(data)) : data));
  const dfd = new Uint8Array(4 + 24 + 16);
  const dv = new DataView(dfd.buffer);
  dv.setUint32(0, dfd.length, true);
  dv.setUint16(4 + 6, 24 + 16, true);   // descriptorBlockSize
  dfd[4 + 8] = model; dfd[4 + 9] = 1; dfd[4 + 10] = transfer; dfd[4 + 11] = flags;
  dfd[4 + 24 + 3] = channel;
  const levelIndex = 80, dfdAt = levelIndex + levels.length * 24;
  let at = dfdAt + dfd.length;
  const sgdAt = at;
  if (globalData) at += globalData.length;
  // Levels go smallest first in the file.
  const offsets = new Array(levels.length);
  for (let i = levels.length - 1; i >= 0; i--) { at = Math.ceil(at / 8) * 8; offsets[i] = at; at += stored[i].length; }
  const file = new Uint8Array(at);
  const v = new DataView(file.buffer);
  file.set(IDENTIFIER, 0);
  [vkFormat, 1, width, height, 0, 0, 1, levels.length, scheme].forEach((x, i) => v.setUint32(12 + i * 4, x, true));
  v.setUint32(48, dfdAt, true); v.setUint32(52, dfd.length, true);
  if (globalData) { v.setUint32(64, sgdAt, true); v.setUint32(72, globalData.length, true); file.set(globalData, sgdAt); }
  levels.forEach((data, i) => {
    v.setUint32(levelIndex + i * 24, offsets[i], true);
    v.setUint32(levelIndex + i * 24 + 8, stored[i].length, true);
    v.setUint32(levelIndex + i * 24 + 16, data.length, true);
    file.set(stored[i], offsets[i]);
  });
  file.set(dfd, dfdAt);
  return file;
}

let seed = 5;
const bytes = (n) => Uint8Array.from({ length: n }, () => Math.floor((seed = (seed * 16807) % 2147483647) / 2147483647 * 256));
/** UASTC levels for a w x h image: 16 bytes a 4x4 block, down to 1x1. */
const uastcLevels = (w, h) => {
  const out = [];
  for (let l = 0; Math.max(w, h) >> l >= 1; l++) out.push(bytes(16 * Math.ceil(Math.max(1, w >> l) / 4) * Math.ceil(Math.max(1, h >> l) / 4)));
  return out;
};

console.log('\nktx2');

test('UASTC under zstd: the levels unpacked, largest first, with what the descriptor says', () => {
  const levels = uastcLevels(64, 32);
  const t = readKTX2(ktx2({ width: 64, height: 32, levels, scheme: KTX2_ZSTD }));
  assert.equal(t.basis, 'uastc');
  assert.equal(t.srgb, true);
  assert.equal(t.channels, 'rgba');
  assert.equal(t.supercompression, KTX2_ZSTD);
  assert.equal(t.levels.length, levels.length);
  t.levels.forEach((level, i) => {
    assert.deepEqual([level.width, level.height], [Math.max(1, 64 >> i), Math.max(1, 32 >> i)], `level ${i} size`);
    assert.deepEqual(level.data, levels[i], `level ${i} bytes`);
  });
});

test('linear, premultiplied and each UASTC channel layout are read from the descriptor', () => {
  const levels = uastcLevels(4, 4);
  const t = readKTX2(ktx2({ width: 4, height: 4, levels, transfer: 1, flags: 1, channel: 0 }));
  assert.equal(t.srgb, false);
  assert.equal(t.premultiplied, true);
  assert.equal(t.channels, 'rgb');
  for (const [channel, name] of [[4, 'rrr'], [5, 'rrrg'], [6, 'rg']]) {
    assert.equal(readKTX2(ktx2({ width: 4, height: 4, levels, channel })).channels, name);
  }
});

test('a GPU format stored as it is (BC7) comes through untouched', () => {
  const levels = [bytes(16 * 4), bytes(16)];
  const t = readKTX2(ktx2({ vkFormat: 145, width: 8, height: 8, levels, model: 128 }));
  assert.equal(t.vkFormat, 145);
  assert.equal(t.basis, null);
  assert.deepEqual(t.levels.map((l) => l.data), levels);
});

test('ETC1S keeps its BasisLZ levels and global data for its own decoder', () => {
  const levels = [bytes(40), bytes(12)], global = bytes(36);
  const t = readKTX2(ktx2({ width: 8, height: 8, levels, scheme: KTX2_BASISLZ, model: 163, globalData: global }));
  assert.equal(t.basis, 'etc1s');
  assert.deepEqual(t.globalData, global);
  assert.deepEqual(t.levels.map((l) => l.data), levels);
});

test('what is not a KTX2 file, or not one this reads, says so', () => {
  assert.throws(() => readKTX2(new Uint8Array(100)), /ktx2: not a KTX2 file/);
  const good = ktx2({ width: 4, height: 4, levels: uastcLevels(4, 4) });
  assert.throws(() => readKTX2(good.subarray(0, good.length - 4)), /past the end of the file/);
  assert.throws(() => readKTX2(ktx2({ width: 4, height: 4, levels: uastcLevels(4, 4), model: 0 })), /neither Basis UASTC nor ETC1S/);
  const zlibbed = ktx2({ width: 4, height: 4, levels: uastcLevels(4, 4) });
  new DataView(zlibbed.buffer).setUint32(44, 3, true);
  assert.throws(() => readKTX2(zlibbed), /ZLIB supercompression/);
  assert.throws(() => readKTX2(ktx2({ width: 4, height: 4, levels: [bytes(8)], model: 163 })), /ETC1S without BasisLZ/);
});

console.log(`\n${passed} checks passed\n`);
