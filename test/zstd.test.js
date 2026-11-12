// Zstandard decompression. Run: node test/zstd.test.js
//
// Node compresses (zlib.zstdCompressSync, the reference library underneath),
// src/core/zstd.js decompresses, and every byte must come back. The inputs are
// chosen to reach each part of the format: incompressible bytes (raw blocks),
// one byte repeated (RLE blocks), text (Huffman literals, FSE sequences, the
// repeat offsets), texture-like structured data, many blocks, and every level
// from fastest to strongest, whose encoders choose tables differently.

import assert from 'node:assert/strict';
import zlib from 'node:zlib';
import { zstdDecompress } from '../src/core/zstd.js';

let passed = 0;
function test(name, fn) { fn(); passed++; console.log(`  ok  ${name}`); }

// Node's zstd compressor came in 22.15 and 23.8; without one there is nothing to check against.
if (typeof zlib.zstdCompressSync !== 'function') {
  console.log(`
zstd
  --  skipped: this Node (${process.version}) has no zlib.zstdCompressSync to compress with
`);
  process.exit(0);
}

let seed = 7;
const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);

const compress = (data, level = 3, checksum = false) => new Uint8Array(zlib.zstdCompressSync(data, {
  params: { [zlib.constants.ZSTD_c_compressionLevel]: level, [zlib.constants.ZSTD_c_checksumFlag]: checksum ? 1 : 0 },
}));

function roundTrip(data, level, label, checksum = false) {
  const packed = compress(data, level, checksum);
  const back = zstdDecompress(packed, data.length);
  assert.equal(back.length, data.length, `${label}: length`);
  for (let i = 0; i < data.length; i++) if (back[i] !== data[i]) assert.fail(`${label}: byte ${i} is ${back[i]}, not ${data[i]}`);
  // Without the size: from the frame's own, or grown.
  assert.deepEqual(zstdDecompress(packed), back, `${label}: with no size given`);
}

const text = (n) => {
  const words = 'the a texture block of mip level and zstd frame huffman literal sequence offset match window winding engine'.split(' ');
  let s = '';
  while (s.length < n) s += words[Math.floor(random() * words.length)] + (random() < 0.1 ? '.\n' : ' ');
  return new TextEncoder().encode(s.slice(0, n));
};
const incompressible = (n) => Uint8Array.from({ length: n }, () => Math.floor(random() * 256));
// Like a block-compressed texture: 16-byte blocks of a few distinct patterns with noise.
const textureLike = (n) => {
  const out = new Uint8Array(n);
  const patterns = Array.from({ length: 12 }, () => incompressible(16));
  for (let i = 0; i < n; i += 16) {
    out.set(patterns[Math.floor(random() * patterns.length)], i);
    if (random() < 0.3) out[i + Math.floor(random() * 16)] ^= Math.floor(random() * 256);
  }
  return out;
};

console.log('\nzstd');

test('text, at every level from fastest to strongest', () => {
  const data = text(60_000);
  for (const level of [-5, 1, 3, 6, 9, 15, 19, 22]) roundTrip(data, level, `level ${level}`);
});

test('incompressible bytes, which go raw', () => {
  roundTrip(incompressible(300_000), 3, 'random');
});

test('one byte repeated, which goes as RLE', () => {
  roundTrip(new Uint8Array(500_000).fill(0x5a), 3, 'rle');
  roundTrip(new Uint8Array(1).fill(9), 3, 'one byte');
});

test('texture-like blocks, across many 128 KiB blocks', () => {
  const data = textureLike(1 << 21);
  for (const level of [1, 3, 9, 19]) roundTrip(data, level, `texture level ${level}`);
});

test('short inputs, where literals and sequences take their smallest headers', () => {
  for (let n = 0; n < 300; n += 7) roundTrip(text(n), 3, `${n} bytes of text`);
  roundTrip(new Uint8Array(0), 3, 'nothing');
});

test('mixed content in one stream: text, noise, runs', () => {
  const parts = [text(40_000), incompressible(20_000), new Uint8Array(30_000).fill(1), textureLike(50_000), text(10_000)];
  const data = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) { data.set(p, at); at += p.length; }
  for (const level of [1, 5, 12, 19]) roundTrip(data, level, `mixed level ${level}`);
});

test('a content checksum is skipped, and frames one after another are joined', () => {
  roundTrip(text(20_000), 3, 'checksummed', true);
  const a = text(5000), b = incompressible(3000);
  const joined = new Uint8Array([...compress(a), ...compress(b, 9)]);
  assert.deepEqual(zstdDecompress(joined, a.length + b.length), new Uint8Array([...a, ...b]));
});

test('what is not zstd, or not what was promised, says so', () => {
  assert.throws(() => zstdDecompress(Uint8Array.of(1, 2, 3, 4, 5, 6)), /not a zstd frame/);
  const packed = compress(text(1000));
  assert.throws(() => zstdDecompress(packed, 999), /zstd:/);
  assert.throws(() => zstdDecompress(packed, 1001), /zstd:/);
});

console.log(`\n${passed} checks passed\n`);
