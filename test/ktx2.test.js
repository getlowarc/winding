// The KTX2 container. Run: node test/ktx2.test.js
//
// Files are assembled here, byte by byte, the way the spec lays them out --
// identifier, header, index, level index, Data Format Descriptor, then the
// levels smallest first -- so every field the reader takes is one this test
// put there on purpose.

import assert from 'node:assert/strict';
import zlib from 'node:zlib';
import { readKTX2, transcodeKTX2, KTX2_NONE, KTX2_BASISLZ, KTX2_ZSTD } from '../src/scene/ktx2.js';
import { uastcToRGBA, uastcToBC7, uastcToASTC } from '../src/scene/uastc.js';
import { loadBasis, encodeUASTC, transcodeReference } from './fixtures/basis.mjs';

let passed = 0;
function test(name, fn) { fn(); passed++; console.log(`  ok  ${name}`); }
const B = await loadBasis();

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

// Solid-colour UASTC blocks (mode 8), so these need no encoder: a 4x4 block each.
const solidLevels = (w, h, count) => Array.from({ length: count }, (_, l) => {
  const blocks = Math.ceil(Math.max(1, w >> l) / 4) * Math.ceil(Math.max(1, h >> l) / 4);
  const level = new Uint8Array(blocks * 16);
  for (let b = 0; b < blocks; b++) level.set([0b10111 | ((40 * l) << 5) & 255, (40 * l) >> 3, 0, 0, 0], b * 16);
  return level;
});

test('transcoding a file gives each level in the format asked, as the block transcoders write it', () => {
  const levels = solidLevels(16, 8, 5);
  const file = ktx2({ width: 16, height: 8, levels, scheme: KTX2_ZSTD });
  for (const [target, transcode] of [['astc', uastcToASTC], ['bc7', uastcToBC7], ['rgba', uastcToRGBA]]) {
    const image = transcodeKTX2(file, target);
    assert.deepEqual([image.width, image.height, image.target], [16, 8, target]);
    assert.equal(image.levels.length, 5, `${target}: every level the file has`);
    image.levels.forEach((level, l) => {
      assert.deepEqual([level.width, level.height], [Math.max(1, 16 >> l), Math.max(1, 8 >> l)]);
      assert.deepEqual(level.data, transcode(levels[l], level.width, level.height), `${target}, level ${l}`);
    });
  }
});

test('one level, or a size that is not whole blocks, comes as RGBA8, for mips the GPU can make', () => {
  const one = transcodeKTX2(ktx2({ width: 8, height: 8, levels: solidLevels(8, 8, 1) }), 'bc7');
  assert.equal(one.target, 'rgba');
  assert.equal(one.levels.length, 1);
  assert.equal(one.levels[0].data.length, 8 * 8 * 4);
  const odd = transcodeKTX2(ktx2({ width: 6, height: 4, levels: solidLevels(6, 4, 3) }), 'astc');
  assert.equal(odd.target, 'rgba');
  assert.deepEqual(odd.levels.map((l) => l.data.length), [6 * 4 * 4, 3 * 2 * 4, 1 * 1 * 4], "the file's own mips, texel for texel");
});

test('what transcoding does not read yet, or cannot, says so', () => {
  const etc1s = ktx2({ width: 8, height: 8, levels: [bytes(40)], scheme: KTX2_BASISLZ, model: 163, globalData: bytes(36) });
  assert.throws(() => transcodeKTX2(etc1s, 'bc7'), /ktx2: ETC1S \(BasisLZ\) is not read yet, only UASTC/);
  const bc7 = ktx2({ vkFormat: 145, width: 8, height: 8, levels: [bytes(64)], model: 128 });
  assert.throws(() => transcodeKTX2(bc7, 'bc7'), /ktx2: not Basis Universal/);
  const cube = ktx2({ width: 4, height: 4, levels: solidLevels(4, 4, 1) });
  new DataView(cube.buffer).setUint32(36, 6, true);
  assert.throws(() => transcodeKTX2(cube, 'rgba'), /ktx2: not a 2D image/);
});

if (B === null) {
  console.log("  --  skipped: files from Basis's own encoder (npm run fetch:basis to run it)");
} else {
  test("a file from Basis's own encoder: UASTC, sRGB, alpha and every mip, as it wrote them", () => {
    const w = 96, h = 40, rgba = new Uint8Array(w * h * 4);
    for (let i = 0; i < rgba.length; i += 4) { rgba[i] = i % 251; rgba[i + 1] = (i >> 3) % 256; rgba[i + 2] = 200; rgba[i + 3] = (i >> 5) % 256; }
    const t = readKTX2(encodeUASTC(B, rgba, w, h));
    assert.equal(t.basis, 'uastc');
    assert.equal(t.srgb, true);
    assert.equal(t.channels, 'rgba');
    assert.equal(t.supercompression, KTX2_ZSTD);
    assert.equal(t.levels.length, 7, '96 down to 1');
    t.levels.forEach((level, i) => {
      assert.deepEqual([level.width, level.height], [Math.max(1, w >> i), Math.max(1, h >> i)]);
      assert.equal(level.data.length, 16 * Math.ceil(level.width / 4) * Math.ceil(level.height / 4), `level ${i}: a 16-byte block each 4x4`);
    });
    const linear = readKTX2(encodeUASTC(B, rgba.map((v, i) => (i % 4 === 3 ? 255 : v)), w, h, { srgb: false, mips: false }));
    assert.equal(linear.srgb, false);
    assert.equal(linear.channels, 'rgb', 'opaque: no alpha channel');
    assert.equal(linear.levels.length, 1);
  });

  test("a whole file transcoded is what Basis's own transcoder makes of it, level by level", () => {
    const w = 64, h = 32, rgba = new Uint8Array(w * h * 4);
    for (let i = 0; i < rgba.length; i += 4) { rgba[i] = (i * 7) % 256; rgba[i + 1] = (i >> 4) % 256; rgba[i + 2] = 90; rgba[i + 3] = 255 - ((i >> 6) % 256); }
    const file = encodeUASTC(B, rgba, w, h);
    for (const [target, format] of [['astc', 'cTFASTC_4x4_RGBA'], ['bc7', 'cTFBC7_RGBA'], ['rgba', 'cTFRGBA32']]) {
      transcodeKTX2(file.slice(), target).levels.forEach((level, l) => {
        assert.deepEqual(level.data, transcodeReference(B, file, l, format), `${target}, level ${l}`);
      });
    }
  });
}

console.log(`\n${passed} checks passed\n`);
