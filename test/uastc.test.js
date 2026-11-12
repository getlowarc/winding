// UASTC transcoding. Run: node test/uastc.test.js
//
// Held to Basis Universal itself: images of every kind are encoded to UASTC by
// Binomial's own encoder, then every level transcoded both by src/scene/uastc.js
// and by Binomial's transcoder, and the bytes must be the same. Needs the
// reference (npm run fetch:basis); without it, these checks skip.

import assert from 'node:assert/strict';
import { readKTX2 } from '../src/scene/ktx2.js';
import { unpackBlock, uastcToRGBA, uastcToBC7, uastcToASTC } from '../src/scene/uastc.js';
import { loadBasis, encodeUASTC, transcodeReference } from './fixtures/basis.mjs';

let passed = 0;
function test(name, fn) { fn(); passed++; console.log(`  ok  ${name}`); }

const B = await loadBasis();
console.log('\nuastc');
if (B === null) {
  console.log("  --  skipped: Basis Universal's own encoder and transcoder (npm run fetch:basis to run these)");
  console.log(`\n${passed} checks passed\n`);
  process.exit(0);
}

let seed = 11;
const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);

/** Test images, each { name, w, h, rgba }: chosen to reach every UASTC mode. */
function images() {
  const make = (name, w, h, f) => {
    const rgba = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) rgba.set(f(x, y), (y * w + x) * 4);
    return { name, w, h, rgba };
  };
  const clamp = (v) => Math.max(0, Math.min(255, Math.round(v)));
  return [
    make('smooth gradients', 64, 64, (x, y) => [x * 4, y * 4, (x + y) * 2, 255]),
    make('noise', 32, 32, () => [random() * 256, random() * 256, random() * 256, 255].map(Math.floor)),
    make('hard edges in two and three colours', 64, 64, (x, y) => {
      const k = ((x * 7 + y * 3) >> 3) % 3;
      return [[200, 30, 40, 255], [20, 180, 60, 255], [40, 50, 220, 255]][k].map((v) => clamp(v + (random() - 0.5) * 20));
    }),
    make('stripes and diagonals', 48, 48, (x, y) => ((x + y) % 8 < 4 ? [250, 250, 240, 255] : [10, 10, 30, 255])),
    make('alpha ramps', 64, 64, (x, y) => [x * 4, 128, 255 - y * 4, (x * 3 + y) & 255]),
    make('alpha noise', 32, 32, () => [random() * 256, random() * 256, random() * 256, random() * 256].map(Math.floor)),
    make('cut-out alpha', 64, 64, (x, y) => [(x * 5) & 255, (y * 5) & 255, 90, ((x >> 2) + (y >> 2)) % 2 ? 255 : 0]),
    make('grey and alpha', 64, 64, (x, y) => { const g = clamp(x * 3 + random() * 20); return [g, g, g, clamp(y * 4 + random() * 30)]; }),
    make('grey', 32, 32, (x, y) => { const g = clamp((x ^ y) * 8); return [g, g, g, 255]; }),
    make('solid colours', 32, 32, (x, y) => [[12, 200, 77, 255], [255, 255, 255, 255], [0, 0, 0, 0], [130, 7, 250, 128]][((x >> 3) + (y >> 3)) % 4]),
    // Each 4x4 block two or three flat colours in a random split, the partitioned modes' ground.
    make('blocks of two and three colours', 64, 64, (x, y) => {
      const b = ((y >> 2) * 16 + (x >> 2)) * 7919 % 251, k = b % 3 === 0 ? (x + y * (b & 3)) % 3 : ((x * (b & 1 ? 1 : 0) + y) >> 1) % 2;
      return [[(b * 37) & 255, (b * 91) & 255, (b * 13) & 255, 255], [255 - b, b, 128, 255], [b, 255 - ((b * 5) & 255), 40, 255]][k];
    }),
    make('blocks of two and three colours, with alpha', 64, 64, (x, y) => {
      const b = ((y >> 2) * 16 + (x >> 2)) * 104729 % 251, k = ((x + (b & 1)) >> 1) % 2;
      return [[(b * 37) & 255, (b * 91) & 255, (b * 13) & 255, b], [255 - b, b, 128, 255 - b]][k];
    }),
    make('very smooth', 64, 64, (x, y) => [128 + Math.round(20 * Math.sin(x / 9)), 100 + Math.round(30 * Math.cos(y / 11)), 90, 255]),
    make('an odd size', 37, 21, (x, y) => [x * 6, y * 11, (x * y) & 255, 200 + (x & 31)]),
  ];
}

const corpus = [];
for (const image of images()) {
  for (const level of [0, 2, 4]) {
    for (const rdo of level === 2 ? [false, true] : [false]) {
      corpus.push({ ...image, rdo, name: `${image.name}, level ${level}`, file: encodeUASTC(B, image.rgba, image.w, image.h, { rdo, level }) });
    }
  }
}

test("every block decodes to the RGBA Basis's own transcoder gives, at every level", () => {
  const modes = new Set();
  for (const { name, rdo, file } of corpus) {
    const t = readKTX2(file);
    t.levels.forEach((level, l) => {
      for (let b = 0; b < level.data.length; b += 16) modes.add(unpackBlock(level.data, b).mode);
      const ours = uastcToRGBA(level.data, level.width, level.height);
      const theirs = transcodeReference(B, file, l, 'cTFRGBA32');
      assert.equal(ours.length, theirs.length, `${name}${rdo ? ', RDO' : ''}, level ${l}: size`);
      for (let i = 0; i < ours.length; i++) {
        if (ours[i] !== theirs[i]) assert.fail(`${name}${rdo ? ', RDO' : ''}, level ${l}: byte ${i} (texel ${i >> 2}) is ${ours[i]}, Basis gives ${theirs[i]}`);
      }
    });
  }
  console.log(`      modes seen: ${[...modes].sort((a, b) => a - b).join(' ')}`);
});

/** Every block of every level transcoded by `transcode` and by Basis to `format`, differences reported by UASTC mode. */
function sameBlocks(transcode, format) {
  // The first differing block of each UASTC mode, and how many differ, so one run shows every mode's state.
  const first = new Map(), counts = new Map();
  let blocks = 0;
  for (const { name, rdo, file } of corpus) {
    const t = readKTX2(file);
    t.levels.forEach((level, l) => {
      const ours = transcode(level.data, level.width, level.height);
      const theirs = transcodeReference(B, file, l, format);
      assert.equal(ours.length, theirs.length, `${name}, level ${l}: size`);
      for (let b = 0; b < ours.length; b += 16) {
        blocks++;
        let same = true;
        for (let i = 0; i < 16; i++) if (ours[b + i] !== theirs[b + i]) { same = false; break; }
        if (same) continue;
        const mode = unpackBlock(level.data, b).mode;
        counts.set(mode, (counts.get(mode) ?? 0) + 1);
        if (!first.has(mode)) {
          first.set(mode, `${name}${rdo ? ', RDO' : ''}, level ${l}, block ${b / 16}:\n        ours  ${Array.from(ours.subarray(b, b + 16)).join(',')}\n        Basis ${Array.from(theirs.subarray(b, b + 16)).join(',')}`);
        }
      }
    });
  }
  for (const [mode, text] of [...first].sort((a, b) => a[0] - b[0])) console.log(`      mode ${mode}: ${counts.get(mode)} blocks differ; first, ${text}`);
  assert.equal(first.size, 0, `blocks that differ from Basis's, of ${blocks}`);
  console.log(`      ${blocks} blocks, all the same`);
}

test("every block becomes the BC7 block Basis's own transcoder writes, at every level", () => sameBlocks(uastcToBC7, 'cTFBC7_RGBA'));
test("every block becomes the ASTC 4x4 block Basis's own transcoder writes, at every level", () => sameBlocks(uastcToASTC, 'cTFASTC_4x4_RGBA'));

console.log(`\n${passed} checks passed\n`);
