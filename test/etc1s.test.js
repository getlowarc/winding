// ETC1S (BasisLZ) decoding. Run: node test/etc1s.test.js
//
// Held to Basis Universal itself: images of every kind are encoded to ETC1S by
// Binomial's own encoder, at qualities from coarse (blocks share codebook
// entries, runs are long) to fine, and every level is decoded both by
// src/scene/etc1s.js and by Binomial's transcoder: the RGBA must be the same.
// Needs the reference (npm run fetch:basis); without it, these checks skip.

import assert from 'node:assert/strict';
import { readKTX2, transcodeKTX2 } from '../src/scene/ktx2.js';
import { loadBasis, encodeETC1S, transcodeReference } from './fixtures/basis.mjs';

let passed = 0;
function test(name, fn) { fn(); passed++; console.log(`  ok  ${name}`); }

const B = await loadBasis();
console.log('\netc1s');
if (B === null) {
  console.log("  --  skipped: Basis Universal's own encoder and transcoder (npm run fetch:basis to run these)");
  console.log(`\n${passed} checks passed\n`);
  process.exit(0);
}

let seed = 5;
const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const clamp = (v) => Math.max(0, Math.min(255, Math.round(v)));
const make = (name, w, h, f) => {
  const rgba = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) rgba.set(f(x, y).map(clamp), (y * w + x) * 4);
  return { name, w, h, rgba };
};
const IMAGES = [
  make('gradients and noise', 96, 64, (x, y) => [x * 2.5 + random() * 30, y * 4, (x + y) * 1.5, 255]),
  make('grey', 64, 64, (x, y) => { const g = (x ^ y) * 4 + random() * 10; return [g, g, g, 255]; }),
  // Flat areas: the long endpoint and selector runs.
  make('mostly flat', 128, 128, (x, y) => (x < 100 && y < 100 ? [40, 90, 200, 255] : [x * 2, y * 2, 30, 255])),
  make('alpha ramps', 64, 32, (x, y) => [x * 4, 128, 255 - y * 8, (x * 3 + y * 5) & 255]),
  make('cut-out alpha', 64, 64, (x, y) => [(x * 5) & 255, (y * 5) & 255, 90, ((x >> 3) + (y >> 3)) % 2 ? 255 : 0]),
  make('an odd size', 37, 21, (x, y) => [x * 6, y * 11, (x * y) & 255, 200 + (x & 31)]),
  make('one texel', 1, 1, () => [200, 100, 50, 255]),
];

test("every level decodes to the RGBA Basis's own transcoder gives, at coarse to fine quality", () => {
  let levels = 0, files = 0;
  for (const { name, w, h, rgba } of IMAGES) {
    for (const quality of [1, 64, 255]) {
      const file = encodeETC1S(B, rgba, w, h, { quality });
      files++;
      const image = transcodeKTX2(file, 'bc7');
      assert.equal(image.target, 'rgba', 'ETC1S comes as RGBA8 for now');
      image.levels.forEach((level, l) => {
        const theirs = transcodeReference(B, file, l, 'cTFRGBA32');
        assert.equal(level.data.length, theirs.length, `${name}, quality ${quality}, level ${l}: size`);
        for (let i = 0; i < theirs.length; i++) {
          if (level.data[i] !== theirs[i]) assert.fail(`${name}, quality ${quality}, level ${l}: byte ${i} is ${level.data[i]}, Basis gives ${theirs[i]}`);
        }
        levels++;
      });
    }
  }
  console.log(`      ${files} files, ${levels} levels, all the same`);
});

test('an ETC1S file is read as BasisLZ with its global data, and one level comes as that level', () => {
  const { w, h, rgba } = IMAGES[0];
  const file = encodeETC1S(B, rgba, w, h, { mips: false });
  const t = readKTX2(file);
  assert.equal(t.basis, 'etc1s');
  assert.ok(t.globalData.length > 20);
  const image = transcodeKTX2(file, 'astc', { gpu: true });
  assert.deepEqual([image.target, image.levels.length, image.levels[0].data.length], ['rgba', 1, w * h * 4]);
});

console.log(`\n${passed} checks passed\n`);
