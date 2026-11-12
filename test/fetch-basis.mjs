// The Basis Universal reference, for the transcoder tests: Binomial's own
// transcoder and encoder, built for the web. Test-only -- the engine never
// loads them -- and not kept in the repository: run `npm run fetch:basis`,
// and the tests that compare against them run; without them, those skip.
//
// Pinned to one commit and checked by SHA-256, so a rerun fetches the same
// bytes or fails. Apache-2.0: https://github.com/BinomialLLC/basis_universal

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const COMMIT = '073eb6a0a34a23d77e75e54f41dcd9653a52da83';
const FILES = {
  'webgl/transcoder/build/basis_transcoder.js': null,
  'webgl/transcoder/build/basis_transcoder.wasm': null,
  'webgl/encoder/build/basis_encoder.js': null,
  'webgl/encoder/build/basis_encoder.wasm': null,
  LICENSE: null,
};
// The checksums, recorded the first time; a fetch that disagrees is refused.
const SHA256 = JSON.parse(readFileSync(new URL('./fixtures/basis.sha256.json', import.meta.url), 'utf8').toString() || '{}');
const DIR = new URL('./vendor/basis/', import.meta.url);

mkdirSync(DIR, { recursive: true });
const record = process.argv.includes('--record');
for (const path of Object.keys(FILES)) {
  const name = path.split('/').pop();
  const target = new URL(name, DIR);
  if (existsSync(target) && !record) {
    const hash = createHash('sha256').update(readFileSync(target)).digest('hex');
    if (hash === SHA256[name]) { console.log(`have ${name}`); continue; }
  }
  const url = `https://raw.githubusercontent.com/BinomialLLC/basis_universal/${COMMIT}/${path}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`fetch-basis: ${url} returned ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const hash = createHash('sha256').update(bytes).digest('hex');
  if (record) SHA256[name] = hash;
  else if (hash !== SHA256[name]) throw new Error(`fetch-basis: ${name} has SHA-256 ${hash}, not ${SHA256[name]}`);
  writeFileSync(target, bytes);
  console.log(`fetched ${name}, ${bytes.length} bytes`);
}
if (record) {
  writeFileSync(new URL('./fixtures/basis.sha256.json', import.meta.url), `${JSON.stringify(SHA256, null, 2)}\n`);
  console.log('recorded checksums in test/fixtures/basis.sha256.json');
}
