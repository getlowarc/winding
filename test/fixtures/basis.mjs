// Binomial's Basis Universal encoder and transcoder, for tests that compare
// against the reference. Fetched by `npm run fetch:basis` into test/vendor/
// (never committed); loadBasis() is null without them, and those tests skip.
//
// The builds are CommonJS, and this package is "type": "module", under which
// Node reads them as ES modules and their export never runs -- so their text is
// run in a CommonJS wrapper instead.

import { createRequire } from 'node:module';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const DIR = fileURLToPath(new URL('../vendor/basis/', import.meta.url));
const require = createRequire(import.meta.url);
let basis;

/** The encoder module (which carries the transcoder too), or null when it has not been fetched. */
export async function loadBasis() {
  if (basis !== undefined) return basis;
  const file = path.join(DIR, 'basis_encoder.js');
  if (!existsSync(file) || !existsSync(path.join(DIR, 'basis_encoder.wasm'))) return (basis = null);
  const module = { exports: {} };
  new Function('module', 'exports', 'require', '__filename', '__dirname', readFileSync(file, 'utf8'))(module, module.exports, require, file, DIR);
  // Its progress lines go to the console; tests want their own output.
  basis = await module.exports({ locateFile: (name) => path.join(DIR, name), print: () => {}, printErr: () => {} });
  basis.initializeBasis();
  return basis;
}

/**
 * A w x h RGBA image as a UASTC LDR 4x4 KTX2 file, zstd-supercompressed,
 * with mips unless `mips` is false. `rdo` trades quality for size as Basis's
 * own option does; `srgb` sets the transfer function.
 */
export function encodeUASTC(B, rgba, w, h, { srgb = true, mips = true, rdo = false } = {}) {
  const e = new B.BasisEncoder();
  try {
    e.setSliceSourceImage(0, rgba, w, h, B.ldr_image_type.cRGBA32.value);
    e.setFormatMode(B.basis_tex_format.cUASTC_LDR_4x4.value);
    e.setCreateKTX2File(true);
    e.setKTX2UASTCSupercompression(true);
    e.setKTX2AndBasisSRGBTransferFunc(srgb);
    e.setMipGen(mips);
    e.setCheckForAlpha(true);
    if (rdo) e.setRDOUASTC(true);
    const out = new Uint8Array(w * h * 8 + (1 << 16));
    const n = e.encode(out);
    if (!n) throw new Error('basis: encode failed');
    return out.slice(0, n);
  } finally {
    e.delete();
  }
}

/** Level `level` of a KTX2 file, transcoded by the reference to `format` (a transcoder_texture_format name). */
export function transcodeReference(B, ktx2, level, format) {
  const k = new B.KTX2File(ktx2);
  try {
    if (!k.isValid() || !k.startTranscoding()) throw new Error('basis: not a file the reference transcodes');
    const f = B.transcoder_texture_format[format].value;
    const out = new Uint8Array(k.getImageTranscodedSizeInBytes(level, 0, 0, f));
    if (!k.transcodeImage(out, level, 0, 0, f, 0, -1, -1)) throw new Error(`basis: transcoding level ${level} to ${format} failed`);
    return out;
  } finally {
    k.close();
    k.delete();
  }
}
