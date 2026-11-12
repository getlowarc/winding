// ETC1S (Basis Universal's BasisLZ supercompression), decoded: a KTX2 file's
// global data -- the endpoint and selector codebooks and the slices' Huffman
// tables -- then each level's slice, block by block, to which endpoint and
// which selector entry it uses, and from those to RGBA8 texels.
//
// Written from Binomial's ".basis File Format and ETC1S Texture Video
// Specification" (public domain), sections 6 to 11, and the KTX 2.0
// specification's BasisLZ global data; held to Basis Universal's own
// transcoder by test/etc1s.test.js. Texture video (P-frames) is not read:
// KTX2 for glTF holds still images.
//
// An ETC1S block is an ETC1 block with one colour and one intensity table for
// both halves: a 5-bit RGB base, a 3-bit table, and a 2-bit selector a texel.

const fail = (why) => { throw new Error(`etc1s: ${why}`); };

/** Bits LSB first, from bytes [start, end); past the end, zeros. At most 16 a read. */
class Bits {
  constructor(bytes, start, end) { this.bytes = bytes; this.at = start; this.end = end; this.buffer = 0; this.held = 0; }
  /** At least 17 bits held (the buffer is 32): enough for any one read. */
  fill() {
    while (this.held <= 24) {
      const byte = this.at < this.end ? this.bytes[this.at] : 0;
      this.at++;
      this.buffer = (this.buffer | (byte << this.held)) >>> 0;
      this.held += 8;
    }
  }
  get(n) {
    if (this.held < n) this.fill();
    const v = this.buffer & ((1 << n) - 1);
    this.buffer >>>= n;
    this.held -= n;
    return v;
  }
  /** One bit. */
  bit() { return this.get(1); }
}

// ---- Huffman (section 6): canonical codes as Deflate's, sent LSB first.

const CODELENGTH_ORDER = [17, 18, 19, 20, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15, 16];

/** A decoder for code lengths `lengths`: by the next maxBits bits read, each entry symbol << 5 | length. */
function huffmanTable(lengths) {
  let maxBits = 0;
  for (const l of lengths) if (l > maxBits) maxBits = l;
  if (maxBits === 0) {
    // No codes: a table of one symbol and no bits. (A single used symbol is sent with a 1-bit code.)
    return { maxBits: 0, entries: Uint32Array.of(0) };
  }
  const count = new Array(maxBits + 1).fill(0);
  for (const l of lengths) if (l) count[l]++;
  // RFC 1951 3.2.2: each length's first code, the shorter ones' codes counted up and shifted.
  const next = new Array(maxBits + 1).fill(0);
  for (let bits = 1, code = 0; bits <= maxBits; bits++) { code = (code + count[bits - 1]) << 1; next[bits] = code; }
  const entries = new Uint32Array(1 << maxBits).fill(0xffffffff);
  lengths.forEach((l, symbol) => {
    if (!l) return;
    const code = next[l]++;
    // Sent MSB of the code first into an LSB-first stream: the table is indexed by the code reversed.
    let reversed = 0;
    for (let i = 0; i < l; i++) reversed |= ((code >> i) & 1) << (l - 1 - i);
    for (let fill = reversed; fill < entries.length; fill += 1 << l) entries[fill] = (symbol << 5) | l;
  });
  return { maxBits, entries };
}

/** The next symbol by `table`. */
function decode(bits, table) {
  if (table.maxBits === 0) return 0;
  if (bits.held < table.maxBits) bits.fill();
  const entry = table.entries[bits.buffer & ((1 << table.maxBits) - 1)];
  if (entry === 0xffffffff) fail('a Huffman code that is not in its table');
  const n = entry & 31;
  bits.buffer >>>= n;
  bits.held -= n;
  return entry >>> 5;
}

/** A compressed Huffman table (section 6), read from `bits`. */
function readHuffman(bits) {
  const total = bits.get(14);
  if (total < 1) fail('a Huffman table with no symbols');
  const codelengthCodes = bits.get(5);
  if (codelengthCodes < 1 || codelengthCodes > 21) fail(`${codelengthCodes} code length codes`);
  const small = new Array(21).fill(0);
  for (let i = 0; i < codelengthCodes; i++) small[CODELENGTH_ORDER[i]] = bits.get(3);
  const smallTable = huffmanTable(small);
  const lengths = new Uint8Array(total);
  let i = 0;
  while (i < total) {
    const symbol = decode(bits, smallTable);
    if (symbol <= 16) { lengths[i++] = symbol; continue; }
    let run, value = 0;
    if (symbol === 17) run = bits.get(3) + 3;
    else if (symbol === 18) run = bits.get(7) + 11;
    else {
      run = symbol === 19 ? bits.get(2) + 3 : bits.get(7) + 7;
      if (i === 0 || lengths[i - 1] === 0) fail('a repeat with no length before it');
      value = lengths[i - 1];
    }
    if (i + run > total) fail('code lengths past the symbols');
    lengths.fill(value, i, i + run);
    i += run;
  }
  return huffmanTable(lengths);
}

// ---- codebooks (sections 7, 8, 9)

/** The endpoint codebook: per entry, RGB as 5-bit values and the 3-bit intensity table. */
function readEndpoints(bytes, start, end, count) {
  const bits = new Bits(bytes, start, end);
  const models = [readHuffman(bits), readHuffman(bits), readHuffman(bits)];
  const intensityModel = readHuffman(bits);
  const grey = bits.bit();
  const colour = new Uint8Array(count * 3), intensity = new Uint8Array(count);
  const prev = [16, 16, 16];
  let prevIntensity = 0;
  for (let i = 0; i < count; i++) {
    prevIntensity = intensity[i] = (decode(bits, intensityModel) + prevIntensity) & 7;
    for (let c = 0; c < (grey ? 1 : 3); c++) {
      const model = prev[c] <= 9 ? models[0] : prev[c] <= 21 ? models[1] : models[2];
      prev[c] = colour[i * 3 + c] = (prev[c] + decode(bits, model)) & 31;
    }
    if (grey) colour[i * 3 + 1] = colour[i * 3 + 2] = colour[i * 3];
  }
  return { colour, intensity };
}

/** The selector codebook: per entry, four bytes, row y's four 2-bit selectors in byte y, x = 0 lowest. */
function readSelectors(bytes, start, end, count) {
  const bits = new Bits(bytes, start, end);
  if (bits.bit()) fail('global selector codebooks, which Basis Universal never writes');
  if (bits.bit()) fail('hybrid selector codebooks, which Basis Universal never writes');
  const raw = bits.bit();
  const selectors = new Uint8Array(count * 4);
  if (raw) {
    for (let i = 0; i < count * 4; i++) selectors[i] = bits.get(8);
    return selectors;
  }
  const model = readHuffman(bits);
  const prev = [0, 0, 0, 0];
  for (let i = 0; i < count; i++) {
    for (let j = 0; j < 4; j++) {
      prev[j] = selectors[i * 4 + j] = i === 0 ? bits.get(8) : decode(bits, model) ^ prev[j];
    }
  }
  return selectors;
}

/**
 * A KTX2 file's BasisLZ global data (`globalData`, `imageCount` images):
 * { endpoints, selectors, tables, images: [{ flags, rgb: [offset, length],
 * alpha: [offset, length] }] }, offsets within each image's level.
 */
export function readGlobalData(globalData, imageCount) {
  const bytes = globalData;
  if (bytes.length < 20) fail('BasisLZ global data too short');
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const endpointCount = dv.getUint16(0, true), selectorCount = dv.getUint16(2, true);
  const endpointsLength = dv.getUint32(4, true), selectorsLength = dv.getUint32(8, true);
  const tablesLength = dv.getUint32(12, true), extendedLength = dv.getUint32(16, true);
  const images = [];
  let at = 20;
  for (let i = 0; i < imageCount; i++, at += 20) {
    if (at + 20 > bytes.length) fail('image descriptions past the global data');
    images.push({
      flags: dv.getUint32(at, true),
      rgb: [dv.getUint32(at + 4, true), dv.getUint32(at + 8, true)],
      alpha: [dv.getUint32(at + 12, true), dv.getUint32(at + 16, true)],
    });
  }
  const endpointsAt = at, selectorsAt = endpointsAt + endpointsLength, tablesAt = selectorsAt + selectorsLength;
  if (tablesAt + tablesLength + extendedLength > bytes.length) fail('codebooks past the global data');
  const endpoints = readEndpoints(bytes, endpointsAt, selectorsAt, endpointCount);
  const selectors = readSelectors(bytes, selectorsAt, tablesAt, selectorCount);
  const bits = new Bits(bytes, tablesAt, tablesAt + tablesLength);
  const tables = {
    endpointPred: readHuffman(bits), deltaEndpoint: readHuffman(bits),
    selector: readHuffman(bits), historyRle: readHuffman(bits), historySize: bits.get(13),
  };
  return { endpointCount, selectorCount, endpoints, selectors, tables, images };
}

// ---- slices (section 10)

/** VLC: chunks of `chunkBits`, each with a continue bit above it. */
function vlc(bits, chunkBits) {
  const size = 1 << chunkBits;
  let v = 0;
  for (let shift = 0; shift < 32; shift += chunkBits) {
    const s = bits.get(chunkBits + 1);
    v += (s & (size - 1)) * 2 ** shift;
    if ((s & size) === 0) break;
  }
  return v;
}

/**
 * One slice of `across` x `down` blocks at bytes [start, end): each block's
 * endpoint and selector entry, raster order, into Uint16Arrays.
 */
export function decodeSlice(global, bytes, start, end, across, down) {
  const { tables, endpointCount, selectorCount } = global;
  const bits = new Bits(bytes, start, end);
  const endpoints = new Uint16Array(across * down), selectors = new Uint16Array(across * down);
  // Two rows of per-block prediction state: the endpoint used, and the next row's predictor bits.
  const predEndpoint = [new Uint16Array(across), new Uint16Array(across)];
  const predBits = [new Uint8Array(across), new Uint8Array(across)];
  const historySize = tables.historySize;
  const historyRle = selectorCount + historySize;
  // The approximate move-to-front selector history (10.1).
  const history = new Int32Array(historySize);
  let rover = historySize >> 1;
  let rleCount = 0, curPredBits = 0, prevPredSym = 0, predRepeat = 0, prevEndpoint = 0;
  for (let y = 0; y < down; y++) {
    const row = y & 1;
    for (let x = 0; x < across; x++) {
      if ((x & 1) === 0) {
        if (row === 0) {
          if (predRepeat) { predRepeat--; curPredBits = prevPredSym; } else {
            curPredBits = decode(bits, tables.endpointPred);
            if (curPredBits === 256) {
              predRepeat = vlc(bits, 4) + 3 - 1;
              curPredBits = prevPredSym;
            } else prevPredSym = curPredBits;
          }
          predBits[row ^ 1][x] = curPredBits >> 4;
        } else curPredBits = predBits[row][x];
      }
      const pred = curPredBits & 3;
      curPredBits >>= 2;
      let endpoint;
      if (pred === 0) endpoint = prevEndpoint;
      else if (pred === 1) endpoint = predEndpoint[row ^ 1][x];
      else if (pred === 2) endpoint = predEndpoint[row ^ 1][x - 1];
      else {
        endpoint = decode(bits, tables.deltaEndpoint) + prevEndpoint;
        if (endpoint >= endpointCount) endpoint -= endpointCount;
      }
      predEndpoint[row][x] = endpoint;
      prevEndpoint = endpoint;

      let symbol;
      if (rleCount > 0) { rleCount--; symbol = selectorCount; } else {
        symbol = decode(bits, tables.selector);
        if (symbol === historyRle) {
          const run = decode(bits, tables.historyRle);
          rleCount = (run === 63 ? vlc(bits, 7) : run) + 3;
          symbol = selectorCount;
          rleCount--;
        }
      }
      let selector;
      if (symbol >= selectorCount) {
        const i = symbol - selectorCount;
        if (i >= historySize) fail('a selector history entry past its size');
        selector = history[i];
        if (i !== 0) { const a = history[i >> 1]; history[i >> 1] = history[i]; history[i] = a; }
      } else {
        selector = symbol;
        if (historySize) { history[rover++] = selector; if (rover === historySize) rover = historySize >> 1; }
      }
      if (endpoint >= endpointCount || selector >= selectorCount) fail('a block naming a codebook entry past its end');
      endpoints[y * across + x] = endpoint;
      selectors[y * across + x] = selector;
    }
  }
  return { endpoints, selectors };
}

// ---- to texels

/** ETC1's intensity modifiers, each table smallest first, as an ETC1S selector indexes them. */
const INTENSITY = [[-8, -2, 2, 8], [-17, -5, 5, 17], [-29, -9, 9, 29], [-42, -13, 13, 42],
  [-60, -18, 18, 60], [-80, -24, 24, 80], [-106, -33, 33, 106], [-183, -47, 47, 183]];

const LITTLE_ENDIAN = new Uint8Array(Uint32Array.of(1).buffer)[0] === 1;

/**
 * Each endpoint entry's four colours, by entry * 4 + selector: RGBA (A 255)
 * packed as a little-endian word, and the green alone, for an alpha slice.
 * Worked out once a file rather than once a texel.
 */
function palettes(global) {
  if (global.palette) return global.palette;
  const { colour, intensity } = global.endpoints, count = global.endpointCount;
  const rgba = new Uint32Array(count * 4), green = new Uint8Array(count * 4);
  for (let e = 0; e < count; e++) {
    const table = INTENSITY[intensity[e]];
    for (let s = 0; s < 4; s++) {
      const [r, g, b] = [0, 1, 2].map((c) => {
        const v = colour[e * 3 + c];
        return Math.min(255, Math.max(0, ((v << 3) | (v >> 2)) + table[s]));
      });
      rgba[e * 4 + s] = (r | (g << 8) | (b << 16) | (255 << 24)) >>> 0;
      green[e * 4 + s] = g;
    }
  }
  return (global.palette = { rgba, green });
}

/** A level's decoded slice as RGBA8 texels `width` x `height` (alpha from `alpha`'s green, or 255). */
export function etc1sToRGBA(global, slice, alpha, width, height) {
  const across = Math.ceil(width / 4);
  const out = new Uint8Array(width * height * 4);
  const words = LITTLE_ENDIAN ? new Uint32Array(out.buffer) : null;
  const { rgba, green } = palettes(global), selectors = global.selectors;
  const blocks = slice.endpoints.length;
  for (let b = 0; b < blocks; b++) {
    const bx = (b % across) * 4, by = Math.floor(b / across) * 4;
    const e = slice.endpoints[b] * 4, s = slice.selectors[b] * 4;
    const ae = alpha ? alpha.endpoints[b] * 4 : 0, as = alpha ? alpha.selectors[b] * 4 : 0;
    const w = Math.min(4, width - bx), h = Math.min(4, height - by);
    for (let y = 0; y < h; y++) {
      const row = selectors[s + y], alphaRow = alpha ? selectors[as + y] : 0;
      for (let x = 0; x < w; x++) {
        const t = (by + y) * width + bx + x, colour = rgba[e + ((row >> (2 * x)) & 3)];
        if (words) words[t] = colour;
        else { out[t * 4] = colour & 255; out[t * 4 + 1] = (colour >> 8) & 255; out[t * 4 + 2] = (colour >> 16) & 255; out[t * 4 + 3] = 255; }
        if (alpha) out[t * 4 + 3] = green[ae + ((alphaRow >> (2 * x)) & 3)];
      }
    }
  }
  return out;
}

// ---- to BC7 (mode 5), as Basis Universal's transcoder writes it --------
//
// Ported from basisu_transcoder.cpp (Apache-2.0, Binomial LLC): its ETC1S to
// BC7 mode 5 conversion (convert_etc1s_to_bc7_m5_color and _alpha), its
// default chroma filter (chroma_filter_bc7_mode5) and the real-time mode 5
// encoder that filter re-encodes with. Its conversion tables are not copied:
// each entry is worked out here, when first used, by the same search Basis
// ran to make them (create_etc1_to_bc7_m5_color/alpha_conversion_table).
// test/etc1s.test.js holds the result to Basis's own output, block for block.

const f = Math.fround;
const expand5 = (v) => (v << 3) | (v >> 2);
const clamp255 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);
const SELECTOR_RANGES = [[0, 3], [1, 3], [0, 2], [1, 2], [2, 3], [0, 1]];
const RANGE_INDEX = [[-1, 5, 2, 0], [-1, -1, 3, 1], [-1, -1, -1, 4], [-1, -1, -1, -1]];
const MAPPINGS = [[0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 1, 3], [0, 0, 2, 3], [0, 1, 1, 1],
  [0, 1, 2, 2], [0, 1, 2, 3], [0, 2, 3, 3], [1, 2, 2, 2], [1, 2, 3, 3]];

/** Per selector entry: its lowest and highest selector and how many differ. */
function selectorFlags(global) {
  if (global.selectorFlags) return global.selectorFlags;
  const s = global.selectors, n = global.selectorCount;
  const lo = new Uint8Array(n), hi = new Uint8Array(n), unique = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    let seen = 0;
    for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) seen |= 1 << ((s[i * 4 + y] >> (2 * x)) & 3);
    lo[i] = 31 - Math.clz32(seen & -seen);
    hi[i] = 31 - Math.clz32(seen);
    unique[i] = (seen & 1) + ((seen >> 1) & 1) + ((seen >> 2) & 1) + ((seen >> 3) & 1);
  }
  return (global.selectorFlags = { lo, hi, unique });
}

/** g_bc7_m5_equals_1: for each 8-bit value, the 7-bit endpoints whose index-1 colour comes closest. */
let EQUALS_1 = null;
export function equals1() {
  if (EQUALS_1) return EQUALS_1;
  EQUALS_1 = new Uint8Array(512);   // [value * 2] = lo, [value * 2 + 1] = hi
  for (let i = 0; i < 256; i++) {
    let lowest = 256;
    for (let lo = 0; lo < 128; lo++) {
      for (let hi = 0; hi < 128; hi++) {
        const e = Math.abs(((((lo << 1) | (lo >> 6)) * 43 + ((hi << 1) | (hi >> 6)) * 21 + 32) >> 6) - i);
        if (e < lowest) { lowest = e; EQUALS_1[i * 2] = lo; EQUALS_1[i * 2 + 1] = hi; }
      }
    }
  }
  return EQUALS_1;
}

/** The colour table's ten mappings for (intensity, 5-bit value, selector range): lo, hi, err each. */
const COLOUR_TABLE = new Map();
export function colourSolutions(inten, g, range) {
  const key = (inten * 32 + g) * 6 + range;
  let solutions = COLOUR_TABLE.get(key);
  if (solutions) return solutions;
  solutions = new Uint16Array(30);
  const [low, high] = SELECTOR_RANGES[range], base = expand5(g), table = INTENSITY[inten];
  const block = table.map((m) => clamp255(base + m));
  const heavy = inten === 7 && low === 0 && high === 3;
  const colours = [0, 0, 0, 0];
  for (let m = 0; m < 10; m++) {
    const map = MAPPINGS[m];
    let bestLo = 0, bestHi = 0, best = Infinity;
    for (let hi = 0; hi < 128; hi++) {
      const c3 = (hi << 1) | (hi >> 6);
      for (let lo = 0; lo < 128; lo++) {
        const c0 = (lo << 1) | (lo >> 6);
        colours[0] = c0; colours[3] = c3;
        colours[1] = (c0 * 43 + c3 * 21 + 32) >> 6;
        colours[2] = (c0 * 21 + c3 * 43 + 32) >> 6;
        let err = 0;
        for (let s = low; s <= high; s++) {
          const d = block[s] - colours[map[s]];
          err += d * d * (heavy && (s === 0 || s === 3) ? 5 : 1);
        }
        if (err < best) { best = err; bestLo = lo; bestHi = hi; }
      }
    }
    solutions[m * 3] = bestLo; solutions[m * 3 + 1] = bestHi; solutions[m * 3 + 2] = Math.min(best, 0xffff);
  }
  COLOUR_TABLE.set(key, solutions);
  return solutions;
}

/** The alpha table entry for (intensity, 5-bit value, selector range): [lo, hi, selector translation]. */
const ALPHA_TABLE = new Map();
export function alphaSolution(inten, g, range) {
  const key = (inten * 32 + g) * 6 + range;
  let solution = ALPHA_TABLE.get(key);
  if (solution) return solution;
  const [low, high] = SELECTOR_RANGES[range], base = expand5(g), table = INTENSITY[inten];
  const block = table.map((m) => clamp255(base + m));
  const heavy = inten === 7 && low === 0 && high === 3;
  let best = Infinity;
  solution = [0, 0, 0];
  const colours = [0, 0, 0, 0];
  for (let hi = 0; hi < 256; hi++) {
    for (let lo = 0; lo < 256; lo++) {
      colours[0] = lo; colours[3] = hi;
      colours[1] = (lo * 43 + hi * 21 + 32) >> 6;
      colours[2] = (lo * 21 + hi * 43 + 32) >> 6;
      let total = 0, selectors = 0;
      for (let s = low; s <= high; s++) {
        let bestMapping = Infinity, bestK = 0;
        for (let k = 0; k < 4; k++) {
          let e = block[s] - colours[k];
          e *= e;
          if (heavy && (s === 0 || s === 3)) e *= 5;
          if (e < bestMapping) { bestMapping = e; bestK = k; }
        }
        total += bestMapping;
        selectors |= bestK << (s * 2);
      }
      if (total < best) { best = total; solution = [lo, hi, selectors]; }
    }
  }
  ALPHA_TABLE.set(key, solution);
  return solution;
}

// A BC7 block being written: 128 bits as four 32-bit words, LSB first.
const BLK = new Int32Array(5);
/** `bits` (at most 31) of `value` at bit `at`, replacing what was there. */
function setBits(value, bits, at) {
  const w = at >> 5, s = at & 31, mask = bits === 32 ? -1 : (1 << bits) - 1;
  BLK[w] = (BLK[w] & ~(mask << s)) | ((value & mask) << s);
  if (s + bits > 32) {
    const n = 32 - s;
    BLK[w + 1] = (BLK[w + 1] & ~(mask >>> n)) | ((value & mask) >>> n);
  }
}
function getBits(bits, at) {
  const w = at >> 5, s = at & 31;
  let v = BLK[w] >>> s;
  if (s + bits > 32) v |= BLK[w + 1] << (32 - s);
  return (v & ((1 << bits) - 1)) >>> 0;
}
function loadBlock(out, o) {
  for (let w = 0; w < 4; w++) BLK[w] = out[o + w * 4] | (out[o + w * 4 + 1] << 8) | (out[o + w * 4 + 2] << 16) | (out[o + w * 4 + 3] << 24);
}
function storeBlock(out, o) {
  for (let k = 0; k < 16; k++) out[o + k] = BLK[k >> 2] >>> ((k & 3) * 8);
}
// Mode 5's fields: R0 R1 G0 G1 B0 B1 from bit 8, 7 bits each; A0 at 50 and A1 at 58, 8 bits each.
function setColour(r0, g0, b0, r1, g1, b1) {
  setBits(r0, 7, 8); setBits(r1, 7, 15); setBits(g0, 7, 22); setBits(g1, 7, 29); setBits(b0, 7, 36); setBits(b1, 7, 43);
}
function setAlpha(a0, a1) { setBits(a0, 8, 50); setBits(a1, 8, 58); }

/** convert_etc1s_to_bc7_m5_color: the block into BLK, opaque. */
function bc7Colour(global, flags, e, sel) {
  BLK.fill(0);
  setBits(1 << 5, 6, 0);
  setAlpha(255, 255);
  const { colour, intensity } = global.endpoints;
  const r5 = colour[e * 3], g5 = colour[e * 3 + 1], b5 = colour[e * 3 + 2], inten = intensity[e];
  const low = flags.lo[sel], high = flags.hi[sel], rows = global.selectors, row0 = sel * 4;
  const table = INTENSITY[inten];
  const colourOf = (c5, s) => clamp255(expand5(c5) + table[s]);
  const selectorAt = (x, y) => (rows[row0 + y] >> (2 * x)) & 3;
  if (flags.unique[sel] === 1) {
    const q = equals1();
    const r = colourOf(r5, low), g = colourOf(g5, low), b = colourOf(b5, low);
    setColour(q[r * 2], q[g * 2], q[b * 2], q[r * 2 + 1], q[g * 2 + 1], q[b * 2 + 1]);
    setBits(0x2aaaaaab, 31, 66);
    return;
  }
  if (flags.unique[sel] === 2) {
    const c0 = [colourOf(r5, low), colourOf(g5, low), colourOf(b5, low)];
    const c1 = [colourOf(r5, high), colourOf(g5, high), colourOf(b5, high)];
    setColour(c0[0] >> 1, c0[1] >> 1, c0[2] >> 1, c1[0] >> 1, c1[1] >> 1, c1[2] >> 1);
    let outLow = 0, offset = 0, bits = 0;
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        let os = selectorAt(x, y) === low ? outLow : 3 ^ outLow;
        let n = 2;
        if ((x | y) === 0) {
          if (os & 2) {
            setColour(c1[0] >> 1, c1[1] >> 1, c1[2] >> 1, c0[0] >> 1, c0[1] >> 1, c0[2] >> 1);
            outLow = 3;
            os = 0;
          }
          n = 1;
        }
        bits |= os << offset;
        offset += n;
      }
    }
    setBits(bits >>> 0, 31, 66);
    return;
  }
  const range = RANGE_INDEX[low][high];
  const tr = colourSolutions(inten, r5, range), tg = colourSolutions(inten, g5, range), tb = colourSolutions(inten, b5, range);
  let best = Infinity, mapping = 0;
  for (let m = 0; m < 10; m++) {
    const err = tr[m * 3 + 2] + tg[m * 3 + 2] + tb[m * 3 + 2];
    if (err < best) { best = err; mapping = m; }
  }
  const xlat = MAPPINGS[mapping], o = mapping * 3;
  let inv = 0;
  if (xlat[selectorAt(0, 0)] & 2) {
    setColour(tr[o + 1], tg[o + 1], tb[o + 1], tr[o], tg[o], tb[o]);
    inv = 3;
  } else setColour(tr[o], tg[o], tb[o], tr[o + 1], tg[o + 1], tb[o + 1]);
  let bits = 0, offset = 0;
  for (let y = 0; y < 4; y++) {
    for (let x = 0; x < 4; x++) {
      bits |= (xlat[selectorAt(x, y)] ^ inv) << offset;
      offset += (x | y) === 0 ? 1 : 2;
    }
  }
  setBits(bits >>> 0, 31, 66);
}

/** convert_etc1s_to_bc7_m5_alpha: alpha from an alpha slice's block, into BLK. */
function bc7Alpha(global, flags, e, sel) {
  const { colour, intensity } = global.endpoints;
  const r5 = colour[e * 3], g5 = colour[e * 3 + 1], inten = intensity[e];
  const low = flags.lo[sel], high = flags.hi[sel], rows = global.selectors;
  const selectorAt = (x, y) => (rows[sel * 4 + y] >> (2 * x)) & 3;
  const table = INTENSITY[inten];
  if (flags.unique[sel] === 1) {
    const r = clamp255(expand5(r5) + table[low]);
    setAlpha(r, r);
    return;
  }
  if (flags.unique[sel] === 2) {
    const g = table.map((m) => clamp255(expand5(g5) + m));
    setAlpha(g[low], g[high]);
    let outLow = 0, offset = 0, bits = 0;
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        let os = selectorAt(x, y) === low ? outLow : 3 ^ outLow;
        let n = 2;
        if ((x | y) === 0) {
          if (os & 2) { setAlpha(g[high], g[low]); outLow = 3; os = 0; }
          n = 1;
        }
        bits |= os << offset;
        offset += n;
      }
    }
    setBits(bits >>> 0, 31, 97);
    return;
  }
  const [lo, hi, trans] = alphaSolution(inten, r5, RANGE_INDEX[low][high]);
  setAlpha(lo, hi);
  let translation = trans, offset = 0, bits = 0;
  for (let y = 0; y < 4; y++) {
    for (let x = 0; x < 4; x++) {
      let os = (translation >> (selectorAt(x, y) * 2)) & 3;
      let n = 2;
      if ((x | y) === 0) {
        if (os & 2) { setAlpha(hi, lo); translation ^= 0xff; os ^= 3; }
        n = 1;
      }
      bits |= os << offset;
      offset += n;
    }
  }
  setBits(bits >>> 0, 31, 97);
}

// ---- the real-time mode 5 encoder the chroma filter re-encodes with (bc7_mode_5_encoder)

let MIDPOINTS = null;
function midpoints() {
  if (MIDPOINTS) return MIDPOINTS;
  MIDPOINTS = new Float32Array(128);
  for (let i = 0; i < 128; i++) {
    let vl = i << 1; vl |= vl >> 7;
    let vh = Math.min(127, i + 1) << 1; vh |= vh >> 7;
    MIDPOINTS[i] = i === 127 ? 1e15 : f(f(f(vl / 255) + f(vh / 255)) / 2);
  }
  return MIDPOINTS;
}
function toSeven(c) {
  const vl = Math.trunc(f(c * 127));
  return Math.min(127, Math.max(0, vl + (c > MIDPOINTS[vl] ? 1 : 0)));
}
const toSevenFrom8 = (c8) => toSeven(f(c8 * f(1 / 255)));
const from7 = (v) => (v << 1) | (v >> 6);
const W2 = [0, 21, 43, 64];
const interp2 = (l, h, w) => (l * (64 - W2[w]) + h * W2[w] + 32) >> 6;
const mul8 = (a, b) => { const t = a * b + 128; return (t + (t >> 8)) >> 8; };
const lerp8 = (a, b, s) => a + mul8(b - a, s);

function evalWeights(px, weights, lr, lg, lb, hr, hg, hb) {
  lr = from7(lr); lg = from7(lg); lb = from7(lb); hr = from7(hr); hg = from7(hg); hb = from7(hb);
  // The ends' interpolation is the ends themselves: weight 0 and 64.
  let ar = hr - lr, ag = hg - lg, ab = hb - lb;
  const dot = (w) => interp2(lr, hr, w) * ar + interp2(lg, hg, w) * ag + interp2(lb, hb, w) * ab;
  const d0 = lr * ar + lg * ag + lb * ab, d1 = dot(1), d2 = dot(2), d3 = hr * ar + hg * ag + hb * ab;
  const t0 = d0 + d1, t1 = d1 + d2, t2 = d2 + d3;
  ar *= 2; ag *= 2; ab *= 2;
  for (let i = 0; i < 16; i++) {
    const d = px[i * 3] * ar + px[i * 3 + 1] * ag + px[i * 3 + 2] * ab;
    weights[i] = (d > t0 ? 1 : 0) + (d >= t1 ? 1 : 0) + (d >= t2 ? 1 : 0);
  }
}

function packRGB(lr, lg, lb, hr, hg, hb, weights) {
  let inv = 0;
  if (weights[0] & 2) { [lr, hr] = [hr, lr]; [lg, hg] = [hg, lg]; [lb, hb] = [hb, lb]; inv = 3; }
  BLK.fill(0);
  setBits(32, 6, 0);
  setColour(lr, lg, lb, hr, hg, hb);
  setAlpha(255, 255);
  let offset = 66;
  for (let i = 0; i < 16; i++) { const n = i ? 2 : 1; setBits(inv ^ weights[i], n, offset); offset += n; }
}

const WEIGHT_VALS4 = [0x000009, 0x010204, 0x040201, 0x090000];
function leastSquares(px, sel, totals) {
  let accum = 0;
  const uq = [0, 0, 0];
  for (let i = 0; i < 16; i++) {
    accum += WEIGHT_VALS4[sel[i]];
    for (let c = 0; c < 3; c++) uq[c] += sel[i] * px[i * 3 + c];
  }
  const z00 = (accum >> 16) & 0xff, z10 = (accum >> 8) & 0xff, z11 = accum & 0xff, z01 = z10;
  let det = f(f(z00 * z11) - f(z01 * z10));
  if (Math.abs(det) < f(1e-8)) return null;
  det = f(f(3 / 255) / det);
  const iz00 = f(z11 * det), iz01 = f(-z01 * det), iz10 = f(-z10 * det), iz11 = f(z00 * det);
  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
  const lo = [], hi = [];
  for (let c = 0; c < 3; c++) {
    const q10 = totals[c] * 3 - uq[c];
    hi.push(toSeven(clamp01(f(f(iz00 * uq[c]) + f(iz01 * q10)))));
    lo.push(toSeven(clamp01(f(f(iz10 * uq[c]) + f(iz11 * q10)))));
  }
  return [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]];
}

/** encode_bc7_mode_5_block, not in its high-quality mode: 16 RGB texels, px[i * 3 + c], into BLK. */
const TOTALS = new Int32Array(3), MIN = new Int32Array(3), MAX = new Int32Array(3), WEIGHTS = new Uint8Array(16);
function encodeMode5(px) {
  midpoints();
  const totals = TOTALS, min = MIN, max = MAX;
  totals.fill(0); min.fill(255); max.fill(0);
  for (let i = 0; i < 16; i++) {
    for (let c = 0; c < 3; c++) {
      const v = px[i * 3 + c];
      totals[c] += v;
      if (v < min[c]) min[c] = v;
      if (v > max[c]) max[c] = v;
    }
  }
  const weights = WEIGHTS;
  if (min[0] === max[0] && min[1] === max[1] && min[2] === max[2]) {
    const q = equals1();
    weights.fill(1);
    packRGB(q[min[0] * 2], q[min[1] * 2], q[min[2] * 2], q[min[0] * 2 + 1], q[min[1] * 2 + 1], q[min[2] * 2 + 1], weights);
    return;
  }
  const meanR = (totals[0] + 8) >> 4, meanG = (totals[1] + 8) >> 4, meanB = (totals[2] + 8) >> 4;
  const icov = [0, 0, 0, 0, 0, 0];
  for (let i = 0; i < 16; i++) {
    const r = px[i * 3] - meanR, g = px[i * 3 + 1] - meanG, b = px[i * 3 + 2] - meanB;
    icov[0] += r * r; icov[1] += r * g; icov[2] += r * b; icov[3] += g * g; icov[4] += g * b; icov[5] += b * b;
  }
  const maxVar = Math.max(icov[0], icov[3], icov[5]);
  if (maxVar < 160) {
    const lo = [0, 1, 2].map((c) => toSevenFrom8(lerp8(min[c], max[c], 16)));
    const hi = [0, 1, 2].map((c) => toSevenFrom8(lerp8(min[c], max[c], 239)));
    evalWeights(px, weights, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]);
    packRGB(lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], weights);
    return;
  }
  const sc = f(1 / maxVar);
  const wx = f(sc * icov[0]), wy = f(sc * icov[3]), wz = f(sc * icov[5]);
  const altR = f(f(f(icov[0] * wx) + f(icov[1] * wy)) + f(icov[2] * wz));
  const altG = f(f(f(icov[1] * wx) + f(icov[3] * wy)) + f(icov[4] * wz));
  const altB = f(f(f(icov[2] * wx) + f(icov[4] * wy)) + f(icov[5] * wz));
  let sr = 306, sg = 601, sb = 117;
  const k = Math.max(Math.abs(altR), Math.abs(altG), Math.abs(altB));
  if (k >= f(1e-10)) {
    const m = f(2048 / k);
    sr = Math.trunc(f(altR * m)); sg = Math.trunc(f(altG * m)); sb = Math.trunc(f(altB * m));
  }
  sr = (sr << 4) | 0; sg = (sg << 4) | 0; sb = (sb << 4) | 0;
  let lowDot = 0x7fffffff, highDot = -0x80000000;
  for (let i = 0; i < 16; i++) {
    const d = (((px[i * 3] * sr + px[i * 3 + 1] * sg + px[i * 3 + 2] * sb) | 0) & ~0xf) + i;
    if (d < lowDot) lowDot = d;
    if (d > highDot) highDot = d;
  }
  const lc = lowDot & 15, hc = highDot & 15;
  let e = [toSevenFrom8(px[lc * 3]), toSevenFrom8(px[lc * 3 + 1]), toSevenFrom8(px[lc * 3 + 2]),
    toSevenFrom8(px[hc * 3]), toSevenFrom8(px[hc * 3 + 1]), toSevenFrom8(px[hc * 3 + 2])];
  evalWeights(px, weights, ...e);
  const better = leastSquares(px, weights, totals);
  if (better) { e = better; evalWeights(px, weights, ...e); }
  packRGB(...e, weights);
}

// ---- the chroma filter (chroma_filter_bc7_mode5), on by default in Basis

/** Each endpoint's chroma, Co then Cg (exact: quarters of whole numbers), once a file. */
function chromaOf(global) {
  if (global.chroma) return global.chroma;
  const colour = global.endpoints.colour, n = global.endpointCount, out = new Float64Array(n * 2);
  for (let e = 0; e < n; e++) {
    const r = expand5(colour[e * 3]), g = expand5(colour[e * 3 + 1]), b = expand5(colour[e * 3 + 2]);
    out[e * 2] = r * 0.5 - b * 0.5;
    out[e * 2 + 1] = -r * 0.25 + g * 0.5 - b * 0.25;
  }
  return (global.chroma = out);
}

function chromaFilter(global, endpointsOf, out, across, down) {
  const chroma = chromaOf(global);
  const at = (x, y) => endpointsOf[Math.min(down - 1, Math.max(0, y)) * across + Math.min(across - 1, Math.max(0, x))] * 2;
  const lerp = (p, q, s) => p + (q - p) * s;
  const px = new Uint8Array(48), yVals = new Float32Array(16);
  for (let by = 0; by < down; by++) {
    for (let bx = 0; bx < across; bx++) {
      const centre = endpointsOf[by * across + bx] * 2, co = chroma[centre], cg = chroma[centre + 1];
      let filter = false;
      for (let dy = -1; dy <= 1 && !filter; dy++) {
        const oy = by + dy;
        if (oy < 0 || oy >= down) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const ox = bx + dx;
          if ((dx | dy) === 0 || ox < 0 || ox >= across) continue;
          const near = endpointsOf[oy * across + ox] * 2;
          if (Math.abs(chroma[near] - co) > 10 || Math.abs(chroma[near + 1] - cg) > 10) { filter = true; break; }
        }
      }
      if (!filter) continue;
      const o = (by * across + bx) * 16;
      loadBlock(out, o);
      const lr = from7(getBits(7, 8)), hr = from7(getBits(7, 15)), lg = from7(getBits(7, 22));
      const hg = from7(getBits(7, 29)), lb = from7(getBits(7, 36)), hb = from7(getBits(7, 43));
      const y4 = [0, 1, 2, 3].map((i) => interp2(lr, hr, i) * 0.25 + interp2(lg, hg, i) * 0.5 + interp2(lb, hb, i) * 0.25);
      let sum = 0, sumSq = 0, offset = 66;
      for (let i = 0; i < 16; i++) {
        const n = i ? 2 : 1, y = y4[getBits(n, offset)];
        offset += n;
        yVals[i] = y;
        sum = f(sum + y);
        sumSq = f(sumSq + f(y * y));
      }
      const S = 1 / 16, scaled = f(sum * S);
      if (f(f(sumSq * S) - f(scaled * scaled)) < 3) continue;
      for (let py = 0; py < 4; py++) {
        const uby = by + ((py - 2) >> 2), fy = (((py + 2) & 3) + 0.5) * 0.25;
        for (let pxi = 0; pxi < 4; pxi++) {
          const ubx = bx + ((pxi - 2) >> 2), fx = (((pxi + 2) & 3) + 0.5) * 0.25;
          const a = at(ubx, uby), b = at(ubx + 1, uby), c = at(ubx, uby + 1), d = at(ubx + 1, uby + 1);
          const fco = lerp(lerp(chroma[a], chroma[b], fx), lerp(chroma[c], chroma[d], fx), fy);
          const fcg = lerp(lerp(chroma[a + 1], chroma[b + 1], fx), lerp(chroma[c + 1], chroma[d + 1], fx), fy);
          const y = yVals[py * 4 + pxi], t = (py * 4 + pxi) * 3;
          const clamp = (v) => Math.trunc(0.5 + (v < 0 ? 0 : v > 255 ? 255 : v));
          px[t] = clamp(y + fco - fcg); px[t + 1] = clamp(y + fcg); px[t + 2] = clamp(y - fco - fcg);
        }
      }
      encodeMode5(px);
      storeBlock(out, o);
    }
  }
}

/**
 * A level's decoded slices as BC7 (mode 5) blocks, as Basis Universal's
 * transcoder writes them. `filter` false leaves out the chroma filter, as
 * Basis's cDecodeFlagsNoETC1SChromaFiltering does.
 */
export function etc1sToBC7(global, slice, alpha, width, height, { filter = true } = {}) {
  const across = Math.ceil(width / 4), down = Math.ceil(height / 4);
  const flags = selectorFlags(global);
  const out = new Uint8Array(across * down * 16);
  for (let b = 0; b < across * down; b++) {
    bc7Colour(global, flags, slice.endpoints[b], slice.selectors[b]);
    storeBlock(out, b * 16);
  }
  if (filter) chromaFilter(global, slice.endpoints, out, across, down);
  if (alpha) {
    for (let b = 0; b < across * down; b++) {
      loadBlock(out, b * 16);
      bc7Alpha(global, flags, alpha.endpoints[b], alpha.selectors[b]);
      storeBlock(out, b * 16);
    }
  }
  return out;
}
