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
