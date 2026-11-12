// UASTC LDR 4x4 (Basis Universal), transcoded: each 16-byte block unpacked to
// its endpoints and weights, then written out as RGBA8 texels, a BC7 block, or
// an ASTC 4x4 block (the last losing nothing; each held to Basis's own output).
//
// Written from Binomial's "UASTC LDR 4x4 Texture Specification", which is in
// the public domain (CC0): its mode layouts, BISE ranges, dequantization,
// partition and anchor tables, weight tables, and its BC7 conversion -- the
// p-bit search and the solid-colour tables included -- so that the BC7 this
// writes is the BC7 Basis Universal's own transcoder writes, block for block
// (test/uastc.test.js holds it to that). BC7's own partition and anchor tables
// are the format's, checked here against every pattern UASTC uses.
//
// A block: the mode's Huffman code from bit 0, then hint bits a transcoder to
// BC7 or RGBA does not need, then the partition pattern or component selector,
// the endpoints' trits or quints, the endpoints' bits, and the weights, each
// read LSB first.

/** Each mode's Huffman code length, by the low 7 bits of a block's first byte, and the mode. */
const HUFF_MODES = [11, 0, 10, 3, 11, 15, 12, 7, 11, 18, 10, 5, 11, 14, 12, 9, 11, 0, 10, 4, 11, 16, 12, 8, 11, 18, 10, 6, 11, 2, 12, 13,
  11, 0, 10, 3, 11, 17, 12, 7, 11, 18, 10, 5, 11, 14, 12, 9, 11, 0, 10, 4, 11, 1, 12, 8, 11, 18, 10, 6, 11, 2, 12, 13,
  11, 0, 10, 3, 11, 19, 12, 7, 11, 18, 10, 5, 11, 14, 12, 9, 11, 0, 10, 4, 11, 16, 12, 8, 11, 18, 10, 6, 11, 2, 12, 13,
  11, 0, 10, 3, 11, 17, 12, 7, 11, 18, 10, 5, 11, 14, 12, 9, 11, 0, 10, 4, 11, 1, 12, 8, 11, 18, 10, 6, 11, 2, 12, 13];
const CODE_BITS = [4, 6, 5, 5, 5, 5, 5, 5, 5, 5, 3, 2, 3, 5, 5, 7, 6, 6, 4, 7];

// Per mode (8, the solid colour, apart): where the pattern, selector or
// endpoints begin, past the hints; then the spec's description tables.
const START = [19, 21, 20, 20, 20, 20, 20, 20, 0, 28, 20, 19, 20, 28, 28, 30, 29, 29, 19];
const WEIGHT_BITS = [4, 2, 3, 2, 2, 3, 2, 2, 0, 2, 4, 2, 3, 1, 2, 4, 2, 2, 5];
const ENDPOINT_RANGE = [19, 20, 8, 7, 12, 20, 18, 12, 0, 8, 13, 13, 19, 20, 20, 20, 20, 20, 11];
const SUBSETS = [1, 1, 2, 3, 2, 1, 1, 2, 0, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1];
const PLANES = [1, 1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1];
const COMPS = [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 3];
const PATTERN_BITS = [0, 0, 5, 4, 5, 0, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 5, 0, 0];
const HAS_SELECTOR = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0];
const SOLID = 8;

/** BISE ranges UASTC uses: [bits, trits, quints]. */
const RANGES = { 7: [2, 1, 0], 8: [4, 0, 0], 11: [5, 0, 0], 12: [3, 0, 1], 13: [4, 1, 0], 18: [5, 0, 1], 19: [6, 1, 0], 20: [8, 0, 0] };
const RANGE_BITS = new Uint8Array(21), RANGE_TRITS = new Uint8Array(21), RANGE_QUINTS = new Uint8Array(21);
for (const [range, [b, t, q]] of Object.entries(RANGES)) { RANGE_BITS[range] = b; RANGE_TRITS[range] = t; RANGE_QUINTS[range] = q; }

// Endpoint dequantization, by BISE integer, to [0, 255]: the spec's decoding
// tables for the trit and quint ranges; bit replication for the binary ones.
const DEQUANT = {
  7: [0, 255, 69, 186, 23, 232, 92, 163, 46, 209, 116, 139],
  12: [0, 255, 32, 223, 65, 190, 97, 158, 6, 249, 39, 216, 71, 184, 104, 151, 13, 242, 45, 210, 78, 177, 110, 145, 19, 236, 52, 203, 84, 171, 117, 138, 26, 229, 58, 197, 91, 164, 123, 132],
  13: [0, 255, 16, 239, 32, 223, 48, 207, 65, 190, 81, 174, 97, 158, 113, 142, 5, 250, 21, 234, 38, 217, 54, 201, 70, 185, 86, 169, 103, 152, 119, 136, 11, 244, 27, 228, 43, 212, 59, 196, 76, 179, 92, 163, 108, 147, 124, 131],
  18: [0, 255, 8, 247, 16, 239, 24, 231, 32, 223, 40, 215, 48, 207, 56, 199, 64, 191, 72, 183, 80, 175, 88, 167, 96, 159, 104, 151, 112, 143, 120, 135,
    1, 254, 9, 246, 17, 238, 25, 230, 33, 222, 41, 214, 49, 206, 57, 198, 65, 190, 73, 182, 81, 174, 89, 166, 97, 158, 105, 150, 113, 142, 121, 134,
    3, 252, 11, 244, 19, 236, 27, 228, 35, 220, 43, 212, 51, 204, 59, 196, 67, 188, 75, 180, 83, 172, 91, 164, 99, 156, 107, 148, 115, 140, 123, 132,
    4, 251, 12, 243, 20, 235, 28, 227, 36, 219, 44, 211, 52, 203, 60, 195, 68, 187, 76, 179, 84, 171, 92, 163, 100, 155, 108, 147, 116, 139, 124, 131,
    6, 249, 14, 241, 22, 233, 30, 225, 38, 217, 46, 209, 54, 201, 62, 193, 70, 185, 78, 177, 86, 169, 94, 161, 102, 153, 110, 145, 118, 137, 126, 129],
  19: [0, 255, 4, 251, 8, 247, 12, 243, 16, 239, 20, 235, 24, 231, 28, 227, 32, 223, 36, 219, 40, 215, 44, 211, 48, 207, 52, 203, 56, 199, 60, 195,
    64, 191, 68, 187, 72, 183, 76, 179, 80, 175, 84, 171, 88, 167, 92, 163, 96, 159, 100, 155, 104, 151, 108, 147, 112, 143, 116, 139, 120, 135, 124, 131,
    1, 254, 5, 250, 9, 246, 13, 242, 17, 238, 21, 234, 25, 230, 29, 226, 33, 222, 37, 218, 41, 214, 45, 210, 49, 206, 53, 202, 57, 198, 61, 194,
    65, 190, 69, 186, 73, 182, 77, 178, 81, 174, 85, 170, 89, 166, 93, 162, 97, 158, 101, 154, 105, 150, 109, 146, 113, 142, 117, 138, 121, 134, 125, 130,
    2, 253, 6, 249, 10, 245, 14, 241, 18, 237, 22, 233, 26, 229, 30, 225, 34, 221, 38, 217, 42, 213, 46, 209, 50, 205, 54, 201, 58, 197, 62, 193,
    66, 189, 70, 185, 74, 181, 78, 177, 82, 173, 86, 169, 90, 165, 94, 161, 98, 157, 102, 153, 106, 149, 110, 145, 114, 141, 118, 137, 122, 133, 126, 129],
};
for (const [range, bits] of [[8, 4], [11, 5], [20, 8]]) {
  DEQUANT[range] = Array.from({ length: 1 << bits }, (_, v) => {
    let out = 0;
    for (let shift = 8 - bits; shift > -bits; shift -= bits) out |= shift >= 0 ? v << shift : v >> -shift;
    return out & 255;
  });
}

/** ASTC's weights, to [0, 64], by index bits. */
const WEIGHTS = [null, [0, 64], [0, 21, 43, 64], [0, 9, 18, 27, 37, 46, 55, 64],
  [0, 4, 8, 12, 17, 21, 25, 29, 35, 39, 43, 47, 52, 56, 60, 64],
  [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64]];

// Partition patterns in UASTC's (ASTC's) subset numbering, and each subset's
// anchor texel, whose weight is stored a bit short.
const unpack = (s) => s.split(',').map((row) => Array.from(row, Number));
const PATTERNS2 = unpack('0011001100110011,0001000100010001,1000100010001000,0001001100110111,1111111011101100,0011011101111111,'
  + '1110110010000000,1111111011001000,0000000000010011,1100100000000000,0000000101111111,1111111111101000,1110100000000000,'
  + '1111111100000000,0000111111111111,1111111111110000,1000111011111111,1111111101110001,0111001100010000,0011000100000000,'
  + '0000100011001110,1111111101110011,1000110011001110,0011000100010000,1111011101110011,0110011001100110,1111000000001111,'
  + '1010101010101010,1111000011110000,1001001101101100');
const ANCHORS2 = [[0, 2], [0, 3], [1, 0], [0, 3], [7, 0], [0, 2], [3, 0], [7, 0], [0, 11], [2, 0], [0, 7], [11, 0], [3, 0], [8, 0],
  [0, 4], [12, 0], [1, 0], [8, 0], [0, 1], [0, 2], [0, 4], [8, 0], [1, 0], [0, 2], [4, 0], [0, 1], [4, 0], [1, 0], [4, 0], [1, 0]];
const PATTERNS3 = unpack('0000000011221122,1111111100002222,1111000000002222,1111222200000000,1120112011201120,0112011201120112,'
  + '0211021102110211,2000200021112111,2012201220122012,1111000022221111,0022001100110022');
const ANCHORS3 = [[0, 8, 10], [8, 0, 12], [4, 0, 12], [8, 0, 4], [3, 0, 2], [0, 1, 3], [0, 2, 1], [1, 9, 0], [1, 2, 0], [4, 0, 8], [0, 6, 2]];
const PATTERNS7 = unpack('0000111100000000,0010001000100010,1100110010000000,0000000100110011,1111111100001111,0100010001000100,'
  + '0001001111111111,0111001100110011,1100000000111100,0111011100000000,0000000011101110,1100000000001100,0111001100000000,'
  + '0000000111111111,1111111111110110,1100110011001000,1111111110001000,0011011011001000,1111011100000000');
const ANCHORS7 = [[0, 4], [0, 2], [2, 0], [0, 7], [8, 0], [0, 1], [0, 3], [0, 1], [2, 0], [0, 1], [0, 8], [2, 0], [0, 1], [0, 7],
  [12, 0], [2, 0], [9, 0], [0, 2], [4, 0]];

// The block being unpacked, as four 32-bit words (and a fifth, zero, for a
// field read at the very end): loaded once, so each field is a shift or two.
const BLOCK = new Int32Array(5);
function loadBlock(bytes, at) {
  for (let w = 0; w < 4; w++) {
    const i = at + w * 4;
    BLOCK[w] = bytes[i] | (bytes[i + 1] << 8) | (bytes[i + 2] << 16) | (bytes[i + 3] << 24);
  }
}
/** `count` bits (at most 8) of the loaded block from bit `offset`, LSB first. */
function field(offset, count) {
  const w = offset >> 5, s = offset & 31;
  let v = BLOCK[w] >>> s;
  if (s + count > 32) v |= BLOCK[w + 1] << (32 - s);
  return v & ((1 << count) - 1);
}

// A group's trits (5 in 8 bits) or quints (3 in 7 bits), least significant
// first, by the group's value: the base-3 or base-5 digits UASTC stores.
const TRIT_DIGITS = new Uint8Array(256 * 5), QUINT_DIGITS = new Uint8Array(128 * 3);
for (let v = 0; v < 256; v++) for (let i = 0, x = v; i < 5; i++, x = Math.floor(x / 3)) TRIT_DIGITS[v * 5 + i] = x % 3;
for (let v = 0; v < 128; v++) for (let i = 0, x = v; i < 3; i++, x = Math.floor(x / 5)) QUINT_DIGITS[v * 3 + i] = x % 5;
const TRIT_GROUP_BITS = [0, 2, 4, 5, 7, 8];
const QUINT_GROUP_BITS = [0, 3, 5, 7];
const HIGH = new Uint8Array(24);

/** A fresh unpacked block, every field in place so the object keeps one shape. */
export const newBlock = () => ({
  mode: 0, pattern: 0, selector: 0, solid: new Uint8Array(4),
  endpoints: new Uint8Array(24), raw: new Uint8Array(24), weights: new Uint8Array(32),
});

/**
 * One UASTC block at bytes[at..at+16), unpacked: { mode, solid (RGBA) | endpoints
 * (dequantized, ASTC's order), raw (their BISE integers), weights (raw, planes
 * interleaved), pattern, selector }.
 */
export function unpackBlock(bytes, at, out = newBlock()) {
  const mode = HUFF_MODES[bytes[at] & 127];
  if (mode === 19) throw new Error('uastc: a block in the reserved mode 19');
  out.mode = mode;
  loadBlock(bytes, at);
  if (mode === SOLID) {
    const solid = out.solid;
    solid[0] = field(5, 8); solid[1] = field(13, 8);
    solid[2] = field(21, 8); solid[3] = field(29, 8);
    return out;
  }
  let bit = START[mode];
  const patternBits = PATTERN_BITS[mode];
  out.pattern = patternBits ? field(bit, patternBits) : 0;
  bit += patternBits;
  out.selector = mode === 17 ? 3 : HAS_SELECTOR[mode] ? field(bit, 2) : 0;
  if (HAS_SELECTOR[mode]) bit += 2;

  const range = ENDPOINT_RANGE[mode];
  const bits = RANGE_BITS[range], trits = RANGE_TRITS[range], quints = RANGE_QUINTS[range];
  const count = COMPS[mode] * 2 * SUBSETS[mode];
  if (trits) {
    for (let i = 0; i < count; i += 5) {
      const n = Math.min(5, count - i), v = field(bit, TRIT_GROUP_BITS[n]);
      bit += TRIT_GROUP_BITS[n];
      for (let k = 0; k < n; k++) HIGH[i + k] = TRIT_DIGITS[v * 5 + k];
    }
  } else if (quints) {
    for (let i = 0; i < count; i += 3) {
      const n = Math.min(3, count - i), v = field(bit, QUINT_GROUP_BITS[n]);
      bit += QUINT_GROUP_BITS[n];
      for (let k = 0; k < n; k++) HIGH[i + k] = QUINT_DIGITS[v * 3 + k];
    }
  }
  const dequant = DEQUANT[range], endpoints = out.endpoints, raw = out.raw;
  for (let i = 0; i < count; i++) {
    const low = field(bit, bits);
    bit += bits;
    const v = trits || quints ? (HIGH[i] << bits) | low : low;
    raw[i] = v;
    endpoints[i] = dequant[v];
  }

  // Weights: each subset's anchor texel a bit short (both planes' in a dual-plane mode).
  const wbits = WEIGHT_BITS[mode], weights = out.weights;
  const anchors = anchorMask(mode, out.pattern);
  if (PLANES[mode] === 1) {
    for (let t = 0; t < 16; t++) {
      const n = wbits - ((anchors >> t) & 1);
      weights[t] = field(bit, n);
      bit += n;
    }
  } else {
    for (let t = 0; t < 16; t++) {
      const n = wbits - ((anchors >> t) & 1);
      weights[t * 2] = field(bit, n);
      weights[t * 2 + 1] = field(bit + n, n);
      bit += 2 * n;
    }
  }
  return out;
}

/** Each subset's anchor texel, as a bit per texel. */
const mask = (list) => list.map((anchors) => anchors.reduce((m, t) => m | (1 << t), 0));
const ANCHOR_MASK2 = mask(ANCHORS2), ANCHOR_MASK3 = mask(ANCHORS3), ANCHOR_MASK7 = mask(ANCHORS7);
function anchorMask(mode, pattern) {
  if (SUBSETS[mode] === 1) return 1;
  if (mode === 3) return ANCHOR_MASK3[pattern];
  if (mode === 7) return ANCHOR_MASK7[pattern];
  return ANCHOR_MASK2[pattern];
}

/** ASTC's interpolation of 8-bit endpoints at a [0, 64] weight, decoding as linear (the spec's srgb = false). */
function interpolate(l, h, w) {
  l = (l << 8) | l;
  h = (h << 8) | h;
  return ((l * (64 - w) + h * w + 32) >> 6) >> 8;
}

const VALUE = new Uint8Array(4);

/** An unpacked block's 16 texels as RGBA8, into out at `at` with `stride` bytes a row. */
export function decodeBlock(block, out, at, stride) {
  const { mode } = block;
  if (mode === SOLID) {
    for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) out.set(block.solid, at + y * stride + x * 4);
    return;
  }
  const comps = COMPS[mode], planes = PLANES[mode], table = WEIGHTS[WEIGHT_BITS[mode]];
  const e = block.endpoints, w = block.weights, value = VALUE;
  const subsets = SUBSETS[mode], map = subsets === 1 ? null
    : mode === 3 ? PATTERNS3[block.pattern] : mode === 7 ? PATTERNS7[block.pattern] : PATTERNS2[block.pattern];
  // The component the second plane weights, if there is one.
  const second = planes === 2 ? (comps === 2 && block.selector === 3 ? 1 : block.selector) : -1;
  for (let t = 0; t < 16; t++) {
    const base = map === null ? 0 : map[t] * comps * 2;
    for (let c = 0; c < comps; c++) {
      value[c] = interpolate(e[base + c * 2], e[base + c * 2 + 1], table[w[t * planes + (c === second ? 1 : 0)]]);
    }
    const o = at + (t >> 2) * stride + (t & 3) * 4;
    if (comps === 2) { out[o] = out[o + 1] = out[o + 2] = value[0]; out[o + 3] = value[1]; }
    else { out[o] = value[0]; out[o + 1] = value[1]; out[o + 2] = value[2]; out[o + 3] = comps === 4 ? value[3] : 255; }
  }
}

/** A UASTC level (16 bytes a 4x4 block) as RGBA8 texels, `width` x `height`. */
export function uastcToRGBA(data, width, height) {
  const bw = Math.ceil(width / 4), bh = Math.ceil(height / 4);
  if (data.length < bw * bh * 16) throw new Error(`uastc: ${data.length} bytes for ${bw * bh} blocks`);
  // Whole blocks first, then the texels inside the image kept.
  const padded = new Uint8Array(bw * 4 * bh * 4 * 4), stride = bw * 16;
  const block = newBlock();
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      unpackBlock(data, (by * bw + bx) * 16, block);
      decodeBlock(block, padded, by * 4 * stride + bx * 16, stride);
    }
  }
  if (bw * 4 === width && bh * 4 === height) return padded;
  const out = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) out.set(padded.subarray(y * stride, y * stride + width * 4), y * width * 4);
  return out;
}

// ---- BC7 ----------------------------------------------------------------

// BC7's own partition tables (texel subsets, 64 patterns each) and anchor texels.
const BC7_PARTITIONS2 = unpack(
  '0011001100110011,0001000100010001,0111011101110111,0001001100110111,0000000100010011,0011011101111111,0001001101111111,' +
  '0000000100110111,0000000000010011,0011011111111111,0000000101111111,0000000000010111,0001011111111111,0000000011111111,' +
  '0000111111111111,0000000000001111,0000100011101111,0111000100000000,0000000010001110,0111001100010000,0011000100000000,' +
  '0000100011001110,0000000010001100,0111001100110001,0011000100010000,0000100010001100,0110011001100110,0011011001101100,' +
  '0001011111101000,0000111111110000,0111000110001110,0011100110011100,0101010101010101,0000111100001111,0101101001011010,' +
  '0011001111001100,0011110000111100,0101010110101010,0110100101101001,0101101010100101,0111001111001110,0001001111001000,' +
  '0011001001001100,0011101111011100,0110100110010110,0011110011000011,0110011010011001,0000011001100000,0100111001000000,' +
  '0010011100100000,0000001001110010,0000010011100100,0110110010010011,0011011011001001,0110001110011100,0011100111000110,' +
  '0110110011001001,0110001100111001,0111111010000001,0001100011100111,0000111100110011,0011001111110000,0010001011101110,' +
  '0100010001110111');
const BC7_PARTITIONS3 = unpack(
  '0011001102212222,0001001122112221,0000200122112211,0222002200110111,0000000011221122,0011001100220022,0022002211111111,' +
  '0011001122112211,0000000011112222,0000111111112222,0000111122222222,0012001200120012,0112011201120112,0122012201220122,' +
  '0011011211221222,0011200122002220,0001001101121122,0111001120012200,0000112211221122,0022002200221111,0111011102220222,' +
  '0001000122212221,0000001101220122,0000110022102210,0122012200110000,0012001211222222,0110122112210110,0000011012211221,' +
  '0022110211020022,0110011020022222,0011012201220011,0000200022112221,0000000211221222,0222002200120011,0011001200220222,' +
  '0120012001200120,0000111122220000,0120120120120120,0120201212010120,0011220011220011,0011112222000011,0101010122222222,' +
  '0000000021212121,0022112200221122,0022001100220011,0220122102201221,0101222222220101,0000212121212121,0101010101012222,' +
  '0222011102220111,0002111200021112,0000211221122112,0222011101110222,0002111211120002,0110011001102222,0000000021122112,' +
  '0110011022222222,0022001100110022,0022112211220022,0000000000002112,0002000100020001,0222122202221222,0101222222222222,' +
  '0111201122012220');
const BC7_ANCHOR2 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,2,8,2,2,8,8,15,2,8,2,2,8,8,2,2,15,15,6,8,2,8,15,15,2,8,2,2,2,15,15,6,6,2,6,8,15,15,2,2,15,15,15,15,15,2,2,15];
const BC7_ANCHOR3A = [3,3,15,15,8,3,15,15,8,8,6,6,6,5,3,3,3,3,8,15,3,3,6,10,5,8,8,6,8,5,15,15,8,15,3,5,6,10,8,15,15,3,15,5,15,15,15,15,3,15,5,5,5,8,5,10,5,10,8,13,15,12,3,3];
const BC7_ANCHOR3B = [15,8,8,3,15,15,3,8,15,15,15,15,15,15,15,8,15,8,15,3,15,8,15,8,3,15,6,10,15,15,10,8,15,3,15,10,10,8,9,10,6,15,8,15,3,6,6,8,15,3,15,15,15,15,15,15,15,15,15,15,3,15,15,8];


// UASTC's patterns as BC7's: the BC7 pattern, and how UASTC's subsets map onto
// it -- inverted (2-subset), permuted (3-subset: BC7 subset = perm[UASTC's]), or
// for mode 7, two of a 3-subset BC7 pattern's subsets given the same endpoints.
const COMMON2 = [[0, 0], [1, 0], [2, 1], [3, 0], [4, 1], [5, 0], [6, 1], [7, 1], [8, 0], [9, 1], [10, 0], [11, 1], [12, 1], [13, 1],
  [14, 0], [15, 1], [17, 1], [18, 1], [19, 0], [20, 0], [21, 0], [22, 1], [23, 1], [24, 0], [25, 1], [26, 0], [29, 1], [32, 1], [33, 1], [52, 1]];
const COMMON3 = [[4, 0], [8, 5], [9, 5], [10, 2], [11, 2], [12, 0], [13, 4], [20, 1], [35, 1], [36, 5], [57, 0]];
const PERMS = [[0, 1, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0], [0, 2, 1], [1, 0, 2]];
const COMMON7 = [[10, 4], [11, 4], [0, 3], [2, 4], [8, 5], [13, 4], [1, 2], [33, 2], [40, 3], [20, 4], [21, 0], [58, 3], [3, 0],
  [32, 2], [59, 1], [34, 3], [20, 1], [14, 4], [31, 3]];
/** A BC7 3-subset index to the UASTC 2-subset one, for mode 7 (the spec's convert_subset_index_3_to_2). */
function threeToTwo(p, k) {
  const half = k >> 1;
  let q = half === 0 ? (p <= 1 ? 0 : 1) : half === 1 ? (p === 0 ? 0 : 1) : (p === 0 || p === 2 ? 0 : 1);
  return k & 1 ? 1 - q : q;
}

/** Each UASTC mode's BC7 mode. */
const BC7_MODE = [6, 3, 1, 2, 3, 6, 5, 2, 0, 7, 6, 5, 6, 5, 6, 6, 7, 5, 6];
// UASTC weight indices to BC7's where their bits differ.
const TO_BC7_WEIGHTS = { '1>2': [0, 3], '2>4': [0, 5, 10, 15], '3>4': [0, 2, 4, 6, 9, 11, 13, 15],
  '5>4': [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7, 8, 9, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15] };

const f = Math.fround;

// determine_unique_pbits and determine_shared_pbits, the spec's p-bit
// searches, sum per component a term that depends only on that component's
// 8-bit value, the p-bit and the bit depth. So each term is worked out once,
// in single precision as the spec writes it (Math.fround), and a block adds
// them up in the same order: the same sums, so the same ties as Basis's.
const PBIT_TABLES = [];
/** For `compBits` (before the p-bit), by p * 256 + value: the quantized value, and the unique and shared error terms. */
function pbitTables(compBits) {
  if (PBIT_TABLES[compBits]) return PBIT_TABLES[compBits];
  const totalBits = compBits + 1, iscalep = (1 << totalBits) - 1, scalep = f(iscalep);
  const q = new Uint8Array(512), unique = new Float32Array(512), shared = new Float32Array(512);
  for (let p = 0; p < 2; p++) {
    for (let v = 0; v < 256; v++) {
      const x = f(v / 255);
      const m = Math.min(Math.max(Math.trunc(f(f(f(f(x * scalep) - p) / 2) + 0.5)) * 2 + p, p), iscalep - 1 + p);
      let expanded = (m << (8 - totalBits)) & 255;
      expanded |= expanded >> totalBits;
      const du = f(expanded - f(x * 255)), ds = f(f(expanded / 255) - x);
      q[p * 256 + v] = m >> 1;
      unique[p * 256 + v] = f(du * du);
      shared[p * 256 + v] = f(ds * ds);
    }
  }
  return (PBIT_TABLES[compBits] = { q, unique, shared });
}

// Solid colours: for each 8-bit value, the BC7 endpoints whose interpolation at
// a fixed index lands closest -- mode 6 (index 5, either p-bit) and mode 5
// (index 1) -- with the error, as the spec's solid-block conversion reads them.
const SOLID6 = [[], []], SOLID5 = [];
function solidTables() {
  if (SOLID5.length) return;
  for (let c = 0; c < 256; c++) {
    for (let p = 0; p < 2; p++) {
      let best = { error: 65535, lo: 0, hi: 0 };
      for (let l = 0; l < 128; l++) {
        const low = (l << 1) | p;
        for (let h = 0; h < 128; h++) {
          const high = (h << 1) | p;
          const k = (low * (64 - 21) + high * 21 + 32) >> 6;
          const error = (k - c) * (k - c);
          if (error < best.error) best = { error, lo: l, hi: h };
        }
      }
      SOLID6[p][c] = best;
    }
    let best = { error: 65535, lo: 0, hi: 0 };
    for (let l = 0; l < 128; l++) {
      const low = (l << 1) | (l >> 6);
      for (let h = 0; h < 128; h++) {
        const high = (h << 1) | (h >> 6);
        const k = (low * (64 - 21) + high * 21 + 32) >> 6;
        const error = (k - c) * (k - c);
        if (error < best.error) best = { error, lo: l, hi: h };
      }
    }
    SOLID5[c] = best;
  }
}

// A block's bits, written LSB first into four 32-bit words, then stored as
// its 16 bytes: module state rather than an object a block, since this runs
// for every block of a texture.
const ACC = new Int32Array(5);
let BIT = 0;
function begin() { ACC[0] = ACC[1] = ACC[2] = ACC[3] = 0; BIT = 0; }
/** At most 8 bits. */
function put(value, bits) {
  const w = BIT >> 5, s = BIT & 31;
  ACC[w] |= value << s;
  if (s + bits > 32) ACC[w + 1] |= value >>> (32 - s);
  BIT += bits;
}
function store(out, at) {
  for (let k = 0; k < 16; k++) out[at + k] = ACC[k >> 2] >>> ((k & 3) * 8);
}

// What each UASTC mode and pattern becomes in BC7, worked out once: the BC7
// pattern and subset count, which UASTC subset each BC7 subset takes its
// endpoints from, each texel's BC7 subset, and each subset's anchor texel.
const plan = (pattern, subsets, from) => {
  const map = subsets === 1 ? new Uint8Array(16) : Uint8Array.from(subsets === 2 ? BC7_PARTITIONS2[pattern] : BC7_PARTITIONS3[pattern]);
  const anchors = subsets === 1 ? [0] : subsets === 2 ? [0, BC7_ANCHOR2[pattern]] : [0, BC7_ANCHOR3A[pattern], BC7_ANCHOR3B[pattern]];
  return { pattern, subsets, from: Uint8Array.from(from), map, anchors: Uint8Array.from(anchors), mask: anchors.reduce((m, t) => m | (1 << t), 0) };
};
const PLAN_ONE = plan(0, 1, [0]);
const PLAN_MODE1 = plan(0, 2, [0, 0]);
const PLANS2 = COMMON2.map(([p, invert]) => plan(p, 2, invert ? [1, 0] : [0, 1]));
const PLANS3 = COMMON3.map(([p, perm]) => plan(p, 3, [0, 1, 2].map((b) => PERMS[perm].indexOf(b))));
const PLANS7 = COMMON7.map(([p, k]) => plan(p, 3, [0, 1, 2].map((b) => threeToTwo(b, k))));
function planOf(mode, pattern) {
  if (mode === 1) return PLAN_MODE1;
  if (SUBSETS[mode] === 1) return PLAN_ONE;
  if (mode === 3) return PLANS3[pattern];
  if (mode === 7) return PLANS7[pattern];
  return PLANS2[pattern];
}
// Per UASTC mode: BC7's index bits, and the table taking UASTC's weights to them (null where they match).
const BC7_BITS = BC7_MODE.map((bc7) => (bc7 === 6 ? 4 : bc7 === 1 ? 3 : 2));
const BC7_CONV = BC7_MODE.map((bc7, mode) => (WEIGHT_BITS[mode] === BC7_BITS[mode] || bc7 === 5 || mode === SOLID
  ? null : TO_BC7_WEIGHTS[`${WEIGHT_BITS[mode]}>${BC7_BITS[mode]}`]));

// Scratch, a block at a time: RGBA endpoints per UASTC subset (subset * 8 + lo/hi * 4 + c),
// the BC7 ones per BC7 subset, their p-bits (subset * 2 + lo/hi), and the weights.
const ENDS = new Uint8Array(24), Q = new Uint8Array(24), PB = new Uint8Array(6), W = new Uint8Array(32);

/** An unpacked block's endpoints as RGBA per subset into ENDS, 8-bit. */
function rgbaEndpoints(block) {
  const mode = block.mode, comps = COMPS[mode], e = block.endpoints;
  for (let s = 0; s < SUBSETS[mode]; s++) {
    for (let h = 0; h < 2; h++) {
      const o = s * 8 + h * 4, base = s * comps * 2 + h;
      if (comps === 2) { ENDS[o] = ENDS[o + 1] = ENDS[o + 2] = e[base]; ENDS[o + 3] = e[base + 2]; } else {
        ENDS[o] = e[base]; ENDS[o + 1] = e[base + 2]; ENDS[o + 2] = e[base + 4];
        ENDS[o + 3] = comps === 4 ? e[base + 6] : 255;
      }
    }
  }
}

/** One unpacked UASTC block as a BC7 block, into out at `at`. */
export function encodeBC7(block, out, at) {
  begin();
  bc7Block(block);
  store(out, at);
}

function bc7Block(block) {
  const mode = block.mode;
  if (mode === SOLID) {
    solidTables();
    const s = block.solid, r = s[0], g = s[1], b = s[2], a = s[3];
    const err0 = SOLID6[0][r].error + SOLID6[0][g].error + SOLID6[0][b].error + SOLID6[0][a].error;
    const err1 = SOLID6[1][r].error + SOLID6[1][g].error + SOLID6[1][b].error + SOLID6[1][a].error;
    if (err0 > 0 && err1 > 0) {
      put(1 << 5, 6); put(0, 2);
      put(SOLID5[r].lo, 7); put(SOLID5[r].hi, 7); put(SOLID5[g].lo, 7); put(SOLID5[g].hi, 7); put(SOLID5[b].lo, 7); put(SOLID5[b].hi, 7);
      put(a, 8); put(a, 8);
      // Colour indices all 1, the anchor's a bit short; alpha's all 0, as cleared.
      put(1, 1);
      for (let t = 1; t < 16; t++) put(1, 2);
    } else {
      const p = err1 < err0 ? 1 : 0, solid = SOLID6[p];
      put(1 << 6, 7);
      put(solid[r].lo, 7); put(solid[r].hi, 7); put(solid[g].lo, 7); put(solid[g].hi, 7);
      put(solid[b].lo, 7); put(solid[b].hi, 7); put(solid[a].lo, 7); put(solid[a].hi, 7);
      put(p, 1); put(p, 1);
      put(5, 3);
      for (let t = 1; t < 16; t++) put(5, 4);
    }
    return;
  }

  rgbaEndpoints(block);
  const bc7 = BC7_MODE[mode], raw = block.weights;

  if (bc7 === 5) {
    // Dual plane: the separately weighted component swapped into alpha (BC7's
    // rotation). Colour to 7 bits, alpha kept at 8, each pair swapped and its
    // indices inverted where its anchor's top bit is set.
    const sel = block.selector;
    if (sel < 3) {
      let v = ENDS[sel]; ENDS[sel] = ENDS[3]; ENDS[3] = v;
      v = ENDS[4 + sel]; ENDS[4 + sel] = ENDS[7]; ENDS[7] = v;
    }
    const conv = mode === 13 ? TO_BC7_WEIGHTS['1>2'] : null;
    for (let t = 0; t < 32; t++) W[t] = conv ? conv[raw[t]] : raw[t];
    const flipColour = W[0] > 1, flipAlpha = W[1] > 1;
    put(1 << 5, 6); put((sel + 1) & 3, 2);
    for (let c = 0; c < 3; c++) {
      const lo = Math.floor((ENDS[c] * 127 + 127) / 255), hi = Math.floor((ENDS[4 + c] * 127 + 127) / 255);
      if (flipColour) { put(hi, 7); put(lo, 7); } else { put(lo, 7); put(hi, 7); }
    }
    if (flipAlpha) { put(ENDS[7], 8); put(ENDS[3], 8); } else { put(ENDS[3], 8); put(ENDS[7], 8); }
    put(flipColour ? 3 - W[0] : W[0], 1);
    for (let t = 1; t < 16; t++) put(flipColour ? 3 - W[t * 2] : W[t * 2], 2);
    put(flipAlpha ? 3 - W[1] : W[1], 1);
    for (let t = 1; t < 16; t++) put(flipAlpha ? 3 - W[t * 2 + 1] : W[t * 2 + 1], 2);
    return;
  }

  const { pattern, subsets, from, map, anchors, mask } = planOf(mode, block.pattern);
  const bits = BC7_BITS[mode], conv = BC7_CONV[mode], max = (1 << bits) - 1;
  for (let t = 0; t < 16; t++) W[t] = conv ? conv[raw[t]] : raw[t];

  // Endpoints, quantized to the BC7 mode's bits (with p-bits where it has them).
  const comps = bc7 === 6 || bc7 === 7 ? 4 : 3;
  if (bc7 === 2) {
    for (let s = 0; s < subsets; s++) {
      const o = from[s] * 8;
      for (let c = 0; c < 8; c++) Q[s * 8 + c] = Math.floor((ENDS[o + c] * 31 + 127) / 255);
    }
  } else {
    const { q, unique, shared } = pbitTables(bc7 === 6 || bc7 === 3 ? 7 : bc7 === 1 ? 6 : 5);
    for (let s = 0; s < subsets; s++) {
      const o = from[s] * 8;
      if (bc7 === 1) {
        let e0 = 0, e1 = 0;
        for (let c = 0; c < 3; c++) {
          e0 = f(e0 + f(shared[ENDS[o + c]] + shared[ENDS[o + 4 + c]]));
          e1 = f(e1 + f(shared[256 + ENDS[o + c]] + shared[256 + ENDS[o + 4 + c]]));
        }
        const p = e1 < e0 ? 1 : 0;
        PB[s * 2] = PB[s * 2 + 1] = p;
        for (let c = 0; c < 8; c++) Q[s * 8 + c] = q[p * 256 + ENDS[o + c]];
      } else {
        for (let h = 0; h < 2; h++) {
          const e = o + h * 4;
          let e0 = 0, e1 = 0;
          for (let c = 0; c < comps; c++) { e0 = f(e0 + unique[ENDS[e + c]]); e1 = f(e1 + unique[256 + ENDS[e + c]]); }
          const p = e1 < e0 ? 1 : 0;
          PB[s * 2 + h] = p;
          for (let c = 0; c < 4; c++) Q[s * 8 + h * 4 + c] = q[p * 256 + ENDS[e + c]];
        }
      }
    }
  }
  // Where a subset's anchor index has its top bit set: its endpoints swapped, its indices inverted.
  for (let s = 0; s < subsets; s++) {
    if (W[anchors[s]] <= max >> 1) continue;
    for (let c = 0; c < 4; c++) { const v = Q[s * 8 + c]; Q[s * 8 + c] = Q[s * 8 + 4 + c]; Q[s * 8 + 4 + c] = v; }
    const p = PB[s * 2]; PB[s * 2] = PB[s * 2 + 1]; PB[s * 2 + 1] = p;
    for (let t = 0; t < 16; t++) if (map[t] === s) W[t] = max - W[t];
  }

  put(1 << bc7, bc7 + 1);
  if (subsets > 1) put(pattern, bc7 === 0 ? 4 : 6);
  const cbits = bc7 === 2 || bc7 === 7 ? 5 : bc7 === 1 ? 6 : 7;
  for (let c = 0; c < comps; c++) for (let s = 0; s < subsets; s++) { put(Q[s * 8 + c], cbits); put(Q[s * 8 + 4 + c], cbits); }
  if (bc7 === 1) for (let s = 0; s < subsets; s++) put(PB[s * 2], 1);
  else if (bc7 !== 2) for (let s = 0; s < subsets; s++) { put(PB[s * 2], 1); put(PB[s * 2 + 1], 1); }
  for (let t = 0; t < 16; t++) put(W[t], bits - ((mask >> t) & 1));
}

/** A UASTC level as BC7 blocks: the same 16 bytes a 4x4 block. */
export function uastcToBC7(data, width, height) {
  return transcodeLevel(data, width, height, encodeBC7);
}

function transcodeLevel(data, width, height, encode) {
  const blocks = Math.ceil(width / 4) * Math.ceil(height / 4);
  if (data.length < blocks * 16) throw new Error(`uastc: ${data.length} bytes for ${blocks} blocks`);
  const out = new Uint8Array(blocks * 16), block = newBlock();
  for (let b = 0; b < blocks; b++) encode(unpackBlock(data, b * 16, block), out, b * 16);
  return out;
}

// ---- ASTC 4x4 -------------------------------------------------------------
//
// UASTC is a subset of ASTC 4x4, so this loses nothing: the same weights, the
// same BISE endpoint integers (in the same range ASTC works out from the bits
// left, by the format's design), the patterns at their ASTC seeds (each checked
// to give UASTC's pattern under ASTC's partition function). Only the packing
// differs, and one thing UASTC leaves out: ASTC reads an RGB or RGBA subset whose
// low endpoint sums brighter than its high one as blue-contracted, so such a
// subset's endpoints are swapped and its weights inverted.

/** Each pattern's ASTC partition seed: the 2-subset patterns, the 3-subset, and mode 7's. */
const SEEDS2 = [28, 20, 16, 29, 91, 9, 107, 72, 149, 204, 50, 114, 496, 17, 78, 39, 252, 828, 43, 156, 116, 210, 476, 273, 684, 359, 246, 195, 694, 524];
const SEEDS3 = [260, 74, 32, 156, 183, 15, 745, 0, 335, 902, 254];
const SEEDS7 = [36, 48, 61, 137, 161, 183, 226, 281, 302, 307, 479, 495, 593, 594, 605, 799, 812, 988, 993];
// The block mode for a 4x4 weight grid, by weight bits (ASTC's only 4x4 layout: A = 2, B = 0, and
// the range bits R and H for 2, 4, 8, 16 or 32 levels); dual plane adds bit 10.
const BLOCK_MODE = [0, 0x41, 0x42, 0x53, 0x242, 0x253];
// Colour endpoint modes: LA direct, RGB direct, RGBA direct, by component count.
const CEM = [0, 0, 4, 8, 12];

// ASTC packs 5 trits into 8 bits and 3 quints into 7 by its own mapping; these
// invert the format's decoding, by the group's value in base 3 or 5 (lowest digit first).
let TRIT_CODES = null, QUINT_CODES = null;
function iseTables() {
  if (TRIT_CODES) return;
  const bit = (v, i) => (v >> i) & 1, field = (v, lo, n) => (v >> lo) & ((1 << n) - 1);
  TRIT_CODES = new Uint8Array(243).fill(255);
  for (let T = 0; T < 256; T++) {
    let C, t3, t4, t0, t1, t2;
    if (field(T, 2, 3) === 7) { C = (field(T, 5, 3) << 2) | field(T, 0, 2); t4 = t3 = 2; }
    else {
      C = field(T, 0, 5);
      if (field(T, 5, 2) === 3) { t4 = 2; t3 = bit(T, 7); } else { t4 = bit(T, 7); t3 = field(T, 5, 2); }
    }
    if (field(C, 0, 2) === 3) { t2 = 2; t1 = bit(C, 4); t0 = (bit(C, 3) << 1) | (bit(C, 2) & ~bit(C, 3) & 1); }
    else if (field(C, 2, 2) === 3) { t2 = 2; t1 = 2; t0 = field(C, 0, 2); }
    else { t2 = bit(C, 4); t1 = field(C, 2, 2); t0 = (bit(C, 1) << 1) | (bit(C, 0) & ~bit(C, 1) & 1); }
    // Where a group has more than one code, Basis writes the first.
    const v = t0 + 3 * t1 + 9 * t2 + 27 * t3 + 81 * t4;
    if (TRIT_CODES[v] === 255) TRIT_CODES[v] = T;
  }
  QUINT_CODES = new Uint8Array(125).fill(255);
  for (let Q = 0; Q < 128; Q++) {
    let C, q0, q1, q2;
    if (field(Q, 1, 2) === 3 && field(Q, 5, 2) === 0) {
      q2 = (bit(Q, 0) << 2) | ((bit(Q, 4) & ~bit(Q, 0) & 1) << 1) | (bit(Q, 3) & ~bit(Q, 0) & 1);
      q1 = q0 = 4;
    } else {
      if (field(Q, 1, 2) === 3) { q2 = 4; C = (field(Q, 3, 2) << 3) | ((~field(Q, 5, 2) & 3) << 1) | bit(Q, 0); }
      else { q2 = field(Q, 5, 2); C = field(Q, 0, 5); }
      if (field(C, 0, 3) === 5) { q1 = 4; q0 = field(C, 3, 2); } else { q1 = field(C, 3, 2); q0 = field(C, 0, 3); }
    }
    // Only (4, 4, 4) has more than one code (7, 15, 23, 31); Basis writes 31, the last.
    QUINT_CODES[q0 + 5 * q1 + 25 * q2] = Q;
  }
}
// Where each group's packed bits sit between its values' bits: after value i, this many.
const TRIT_SPLIT = [2, 2, 1, 2, 1], QUINT_SPLIT = [3, 2, 2];

/** `values` BISE integers in `range`, ASTC's interleaved way, from the writer's bit on. */
function putISE(values, count, range) {
  const bits = RANGE_BITS[range], trits = RANGE_TRITS[range], quints = RANGE_QUINTS[range];
  const mask = (1 << bits) - 1;
  if (!trits && !quints) { for (let i = 0; i < count; i++) put(values[i], bits); return; }
  const per = trits ? 5 : 3, base = trits ? 3 : 5, split = trits ? TRIT_SPLIT : QUINT_SPLIT, codes = trits ? TRIT_CODES : QUINT_CODES;
  const end = BIT + count * bits + Math.ceil(count * (trits ? 8 : 7) / per);
  for (let g = 0; g < count; g += per) {
    let digits = 0;
    for (let i = Math.min(per, count - g) - 1; i >= 0; i--) digits = digits * base + (values[g + i] >> bits);
    let packed = codes[digits];
    for (let i = 0; i < per && BIT < end; i++) {
      if (g + i < count) put(values[g + i] & mask, bits);
      const n = Math.min(split[i], end - BIT);
      put(packed & ((1 << n) - 1), n);
      packed >>= split[i];
    }
  }
}

const ASTC_RAW = new Uint8Array(24), ASTC_WEIGHTS = new Uint8Array(32), WEIGHT_STREAM = new Uint8Array(11);
/** Each byte with its bits in reverse order. */
const REVERSED = new Uint8Array(256);
for (let v = 0; v < 256; v++) for (let b = 0; b < 8; b++) REVERSED[v] |= ((v >> b) & 1) << (7 - b);

/** One unpacked UASTC block as an ASTC 4x4 block, into out at `at`. */
export function encodeASTC(block, out, at) {
  begin();
  astcBlock(block);
  store(out, at);
}

function astcBlock(block) {
  const mode = block.mode;
  if (mode === SOLID) {
    // A void-extent block: LDR, its extent all ones (none), the colour as UNORM16.
    put(0xfc, 8); put(0xfd, 8);
    for (let i = 2; i < 8; i++) put(0xff, 8);
    for (const c of block.solid) { put(c, 8); put(c, 8); }
    return;
  }
  iseTables();
  const comps = COMPS[mode], subsets = SUBSETS[mode], planes = PLANES[mode], wbits = WEIGHT_BITS[mode];
  // Copies, so the unpacked block stays as it was for any other use.
  const raw = ASTC_RAW, weights = ASTC_WEIGHTS, e = block.endpoints;
  raw.set(block.raw);
  weights.set(block.weights);
  if (comps >= 3) {
    const max = (1 << wbits) - 1;
    const map = subsets === 1 ? null : mode === 3 ? PATTERNS3[block.pattern] : mode === 7 ? PATTERNS7[block.pattern] : PATTERNS2[block.pattern];
    for (let s = 0; s < subsets; s++) {
      const o = s * comps * 2;
      if (e[o] + e[o + 2] + e[o + 4] <= e[o + 1] + e[o + 3] + e[o + 5]) continue;
      for (let c = 0; c < comps; c++) { const lo = raw[o + c * 2]; raw[o + c * 2] = raw[o + c * 2 + 1]; raw[o + c * 2 + 1] = lo; }
      for (let t = 0; t < 16; t++) {
        if (map !== null && map[t] !== s) continue;
        for (let p = 0; p < planes; p++) weights[t * planes + p] = max - weights[t * planes + p];
      }
    }
  }
  const blockMode = BLOCK_MODE[wbits] | (planes === 2 ? 0x400 : 0);
  put(blockMode & 255, 8);
  put(blockMode >> 8, 3);
  put(subsets - 1, 2);
  if (subsets === 1) put(CEM[comps], 4);
  else {
    const seed = (mode === 3 ? SEEDS3 : mode === 7 ? SEEDS7 : SEEDS2)[block.pattern];
    put(seed & 255, 8);
    put(seed >> 8, 2);
    put(CEM[comps] << 2, 6); // every subset the same mode
  }
  putISE(raw, comps * 2 * subsets, ENDPOINT_RANGE[mode]);
  const weightBits = 16 * planes * wbits;
  if (planes === 2) { BIT = 126 - weightBits; put(block.selector, 2); }
  // The weights from the block's top bit down, each one's bits reversed with
  // it: packed forwards, LSB first, then each byte reversed into the block from its end.
  const stream = WEIGHT_STREAM;
  let acc = 0, held = 0, k = 0;
  for (let i = 0; i < 16 * planes; i++) {
    acc |= weights[i] << held;
    held += wbits;
    while (held >= 8) { stream[k++] = acc & 255; acc >>>= 8; held -= 8; }
  }
  if (held > 0) stream[k++] = acc;
  for (let i = 0; i < k; i++) ACC[(15 - i) >> 2] |= REVERSED[stream[i]] << (((15 - i) & 3) * 8);
}

/** A UASTC level as ASTC 4x4 blocks: the same 16 bytes a 4x4 block, nothing lost. */
export function uastcToASTC(data, width, height) {
  return transcodeLevel(data, width, height, encodeASTC);
}
