// UASTC LDR 4x4 (Basis Universal), transcoded: each 16-byte block unpacked to
// its endpoints and weights, then written out as RGBA8 texels or as a BC7 block.
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

/** A 16-byte block's bits, LSB first. */
function readBits(bytes, at, offset, count) {
  let v = 0;
  for (let i = 0; i < count; i++) {
    const bit = offset + i;
    v += ((bytes[at + (bit >> 3)] >> (bit & 7)) & 1) * 2 ** i;
  }
  return v;
}

/** The trits or quints of a group, least significant first. */
function splitGroup(value, base, count, out) {
  for (let i = 0; i < count; i++) { out.push(value % base); value = Math.floor(value / base); }
}
const TRIT_GROUP_BITS = [0, 2, 4, 5, 7, 8];
const QUINT_GROUP_BITS = [0, 3, 5, 7];

/**
 * One UASTC block at bytes[at..at+16), unpacked: { mode, solid (RGBA) | endpoints
 * (dequantized, ASTC's order), weights (raw, planes interleaved), pattern, selector }.
 */
export function unpackBlock(bytes, at, out = {}) {
  const mode = HUFF_MODES[bytes[at] & 127];
  if (mode === 19) throw new Error('uastc: a block in the reserved mode 19');
  out.mode = mode;
  if (mode === SOLID) {
    out.solid = [readBits(bytes, at, 5, 8), readBits(bytes, at, 13, 8), readBits(bytes, at, 21, 8), readBits(bytes, at, 29, 8)];
    return out;
  }
  let bit = START[mode];
  out.pattern = PATTERN_BITS[mode] ? readBits(bytes, at, bit, PATTERN_BITS[mode]) : 0;
  bit += PATTERN_BITS[mode];
  out.selector = mode === 17 ? 3 : HAS_SELECTOR[mode] ? readBits(bytes, at, bit, 2) : 0;
  if (HAS_SELECTOR[mode]) bit += 2;

  const range = ENDPOINT_RANGE[mode];
  const [bits, trits, quints] = RANGES[range];
  const count = COMPS[mode] * 2 * SUBSETS[mode];
  const high = [];
  if (trits || quints) {
    const per = trits ? 5 : 3, base = trits ? 3 : 5, sizes = trits ? TRIT_GROUP_BITS : QUINT_GROUP_BITS;
    for (let left = count; left > 0; left -= per) {
      const n = Math.min(per, left);
      splitGroup(readBits(bytes, at, bit, sizes[n]), base, n, high);
      bit += sizes[n];
    }
  }
  const dequant = DEQUANT[range];
  const endpoints = out.endpoints ?? (out.endpoints = new Uint8Array(24));
  for (let i = 0; i < count; i++) {
    const low = readBits(bytes, at, bit, bits);
    bit += bits;
    endpoints[i] = dequant[(trits || quints ? high[i] << bits : 0) | low];
  }

  // Weights: each subset's anchor texel a bit short (both planes' in a dual-plane mode).
  const wbits = WEIGHT_BITS[mode], planes = PLANES[mode];
  const anchors = anchorsOf(mode, out.pattern);
  const weights = out.weights ?? (out.weights = new Uint8Array(32));
  for (let t = 0; t < 16; t++) {
    const short = anchors.includes(t) ? 1 : 0;
    for (let p = 0; p < planes; p++) {
      weights[t * planes + p] = readBits(bytes, at, bit, wbits - short);
      bit += wbits - short;
    }
  }
  return out;
}

function anchorsOf(mode, pattern) {
  if (SUBSETS[mode] === 1) return [0];
  if (mode === 3) return ANCHORS3[pattern];
  if (mode === 7) return ANCHORS7[pattern];
  return ANCHORS2[pattern];
}

/** The texel's subset, in UASTC's numbering. */
function subsetOf(mode, pattern, t) {
  if (SUBSETS[mode] === 1) return 0;
  if (mode === 3) return PATTERNS3[pattern][t];
  if (mode === 7) return PATTERNS7[pattern][t];
  return PATTERNS2[pattern][t];
}

/** ASTC's interpolation of 8-bit endpoints at a [0, 64] weight, decoding as linear (the spec's srgb = false). */
function interpolate(l, h, w) {
  l = (l << 8) | l;
  h = (h << 8) | h;
  return ((l * (64 - w) + h * w + 32) >> 6) >> 8;
}

/** An unpacked block's 16 texels as RGBA8, into out at `at` with `stride` bytes a row. */
export function decodeBlock(block, out, at, stride) {
  const { mode } = block;
  if (mode === SOLID) {
    for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) out.set(block.solid, at + y * stride + x * 4);
    return;
  }
  const comps = COMPS[mode], planes = PLANES[mode], table = WEIGHTS[WEIGHT_BITS[mode]];
  const e = block.endpoints, w = block.weights;
  const value = new Uint8Array(4);
  for (let t = 0; t < 16; t++) {
    const base = subsetOf(mode, block.pattern, t) * comps * 2;
    for (let c = 0; c < comps; c++) {
      const plane = planes === 2 && c === (comps === 2 && block.selector === 3 ? 1 : block.selector) ? 1 : 0;
      value[c] = interpolate(e[base + c * 2], e[base + c * 2 + 1], table[w[t * planes + plane]]);
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
  const block = {};
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
