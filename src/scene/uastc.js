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
  // At most 8 bits a field, so the two bytes it can straddle hold it. A byte
  // past the block reads as the next block's, masked away.
  const i = at + (offset >> 3);
  return ((bytes[i] | (bytes[i + 1] << 8)) >> (offset & 7)) & ((1 << count) - 1);
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

/**
 * BC7 endpoints with p-bits, from [0, 1] endpoints xl and xh: the spec's
 * determine_unique_pbits (shared === false) or determine_shared_pbits, in
 * single precision as written, so ties and roundings fall as Basis's do.
 * Writes the colours (comp bits, before the p-bit) and the p-bits.
 */
function bestPbits(shared, totalComps, compBits, xl, xh, lo, hi, pbits) {
  const totalBits = compBits + 1, iscalep = (1 << totalBits) - 1, scalep = f(iscalep);
  let best0 = f(1e9), best1 = f(1e9);
  const minC = [0, 0, 0, 0], maxC = [0, 0, 0, 0];
  for (let p = 0; p < 2; p++) {
    for (let c = 0; c < 4; c++) {
      const qmin = Math.trunc(f(f(f(f(xl[c] * scalep) - p) / 2) + 0.5)) * 2 + p;
      const qmax = Math.trunc(f(f(f(f(xh[c] * scalep) - p) / 2) + 0.5)) * 2 + p;
      minC[c] = Math.min(Math.max(qmin, p), iscalep - 1 + p);
      maxC[c] = Math.min(Math.max(qmax, p), iscalep - 1 + p);
    }
    let err0 = f(0), err1 = f(0);
    for (let i = 0; i < totalComps; i++) {
      let sl = (minC[i] << (8 - totalBits)) & 255; sl |= sl >> totalBits;
      let sh = (maxC[i] << (8 - totalBits)) & 255; sh |= sh >> totalBits;
      if (shared) {
        const dl = f(f(sl / 255) - xl[i]), dh = f(f(sh / 255) - xh[i]);
        err0 = f(err0 + f(f(dl * dl) + f(dh * dh)));
      } else {
        const dl = f(sl - f(xl[i] * 255)), dh = f(sh - f(xh[i] * 255));
        err0 = f(err0 + f(dl * dl));
        err1 = f(err1 + f(dh * dh));
      }
    }
    if (shared) {
      if (err0 < best0) {
        best0 = err0; pbits[0] = pbits[1] = p;
        for (let j = 0; j < 4; j++) { lo[j] = minC[j] >> 1; hi[j] = maxC[j] >> 1; }
      }
    } else {
      if (err0 < best0) { best0 = err0; pbits[0] = p; for (let j = 0; j < 4; j++) lo[j] = minC[j] >> 1; }
      if (err1 < best1) { best1 = err1; pbits[1] = p; for (let j = 0; j < 4; j++) hi[j] = maxC[j] >> 1; }
    }
  }
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


/** A block's bits, written LSB first. */
class BitWriter {
  constructor(out, at) { this.out = out; this.at = at; this.bit = 0; out.fill(0, at, at + 16); }
  /** At most 8 bits: into the byte it starts in and the one after. */
  put(value, bits) {
    const i = this.at + (this.bit >> 3), shift = this.bit & 7;
    this.out[i] |= (value << shift) & 255;
    if (shift + bits > 8) this.out[i + 1] |= value >> (8 - shift);
    this.bit += bits;
  }
}

/** An unpacked block's endpoints as RGBA per subset: [subset][lo|hi][r, g, b, a], 8-bit. */
function rgbaEndpoints(block) {
  const mode = block.mode, comps = COMPS[mode], e = block.endpoints;
  const out = [];
  for (let s = 0; s < SUBSETS[mode]; s++) {
    const pair = [[0, 0, 0, 255], [0, 0, 0, 255]];
    for (let h = 0; h < 2; h++) {
      const v = (c) => e[s * comps * 2 + c * 2 + h];
      if (comps === 2) { pair[h][0] = pair[h][1] = pair[h][2] = v(0); pair[h][3] = v(1); }
      else for (let c = 0; c < comps; c++) pair[h][c] = v(c);
    }
    out.push(pair);
  }
  return out;
}

/** Swap subset s's endpoints and invert its weights where its anchor's top bit is set. */
function fixAnchor(weights, subsetOfTexel, s, anchor, bits, ends, pbits) {
  const max = (1 << bits) - 1;
  if (weights[anchor] <= max >> 1) return;
  [ends[s][0], ends[s][1]] = [ends[s][1], ends[s][0]];
  if (pbits) [pbits[s][0], pbits[s][1]] = [pbits[s][1], pbits[s][0]];
  for (let t = 0; t < 16; t++) if (subsetOfTexel[t] === s) weights[t] = max - weights[t];
}

/** Indices, each anchor texel's a bit short. */
function putIndices(w, weights, bits, anchors) {
  for (let t = 0; t < 16; t++) w.put(weights[t], bits - (anchors.includes(t) ? 1 : 0));
}

/** One unpacked UASTC block as a BC7 block, into out at `at`. */
export function encodeBC7(block, out, at) {
  const w = new BitWriter(out, at);
  const mode = block.mode;
  if (mode === SOLID) {
    solidTables();
    const [r, g, b, a] = block.solid;
    const err0 = SOLID6[0][r].error + SOLID6[0][g].error + SOLID6[0][b].error + SOLID6[0][a].error;
    const err1 = SOLID6[1][r].error + SOLID6[1][g].error + SOLID6[1][b].error + SOLID6[1][a].error;
    if (err0 > 0 && err1 > 0) {
      w.put(1 << 5, 6); w.put(0, 2);
      for (const c of [r, g, b]) { w.put(SOLID5[c].lo, 7); w.put(SOLID5[c].hi, 7); }
      w.put(a, 8); w.put(a, 8);
      putIndices(w, new Array(16).fill(1), 2, [0]);
      putIndices(w, new Array(16).fill(0), 2, [0]);
    } else {
      const p = err1 < err0 ? 1 : 0;
      w.put(1 << 6, 7);
      for (const c of [r, g, b, a]) { w.put(SOLID6[p][c].lo, 7); w.put(SOLID6[p][c].hi, 7); }
      w.put(p, 1); w.put(p, 1);
      putIndices(w, new Array(16).fill(5), 4, [0]);
    }
    return;
  }

  const bc7 = BC7_MODE[mode], ends = rgbaEndpoints(block), planes = PLANES[mode];
  const raw = block.weights;
  const xl = new Array(4), xh = new Array(4);
  const norm = (pair) => { for (let c = 0; c < 4; c++) { xl[c] = f(pair[0][c] / 255); xh[c] = f(pair[1][c] / 255); } };

  if (bc7 === 5) {
    // Dual plane: the separately weighted component swapped into alpha (BC7's rotation).
    const sel = block.selector, rotation = (sel + 1) & 3;
    const [lo, hi] = ends[0];
    if (sel < 3) { [lo[sel], lo[3]] = [lo[3], lo[sel]]; [hi[sel], hi[3]] = [hi[3], hi[sel]]; }
    const conv = mode === 13 ? TO_BC7_WEIGHTS['1>2'] : null;
    const cw = new Array(16), aw = new Array(16);
    for (let t = 0; t < 16; t++) {
      cw[t] = conv ? conv[raw[t * 2]] : raw[t * 2];
      aw[t] = conv ? conv[raw[t * 2 + 1]] : raw[t * 2 + 1];
    }
    const colour = [[lo.slice(0, 3), hi.slice(0, 3)]], alpha = [[[lo[3]], [hi[3]]]];
    const all = new Array(16).fill(0);
    fixAnchor(cw, all, 0, 0, 2, colour);
    fixAnchor(aw, all, 0, 0, 2, alpha);
    w.put(1 << 5, 6); w.put(rotation, 2);
    for (let c = 0; c < 3; c++) for (let h = 0; h < 2; h++) w.put(Math.floor((colour[0][h][c] * 127 + 127) / 255), 7);
    w.put(alpha[0][0][0], 8); w.put(alpha[0][1][0], 8);
    putIndices(w, cw, 2, [0]);
    putIndices(w, aw, 2, [0]);
    return;
  }

  // The BC7 pattern and, per BC7 subset, which UASTC subset's endpoints it takes.
  let pattern = 0, subsets = 1, fromUastc = [0];
  if (mode === 1) { pattern = 0; subsets = 2; fromUastc = [0, 0]; }
  else if (SUBSETS[mode] === 2 && mode !== 7) {
    const [p, invert] = COMMON2[block.pattern];
    pattern = p; subsets = 2; fromUastc = invert ? [1, 0] : [0, 1];
  } else if (mode === 3) {
    const [p, perm] = COMMON3[block.pattern];
    pattern = p; subsets = 3; fromUastc = [0, 1, 2].map((b) => PERMS[perm].indexOf(b));
  } else if (mode === 7) {
    const [p, k] = COMMON7[block.pattern];
    pattern = p; subsets = 3; fromUastc = [0, 1, 2].map((b) => threeToTwo(b, k));
  }
  const map = subsets === 1 ? new Array(16).fill(0) : subsets === 2 ? BC7_PARTITIONS2[pattern] : BC7_PARTITIONS3[pattern];
  const anchors = subsets === 1 ? [0] : subsets === 2 ? [0, BC7_ANCHOR2[pattern]] : [0, BC7_ANCHOR3A[pattern], BC7_ANCHOR3B[pattern]];

  // Weights, in BC7's index bits.
  const bits = bc7 === 6 ? 4 : bc7 === 1 ? 3 : 2;
  const conv = WEIGHT_BITS[mode] === bits ? null : TO_BC7_WEIGHTS[`${WEIGHT_BITS[mode]}>${bits}`];
  const weights = new Array(16);
  for (let t = 0; t < 16; t++) weights[t] = conv ? conv[raw[t]] : raw[t];

  // Endpoints, quantized to the BC7 mode's bits (with p-bits where it has them).
  const q = [], pb = [];
  for (let s = 0; s < subsets; s++) {
    const pair = ends[fromUastc[s]];
    const lo = [0, 0, 0, 0], hi = [0, 0, 0, 0], p = [0, 0];
    if (bc7 === 2) {
      for (let c = 0; c < 3; c++) { lo[c] = Math.floor((pair[0][c] * 31 + 127) / 255); hi[c] = Math.floor((pair[1][c] * 31 + 127) / 255); }
    } else {
      norm(pair);
      bestPbits(bc7 === 1, bc7 === 6 || bc7 === 7 ? 4 : 3, bc7 === 6 || bc7 === 3 ? 7 : bc7 === 1 ? 6 : 5, xl, xh, lo, hi, p);
    }
    q.push([lo, hi]); pb.push(p);
  }
  for (let s = 0; s < subsets; s++) fixAnchor(weights, map, s, anchors[s], bits, q, bc7 === 2 ? null : pb);

  w.put(1 << bc7, bc7 + 1);
  if (subsets > 1) w.put(pattern, bc7 === 0 ? 4 : 6);
  const comps = bc7 === 6 || bc7 === 7 ? 4 : 3, cbits = bc7 === 2 || bc7 === 7 ? 5 : bc7 === 1 ? 6 : 7;
  for (let c = 0; c < comps; c++) for (let s = 0; s < subsets; s++) { w.put(q[s][0][c], cbits); w.put(q[s][1][c], cbits); }
  if (bc7 === 1) for (let s = 0; s < subsets; s++) w.put(pb[s][0], 1);
  else if (bc7 !== 2) for (let s = 0; s < subsets; s++) { w.put(pb[s][0], 1); w.put(pb[s][1], 1); }
  putIndices(w, weights, bits, anchors);
}

/** A UASTC level as BC7 blocks: the same 16 bytes a 4x4 block. */
export function uastcToBC7(data, width, height) {
  const blocks = Math.ceil(width / 4) * Math.ceil(height / 4);
  if (data.length < blocks * 16) throw new Error(`uastc: ${data.length} bytes for ${blocks} blocks`);
  const out = new Uint8Array(blocks * 16), block = {};
  for (let b = 0; b < blocks; b++) encodeBC7(unpackBlock(data, b * 16, block), out, b * 16);
  return out;
}
