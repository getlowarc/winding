// Zstandard decompression (RFC 8878), for KTX2's zstd supercompression.
//
// Browsers unpack gzip and deflate natively (DecompressionStream) but not zstd,
// and a KTX2 texture holding Basis UASTC is nearly always zstd-packed. So this
// is a decoder of our own: frames of raw, RLE and compressed blocks, Huffman-
// coded literals in one stream or four, and sequences under FSE tables --
// predefined, RLE, described, or repeated from the block before. What KTX2
// never uses is left out: dictionaries are refused, and the content checksum
// is skipped rather than verified.
//
// Three readers: bytes forward (headers), bits forward from the low bit (FSE
// table descriptions), and bits BACKWARD from the end (Huffman and FSE
// streams, which an encoder writes so the decoder reads them last-first).

const FRAME_MAGIC = 0xFD2FB528;
const SKIPPABLE_MASK = 0xFFFFFFF0;
const SKIPPABLE_MAGIC = 0x184D2A50;

// Literal and match length codes: baseline and extra bits (RFC 8878 3.1.1.3.2.1.1).
const LL_BASE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32, 40,
  48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];
const LL_BITS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3,
  4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
const ML_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
  27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027, 2051,
  4099, 8195, 16387, 32771, 65539];
const ML_BITS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11,
  12, 13, 14, 15, 16];

// The predefined distributions (3.1.1.3.2.2): -1 is "less than 1".
const LL_DEFAULT = [4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1];
const ML_DEFAULT = [1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1];
const OF_DEFAULT = [1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  -1, -1, -1, -1, -1];

const fail = (why) => { throw new Error(`zstd: ${why}`); };
const highBit = (v) => 31 - Math.clz32(v);

/** Bits read from the low bit up, a byte at a time: FSE table descriptions. */
class ForwardBits {
  constructor(bytes, start, end) { this.bytes = bytes; this.bit = start * 8; this.end = end * 8; }
  peek(n) {
    let v = 0;
    for (let i = 0; i < n; i++) {
      const at = this.bit + i;
      if (at < this.end && (this.bytes[at >> 3] >> (at & 7)) & 1) v |= 1 << i;
    }
    return v;
  }
  read(n) { const v = this.peek(n); this.bit += n; return v; }
  /** The byte after the last bit read. */
  get byte() { return Math.ceil(this.bit / 8); }
}

/**
 * Bits read backward from the end of [start, end): an encoder's stream, its
 * last byte carrying a 1 above the final bit as a marker. Past the start, bits
 * read as 0 and `overflow` says so, which is how the Huffman weight decoder
 * knows where its stream ran out.
 */
class BackwardBits {
  constructor(bytes, start, end) {
    if (end <= start) fail('an empty bitstream');
    const last = bytes[end - 1];
    if (last === 0) fail('a bitstream with no end marker');
    this.bytes = bytes;
    this.start = start;
    // Bits left to read, counted from the stream's first bit.
    this.left = (end - start - 1) * 8 + highBit(last);
  }
  /** The next n bits, most significant first, without taking them. */
  peek(n) {
    // Up to 24 at once from a 32-bit window; an offset's 31 in two.
    if (n > 24) return this._bits(this.left, n - 16) * 65536 + this._bits(this.left - (n - 16), 16);
    return this._bits(this.left, n);
  }
  /** Bits [top - n, top) of the stream, n at most 24; zeros below its start. */
  _bits(top, n) {
    const lo = top - n;
    if (lo < 0) return top <= 0 ? 0 : this._bits(top, top) * (1 << -lo);
    const b = this.start + (lo >> 3), bytes = this.bytes;
    // Bytes past the stream's end read as whatever follows, or undefined (0):
    // only bits below top are kept, and those are the stream's.
    const word = (bytes[b] | (bytes[b + 1] << 8) | (bytes[b + 2] << 16) | (bytes[b + 3] << 24)) >>> 0;
    return (word >>> (lo & 7)) & ((1 << n) - 1);
  }
  read(n) { if (n === 0) return 0; const v = this.peek(n); this.left -= n; return v; }
  get overflow() { return this.left < 0; }
  get done() { return this.left === 0; }
}

/**
 * Bits [top - n, top) of a backward stream starting at `start`, n at most 24:
 * BackwardBits._bits as a plain function, for the sequence loop, which keeps
 * its place in a local rather than an object it reads back on every field.
 */
function backBits(bytes, start, top, n) {
  const lo = top - n;
  if (lo < 0) return top <= 0 ? 0 : backBits(bytes, start, top, top) * (1 << -lo);
  const b = start + (lo >> 3);
  return (((bytes[b] | (bytes[b + 1] << 8) | (bytes[b + 2] << 16) | (bytes[b + 3] << 24)) >>> (lo & 7)) & ((1 << n) - 1));
}

// Whether 32-bit words of a buffer read as its bytes do, least significant first.
const LITTLE_ENDIAN = new Uint8Array(Uint32Array.of(1).buffer)[0] === 1;

/**
 * The input's whole buffer as 32-bit words, for reading a field in one or two
 * loads, or null where it cannot be (a big-endian machine, a buffer past 2^28
 * bytes, whose bit positions would not fit an int).
 */
function wordsOf(bytes) {
  const size = bytes.buffer.byteLength;
  return LITTLE_ENDIAN && size < 2 ** 28 ? new Int32Array(bytes.buffer, 0, size >> 2) : null;
}

/**
 * backBits through the buffer's words: `bitBase` is where the stream starts in
 * the buffer, in bits. Short of n bits, or with no words, the byte path.
 */
function seqRead(bytes, words, bitBase, start, top, n) {
  if (words === null || top < n) return backBits(bytes, start, top, n);
  const lo = bitBase + top - n, w = lo >> 5, sh = lo & 31;
  // The buffer's last bytes, past its last whole word, are not in the view.
  if (w + 1 >= words.length) return backBits(bytes, start, top, n);
  let v = words[w] >>> sh;
  if (sh + n > 32) v |= words[w + 1] << (32 - sh);
  return v & ((1 << n) - 1);
}

/** An FSE decoding table from a distribution: per state its symbol, bits to read, and the baseline they add to. */
function fseTable(probs, accuracy) {
  const size = 1 << accuracy;
  const symbol = new Uint16Array(size), bits = new Uint8Array(size), base = new Uint16Array(size);
  const next = new Uint16Array(probs.length);
  let high = size - 1;
  // "Less than 1": a state each, from the top.
  for (let s = 0; s < probs.length; s++) {
    if (probs[s] === -1) { symbol[high--] = s; next[s] = 1; } else next[s] = probs[s];
  }
  const step = (size >> 1) + (size >> 3) + 3, mask = size - 1;
  let pos = 0;
  for (let s = 0; s < probs.length; s++) {
    for (let i = 0; i < probs[s]; i++) {
      symbol[pos] = s;
      do pos = (pos + step) & mask; while (pos > high);
    }
  }
  if (pos !== 0) fail('a distribution that does not fill its table');
  for (let u = 0; u < size; u++) {
    const s = symbol[u];
    const n = next[s]++;
    bits[u] = accuracy - highBit(n);
    base[u] = (n << bits[u]) - size;
  }
  return { symbol, bits, base, accuracy };
}

/** An FSE table description (4.1.1), read forward from `at`; returns the table and where it ended. */
function readFseTable(bytes, at, end, maxSymbol, maxAccuracy) {
  const r = new ForwardBits(bytes, at, end);
  const accuracy = r.read(4) + 5;
  if (accuracy > maxAccuracy) fail(`an accuracy of ${accuracy}, past ${maxAccuracy}`);
  const probs = [];
  let remaining = (1 << accuracy) + 1, threshold = 1 << accuracy, width = accuracy + 1;
  while (remaining > 1 && probs.length <= maxSymbol) {
    const max = 2 * threshold - 1 - remaining;
    let count;
    const low = r.peek(width - 1);
    if (low < max) {
      count = low;
      r.bit += width - 1;
    } else {
      count = r.peek(width);
      if (count >= threshold) count -= max;
      r.bit += width;
    }
    count--;
    remaining -= count < 0 ? -count : count;
    probs.push(count);
    if (count === 0) {
      // Zeros after a zero: runs of two-bit counts, a 3 meaning more follow.
      let repeat;
      do {
        repeat = r.read(2);
        for (let i = 0; i < repeat; i++) probs.push(0);
      } while (repeat === 3);
    }
    while (remaining < threshold) { width--; threshold >>= 1; }
  }
  if (remaining !== 1) fail('a distribution that does not add up');
  if (probs.length > maxSymbol + 1) fail('a distribution past its last symbol');
  return { table: fseTable(probs, accuracy), end: r.byte };
}

/** A one-symbol "table": RLE mode, the symbol every time, no bits. */
function rleTable(s) {
  return { symbol: Uint16Array.of(s), bits: Uint8Array.of(0), base: Uint16Array.of(0), accuracy: 0 };
}

const DEFAULTS = {
  ll: fseTable(LL_DEFAULT, 6),
  ml: fseTable(ML_DEFAULT, 6),
  of: fseTable(OF_DEFAULT, 5),
};

/** A Huffman tree description (4.2.1): the decoding table, indexed by the next maxBits bits. */
function readHuffman(bytes, at, end) {
  const header = bytes[at];
  const weights = [];
  let after;
  if (header < 128) {
    // Weights under FSE, two interleaved states over one backward stream.
    const stop = at + 1 + header;
    if (stop > end) fail('a Huffman description past its literals');
    const { table, end: streamStart } = readFseTable(bytes, at + 1, stop, 255, 6);
    const r = new BackwardBits(bytes, streamStart, stop);
    let s1 = r.read(table.accuracy), s2 = r.read(table.accuracy);
    for (;;) {
      weights.push(table.symbol[s1]);
      s1 = table.base[s1] + r.read(table.bits[s1]);
      if (r.overflow) { weights.push(table.symbol[s2]); break; }
      weights.push(table.symbol[s2]);
      s2 = table.base[s2] + r.read(table.bits[s2]);
      if (r.overflow) { weights.push(table.symbol[s1]); break; }
      if (weights.length > 255) fail('too many Huffman weights');
    }
    after = stop;
  } else {
    // Four bits a weight, two to a byte, high first.
    const count = header - 127;
    after = at + 1 + Math.ceil(count / 2);
    if (after > end) fail('Huffman weights past their literals');
    for (let i = 0; i < count; i++) weights.push(i & 1 ? bytes[at + 1 + (i >> 1)] & 15 : bytes[at + 1 + (i >> 1)] >> 4);
  }
  // The last weight is implied: what brings the total to a power of two.
  let total = 0;
  for (const w of weights) if (w > 0) total += 1 << (w - 1);
  if (total === 0) fail('a Huffman table with no symbols');
  const maxBits = highBit(total) + 1;
  const rest = (1 << maxBits) - total;
  if (rest & (rest - 1)) fail('Huffman weights that do not complete a tree');
  weights.push(highBit(rest) + 1);
  if (maxBits > 11) fail(`a Huffman code of ${maxBits} bits, past 11`);
  // Lowest weight (longest code) first, each symbol in order, 2^(w-1) entries apiece.
  const size = 1 << maxBits;
  const symbol = new Uint8Array(size), bits = new Uint8Array(size);
  const counts = new Array(maxBits + 2).fill(0);
  for (const w of weights) counts[w]++;
  const start = new Array(maxBits + 2).fill(0);
  for (let w = 1, next = 0; w <= maxBits; w++) { start[w] = next; next += counts[w] << (w - 1); }
  weights.forEach((w, s) => {
    if (w === 0) return;
    const length = 1 << (w - 1);
    for (let i = 0; i < length; i++) { symbol[start[w] + i] = s; bits[start[w] + i] = maxBits + 1 - w; }
    start[w] += length;
  });
  return { table: { symbol, bits, maxBits }, end: after };
}

/** `count` literals from one Huffman stream at [start, end) into out at `at`. */
function huffmanStream(bytes, start, end, table, out, at, count) {
  const r = new BackwardBits(bytes, start, end);
  const { symbol, bits, maxBits } = table;
  const mask = (1 << maxBits) - 1;
  let left = r.left;
  for (let i = 0; i < count; i++) {
    // The next maxBits bits, inline: this loop is most of a literal-heavy block.
    const lo = left - maxBits;
    let v;
    if (lo >= 0) {
      const b = start + (lo >> 3);
      v = (((bytes[b] | (bytes[b + 1] << 8) | (bytes[b + 2] << 16)) >>> (lo & 7)) & mask);
    } else {
      v = r._bits(left, maxBits);
    }
    out[at + i] = symbol[v];
    left -= bits[v];
  }
  if (left !== 0) fail('a Huffman stream that does not end where its literals do');
}

/** One frame's decoding state: what later blocks may reuse. */
class Frame {
  constructor() {
    this.huffman = null;
    this.tables = { ll: null, of: null, ml: null };
    this.repeat = [1, 4, 8];
  }

  /** The literals section at `at`: returns the literals and where the sequences begin. */
  literals(bytes, at, end) {
    const b0 = bytes[at];
    const type = b0 & 3, format = (b0 >> 2) & 3;
    if (type < 2) {
      // Raw or RLE: a size, then the bytes or the one byte.
      let size, header;
      if ((format & 1) === 0) { size = b0 >> 3; header = 1; }
      else if (format === 1) { size = (b0 >> 4) + (bytes[at + 1] << 4); header = 2; }
      else { size = (b0 >> 4) + (bytes[at + 1] << 4) + (bytes[at + 2] << 12); header = 3; }
      const from = at + header;
      if (type === 0) {
        if (from + size > end) fail('raw literals past their block');
        return { literals: bytes.subarray(from, from + size), end: from + size };
      }
      return { literals: new Uint8Array(size).fill(bytes[from]), end: from + 1 };
    }
    // Compressed, or treeless (the last block's tree).
    let regenerated, compressed, header;
    const streams = format === 0 ? 1 : 4;
    if (format < 2) {
      const h = b0 | (bytes[at + 1] << 8) | (bytes[at + 2] << 16);
      regenerated = (h >> 4) & 0x3FF; compressed = (h >> 14) & 0x3FF; header = 3;
    } else if (format === 2) {
      const h = (b0 | (bytes[at + 1] << 8) | (bytes[at + 2] << 16) | (bytes[at + 3] << 24)) >>> 0;
      regenerated = (h >>> 4) & 0x3FFF; compressed = (h >>> 18) & 0x3FFF; header = 4;
    } else {
      const lo = (b0 | (bytes[at + 1] << 8) | (bytes[at + 2] << 16) | (bytes[at + 3] << 24)) >>> 0;
      regenerated = (lo >>> 4) & 0x3FFFF;
      compressed = (lo >>> 22) + (bytes[at + 4] << 10);
      header = 5;
    }
    let from = at + header;
    const stop = from + compressed;
    if (stop > end) fail('compressed literals past their block');
    if (type === 2) {
      const tree = readHuffman(bytes, from, stop);
      this.huffman = tree.table;
      from = tree.end;
    } else if (this.huffman === null) fail('treeless literals with no tree before them');
    const out = new Uint8Array(regenerated);
    if (streams === 1) {
      huffmanStream(bytes, from, stop, this.huffman, out, 0, regenerated);
    } else {
      // A jump table: the first three streams' sizes; the fourth is the rest.
      const s1 = bytes[from] | (bytes[from + 1] << 8);
      const s2 = bytes[from + 2] | (bytes[from + 3] << 8);
      const s3 = bytes[from + 4] | (bytes[from + 5] << 8);
      const each = Math.ceil(regenerated / 4);
      let p = from + 6;
      const sizes = [s1, s2, s3, stop - (from + 6) - s1 - s2 - s3];
      if (sizes[3] <= 0) fail('a jump table past its literals');
      for (let k = 0; k < 4; k++) {
        const count = k < 3 ? each : regenerated - 3 * each;
        huffmanStream(bytes, p, p + sizes[k], this.huffman, out, k * each, count);
        p += sizes[k];
      }
    }
    return { literals: out, end: stop };
  }

  /** A compressed block at [at, end): appended to out at `o`; returns where the output ends. */
  block(bytes, at, end, out, o) {
    const { literals, end: seqAt } = this.literals(bytes, at, end);
    let p = seqAt;
    let count = bytes[p++];
    if (count === 0) {
      out.set(literals, o);
      return o + literals.length;
    }
    if (count === 255) { count = bytes[p] + (bytes[p + 1] << 8) + 0x7F00; p += 2; }
    else if (count >= 128) { count = ((count - 128) << 8) + bytes[p]; p += 1; }
    const modes = bytes[p++];
    const pick = (mode, which, maxSymbol, maxAccuracy) => {
      if (mode === 0) return DEFAULTS[which];
      if (mode === 1) return rleTable(bytes[p++]);
      if (mode === 2) {
        const { table, end: after } = readFseTable(bytes, p, end, maxSymbol, maxAccuracy);
        p = after;
        return table;
      }
      if (this.tables[which] === null) fail('a repeated table with none before it');
      return this.tables[which];
    };
    const ll = this.tables.ll = pick(modes >> 6, 'll', 35, 9);
    const of = this.tables.of = pick((modes >> 4) & 3, 'of', 31, 8);
    const ml = this.tables.ml = pick((modes >> 2) & 3, 'ml', 52, 9);

    const start = p, words = wordsOf(bytes), bitBase = (bytes.byteOffset + start) * 8;
    let left = new BackwardBits(bytes, p, end).left;
    let sll = seqRead(bytes, words, bitBase, start, left, ll.accuracy); left -= ll.accuracy;
    let sof = seqRead(bytes, words, bitBase, start, left, of.accuracy); left -= of.accuracy;
    let sml = seqRead(bytes, words, bitBase, start, left, ml.accuracy); left -= ml.accuracy;
    const llSymbol = ll.symbol, llBits = ll.bits, llBase = ll.base;
    const mlSymbol = ml.symbol, mlBits = ml.bits, mlBase = ml.base;
    const ofSymbol = of.symbol, ofBits = of.bits, ofBase = of.base;
    const rep = this.repeat;
    let lit = 0;
    for (let i = 0; i < count; i++) {
      const ofCode = ofSymbol[sof], mlCode = mlSymbol[sml], llCode = llSymbol[sll];
      if (ofCode > 31) fail('an offset code past 31');
      // An offset's extra bits can be 31, more than one window holds.
      let ofValue;
      if (ofCode <= 24) ofValue = (1 << ofCode) + seqRead(bytes, words, bitBase, start, left, ofCode);
      else ofValue = 2 ** ofCode + seqRead(bytes, words, bitBase, start, left, ofCode - 16) * 65536 + seqRead(bytes, words, bitBase, start, left - (ofCode - 16), 16);
      left -= ofCode;
      const mlExtra = ML_BITS[mlCode], matchLength = ML_BASE[mlCode] + seqRead(bytes, words, bitBase, start, left, mlExtra);
      left -= mlExtra;
      const llExtra = LL_BITS[llCode], literalLength = LL_BASE[llCode] + seqRead(bytes, words, bitBase, start, left, llExtra);
      left -= llExtra;
      let offset;
      if (ofValue > 3) {
        offset = ofValue - 3;
        rep[2] = rep[1]; rep[1] = rep[0]; rep[0] = offset;
      } else {
        const index = literalLength === 0 ? ofValue + 1 : ofValue;
        if (index === 1) offset = rep[0];
        else if (index === 2) { offset = rep[1]; rep[1] = rep[0]; rep[0] = offset; }
        else if (index === 3) { offset = rep[2]; rep[2] = rep[1]; rep[1] = rep[0]; rep[0] = offset; }
        else { offset = rep[0] - 1; rep[2] = rep[1]; rep[1] = rep[0]; rep[0] = offset; }
      }
      if (i < count - 1) {
        let n = llBits[sll]; sll = llBase[sll] + seqRead(bytes, words, bitBase, start, left, n); left -= n;
        n = mlBits[sml]; sml = mlBase[sml] + seqRead(bytes, words, bitBase, start, left, n); left -= n;
        n = ofBits[sof]; sof = ofBase[sof] + seqRead(bytes, words, bitBase, start, left, n); left -= n;
      }
      // The literals, then the match.
      if (lit + literalLength > literals.length) fail('a sequence past its literals');
      if (literalLength < 32) for (let k = 0; k < literalLength; k++) out[o + k] = literals[lit + k];
      else out.set(literals.subarray(lit, lit + literalLength), o);
      o += literalLength; lit += literalLength;
      if (offset > o || offset === 0) fail(`an offset of ${offset} before the start`);
      // A match that overlaps itself repeats what it is still writing: a byte at a time.
      if (offset >= matchLength && matchLength >= 32) out.copyWithin(o, o - offset, o - offset + matchLength);
      else for (let k = 0; k < matchLength; k++) out[o + k] = out[o + k - offset];
      o += matchLength;
    }
    if (left !== 0) fail('sequences that do not end where their block does');
    out.set(literals.subarray(lit), o);
    return o + literals.length - lit;
  }
}

/**
 * Every frame in `bytes`, decompressed and joined. `size`, when known (KTX2
 * gives each level's), sizes the output in one go and is checked; otherwise
 * each frame's content size is used, or the output grows.
 */
export function zstdDecompress(bytes, size = -1) {
  // A known size gets a block's worth of slack: a compressed block is written
  // before its length is known, and is checked against the size after.
  let out = new Uint8Array(size >= 0 ? size + (1 << 17) : 1 << 16);
  let o = 0, at = 0;
  const room = (n) => {
    if (o + n <= out.length) return;
    if (size >= 0 && o + n > size) fail(`more than the ${size} bytes expected`);
    const grown = new Uint8Array(Math.max(out.length * 2, o + n));
    grown.set(out.subarray(0, o));
    out = grown;
  };
  const u32 = (p) => (bytes[p] | (bytes[p + 1] << 8) | (bytes[p + 2] << 16) | (bytes[p + 3] << 24)) >>> 0;
  while (at < bytes.length) {
    const magic = u32(at);
    if ((magic & SKIPPABLE_MASK) === SKIPPABLE_MAGIC) { at += 8 + u32(at + 4); continue; }
    if (magic !== FRAME_MAGIC) fail('not a zstd frame');
    at += 4;
    const descriptor = bytes[at++];
    const fcsFlag = descriptor >> 6, single = (descriptor >> 5) & 1, checksum = (descriptor >> 2) & 1, dictFlag = descriptor & 3;
    if ((descriptor >> 3) & 1) fail('a reserved bit set in a frame header');
    if (!single) at++;  // the window descriptor: the whole frame is held anyway
    if (dictFlag !== 0) {
      const dictionary = [0, bytes[at], bytes[at] | (bytes[at + 1] << 8), u32(at)][dictFlag];
      if (dictionary !== 0) fail('a frame that needs a dictionary');
      at += [0, 1, 2, 4][dictFlag];
    }
    const fcsBytes = [single ? 1 : 0, 2, 4, 8][fcsFlag];
    if (fcsBytes > 0) {
      const content = fcsBytes === 1 ? bytes[at] : fcsBytes === 2 ? (bytes[at] | (bytes[at + 1] << 8)) + 256
        : fcsBytes === 4 ? u32(at) : u32(at) + u32(at + 4) * 2 ** 32;
      at += fcsBytes;
      room(content);
    }
    const frame = new Frame();
    for (let last = 0; !last;) {
      const h = bytes[at] | (bytes[at + 1] << 8) | (bytes[at + 2] << 16);
      last = h & 1;
      const type = (h >> 1) & 3, blockSize = h >> 3;
      at += 3;
      if (type === 0) {
        if (at + blockSize > bytes.length) fail('a raw block past the end');
        room(blockSize);
        out.set(bytes.subarray(at, at + blockSize), o);
        o += blockSize; at += blockSize;
      } else if (type === 1) {
        room(blockSize);
        out.fill(bytes[at], o, o + blockSize);
        o += blockSize; at += 1;
      } else if (type === 2) {
        if (at + blockSize > bytes.length) fail('a compressed block past the end');
        // A block regenerates at most 128 KiB.
        if (size < 0) room(1 << 17);
        else if (o > size) fail(`more than the ${size} bytes expected`);
        o = frame.block(bytes, at, at + blockSize, out, o);
        at += blockSize;
      } else fail('a reserved block type');
    }
    if (checksum) at += 4;
  }
  if (size >= 0 && o !== size) fail(`${o} bytes where ${size} were expected`);
  return out.subarray(0, o);
}
