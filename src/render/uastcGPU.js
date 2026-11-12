// UASTC transcoded on the GPU: a compute shader that turns each 16-byte UASTC
// block into the BC7 or ASTC 4x4 block scene/uastc.js writes, one invocation a
// block. The CPU is left with zstd (scene/ktx2.js, on the loader's workers):
// on Iris Xe, transcoding all of Sponza on the CPU was 2.25 s of one thread.
//
// The shader is scene/uastc.js line for line -- the same unpacking, the same
// BC7 plans and p-bit tables summed in the same single-precision order, the
// same ASTC packing -- and reads that file's own tables (packedTables), so the
// two cannot drift apart. The GPU suite holds it to the JavaScript, which the
// Node suite holds to Basis Universal, block for block.
//
// Out: a buffer with each level's rows of blocks 256-byte aligned, as a copy
// from a buffer into a texture must be, copied into the texture by level.

import { packedTables } from '../scene/uastc.js';
import { createBuffer } from '../rhi/buffer.js';
import { compileShader } from '../rhi/shader.js';
import { sharedPipelines } from '../rhi/pipeline.js';
import { createPipelineLayout } from '../rhi/bindgroups.js';

const WORKGROUP = 64;

const shaderSource = (at) => /* wgsl */ `
${Object.entries(at).map(([name, offset]) => `const ${name}_AT = ${offset}u;`).join('\n')}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read> tables: array<u32>;
// [blocks, levels, invocations a dispatch row, 0], then per level:
// [first block, blocks across, first output word, output words a row].
@group(0) @binding(3) var<storage, read> levelInfo: array<u32>;

fn T(i: u32) -> u32 { return tables[i]; }

// ---- the block, unpacked (scene/uastc.js unpackBlock)
var<private> blk: array<u32, 5>;
var<private> mode: u32;
var<private> pattern: u32;
var<private> selector: u32;
var<private> solid: array<u32, 4>;
var<private> ends: array<u32, 24>;
var<private> raw: array<u32, 24>;
var<private> wts: array<u32, 32>;

fn field(offset: u32, count: u32) -> u32 {
  let w = offset >> 5u;
  let s = offset & 31u;
  var v = blk[w] >> s;
  if (s + count > 32u) { v = v | (blk[w + 1u] << (32u - s)); }
  return v & ((1u << count) - 1u);
}

fn anchorMask() -> u32 {
  if (T(SUBSETS_AT + mode) == 1u) { return 1u; }
  if (mode == 3u) { return T(ANCHOR_MASK3_AT + pattern); }
  if (mode == 7u) { return T(ANCHOR_MASK7_AT + pattern); }
  return T(ANCHOR_MASK2_AT + pattern);
}

/** UASTC's subset of texel t, as a word of two bits a texel. */
fn uastcPattern() -> u32 {
  if (T(SUBSETS_AT + mode) == 1u) { return 0u; }
  if (mode == 3u) { return T(PATTERNS3_AT + pattern); }
  if (mode == 7u) { return T(PATTERNS7_AT + pattern); }
  return T(PATTERNS2_AT + pattern);
}

/** False for the reserved mode 19, which is written as zeros. */
fn unpack() -> bool {
  mode = T(huff_AT + (blk[0] & 127u));
  if (mode == 19u) { return false; }
  if (mode == 8u) {
    solid[0] = field(5u, 8u); solid[1] = field(13u, 8u); solid[2] = field(21u, 8u); solid[3] = field(29u, 8u);
    return true;
  }
  var bit = T(START_AT + mode);
  let patternBits = T(PATTERN_BITS_AT + mode);
  pattern = field(bit, patternBits);
  bit += patternBits;
  selector = 0u;
  if (T(HAS_SELECTOR_AT + mode) == 1u) { selector = field(bit, 2u); bit += 2u; }
  if (mode == 17u) { selector = 3u; }

  let range = T(ENDPOINT_RANGE_AT + mode);
  let bits = T(RANGE_BITS_AT + range);
  let trits = T(RANGE_TRITS_AT + range);
  let quints = T(RANGE_QUINTS_AT + range);
  let count = T(COMPS_AT + mode) * 2u * T(SUBSETS_AT + mode);
  var high: array<u32, 24>;
  if (trits == 1u) {
    for (var i = 0u; i < count; i += 5u) {
      let n = min(5u, count - i);
      let groupBits = T(TRIT_GROUP_BITS_AT + n);
      let v = field(bit, groupBits);
      bit += groupBits;
      for (var k = 0u; k < n; k++) { high[i + k] = T(TRIT_DIGITS_AT + v * 5u + k); }
    }
  } else if (quints == 1u) {
    for (var i = 0u; i < count; i += 3u) {
      let n = min(3u, count - i);
      let groupBits = T(QUINT_GROUP_BITS_AT + n);
      let v = field(bit, groupBits);
      bit += groupBits;
      for (var k = 0u; k < n; k++) { high[i + k] = T(QUINT_DIGITS_AT + v * 3u + k); }
    }
  }
  let dequant = T(DEQUANT_AT_AT + range);
  for (var i = 0u; i < count; i++) {
    let low = field(bit, bits);
    bit += bits;
    var v = low;
    if (trits + quints > 0u) { v = (high[i] << bits) | low; }
    raw[i] = v;
    ends[i] = T(dequant + v);
  }

  let wbits = T(WEIGHT_BITS_AT + mode);
  let anchors = anchorMask();
  if (T(PLANES_AT + mode) == 1u) {
    for (var t = 0u; t < 16u; t++) {
      let n = wbits - ((anchors >> t) & 1u);
      wts[t] = field(bit, n);
      bit += n;
    }
  } else {
    for (var t = 0u; t < 16u; t++) {
      let n = wbits - ((anchors >> t) & 1u);
      wts[t * 2u] = field(bit, n);
      wts[t * 2u + 1u] = field(bit + n, n);
      bit += 2u * n;
    }
  }
  return true;
}

// ---- the block written, LSB first
var<private> acc: array<u32, 5>;
var<private> bitpos: u32;

fn begin() { acc[0] = 0u; acc[1] = 0u; acc[2] = 0u; acc[3] = 0u; acc[4] = 0u; bitpos = 0u; }
fn put(value: u32, bits: u32) {
  let w = bitpos >> 5u;
  let s = bitpos & 31u;
  acc[w] = acc[w] | (value << s);
  if (s + bits > 32u) { acc[w + 1u] = acc[w + 1u] | (value >> (32u - s)); }
  bitpos += bits;
}

// ---- BC7 (scene/uastc.js bc7Block)
var<private> E: array<u32, 24>;
var<private> Q: array<u32, 24>;
var<private> PB: array<u32, 6>;
var<private> W: array<u32, 32>;

fn rgbaEndpoints() {
  let comps = T(COMPS_AT + mode);
  for (var s = 0u; s < T(SUBSETS_AT + mode); s++) {
    for (var h = 0u; h < 2u; h++) {
      let o = s * 8u + h * 4u;
      let base = s * comps * 2u + h;
      if (comps == 2u) {
        E[o] = ends[base]; E[o + 1u] = ends[base]; E[o + 2u] = ends[base]; E[o + 3u] = ends[base + 2u];
      } else {
        E[o] = ends[base]; E[o + 1u] = ends[base + 2u]; E[o + 2u] = ends[base + 4u];
        E[o + 3u] = select(255u, ends[base + 6u], comps == 4u);
      }
    }
  }
}

fn solidBC7() {
  var err0 = 0u;
  var err1 = 0u;
  for (var c = 0u; c < 4u; c++) {
    err0 += T(SOLID6_AT + solid[c]) >> 16u;
    err1 += T(SOLID6_AT + 256u + solid[c]) >> 16u;
  }
  if (err0 > 0u && err1 > 0u) {
    put(1u << 5u, 6u); put(0u, 2u);
    for (var c = 0u; c < 3u; c++) {
      let e = T(SOLID5_AT + solid[c]);
      put(e & 255u, 7u); put((e >> 8u) & 255u, 7u);
    }
    put(solid[3], 8u); put(solid[3], 8u);
    put(1u, 1u);
    for (var t = 1u; t < 16u; t++) { put(1u, 2u); }
  } else {
    let p = select(0u, 1u, err1 < err0);
    put(1u << 6u, 7u);
    for (var c = 0u; c < 4u; c++) {
      let e = T(SOLID6_AT + p * 256u + solid[c]);
      put(e & 255u, 7u); put((e >> 8u) & 255u, 7u);
    }
    put(p, 1u); put(p, 1u);
    put(5u, 3u);
    for (var t = 1u; t < 16u; t++) { put(5u, 4u); }
  }
}

fn dualPlaneBC7() {
  let sel = selector;
  if (sel < 3u) {
    var v = E[sel]; E[sel] = E[3]; E[3] = v;
    v = E[4u + sel]; E[4u + sel] = E[7]; E[7] = v;
  }
  for (var t = 0u; t < 32u; t++) {
    W[t] = select(wts[t], T(CONV12_AT + wts[t]), mode == 13u);
  }
  let flipColour = W[0] > 1u;
  let flipAlpha = W[1] > 1u;
  put(1u << 5u, 6u); put((sel + 1u) & 3u, 2u);
  for (var c = 0u; c < 3u; c++) {
    let lo = (E[c] * 127u + 127u) / 255u;
    let hi = (E[4u + c] * 127u + 127u) / 255u;
    if (flipColour) { put(hi, 7u); put(lo, 7u); } else { put(lo, 7u); put(hi, 7u); }
  }
  if (flipAlpha) { put(E[7], 8u); put(E[3], 8u); } else { put(E[3], 8u); put(E[7], 8u); }
  put(select(W[0], 3u - W[0], flipColour), 1u);
  for (var t = 1u; t < 16u; t++) { put(select(W[t * 2u], 3u - W[t * 2u], flipColour), 2u); }
  put(select(W[1], 3u - W[1], flipAlpha), 1u);
  for (var t = 1u; t < 16u; t++) { put(select(W[t * 2u + 1u], 3u - W[t * 2u + 1u], flipAlpha), 2u); }
}

fn planIndex() -> u32 {
  if (mode == 1u) { return 1u; }
  if (T(SUBSETS_AT + mode) == 1u) { return 0u; }
  if (mode == 3u) { return 32u + pattern; }
  if (mode == 7u) { return 43u + pattern; }
  return 2u + pattern;
}

fn f32At(i: u32) -> f32 { return bitcast<f32>(T(i)); }

fn bc7Block() {
  if (mode == 8u) { solidBC7(); return; }
  rgbaEndpoints();
  let bc7 = T(BC7_MODE_AT + mode);
  if (bc7 == 5u) { dualPlaneBC7(); return; }

  let plan = PLANS_AT + planIndex() * 4u;
  let head = T(plan);
  let bc7Pattern = head & 255u;
  let subsets = (head >> 8u) & 255u;
  let map = T(plan + 1u);
  let anchors = T(plan + 2u);
  let mask = T(plan + 3u);
  let bits = T(BC7_BITS_AT + mode);
  let conv = T(CONV_AT_AT + mode);
  let maxIndex = (1u << bits) - 1u;
  for (var t = 0u; t < 16u; t++) { W[t] = select(T(conv + wts[t]), wts[t], conv == 0xffffffffu); }

  let comps = select(3u, 4u, bc7 == 6u || bc7 == 7u);
  if (bc7 == 2u) {
    for (var s = 0u; s < subsets; s++) {
      let o = ((head >> (16u + 4u * s)) & 15u) * 8u;
      for (var c = 0u; c < 8u; c++) { Q[s * 8u + c] = (E[o + c] * 31u + 127u) / 255u; }
    }
  } else {
    var qAt = Q7_AT; var uniqueAt = UNIQUE7_AT; var sharedAt = SHARED7_AT;
    if (bc7 == 1u) { qAt = Q6_AT; uniqueAt = UNIQUE6_AT; sharedAt = SHARED6_AT; }
    if (bc7 == 7u) { qAt = Q5_AT; uniqueAt = UNIQUE5_AT; sharedAt = SHARED5_AT; }
    for (var s = 0u; s < subsets; s++) {
      let o = ((head >> (16u + 4u * s)) & 15u) * 8u;
      if (bc7 == 1u) {
        var e0 = 0.0;
        var e1 = 0.0;
        for (var c = 0u; c < 3u; c++) {
          e0 = e0 + (f32At(sharedAt + E[o + c]) + f32At(sharedAt + E[o + 4u + c]));
          e1 = e1 + (f32At(sharedAt + 256u + E[o + c]) + f32At(sharedAt + 256u + E[o + 4u + c]));
        }
        let p = select(0u, 1u, e1 < e0);
        PB[s * 2u] = p; PB[s * 2u + 1u] = p;
        for (var c = 0u; c < 8u; c++) { Q[s * 8u + c] = T(qAt + p * 256u + E[o + c]); }
      } else {
        for (var h = 0u; h < 2u; h++) {
          let e = o + h * 4u;
          var e0 = 0.0;
          var e1 = 0.0;
          for (var c = 0u; c < comps; c++) {
            e0 = e0 + f32At(uniqueAt + E[e + c]);
            e1 = e1 + f32At(uniqueAt + 256u + E[e + c]);
          }
          let p = select(0u, 1u, e1 < e0);
          PB[s * 2u + h] = p;
          for (var c = 0u; c < 4u; c++) { Q[s * 8u + h * 4u + c] = T(qAt + p * 256u + E[e + c]); }
        }
      }
    }
  }
  for (var s = 0u; s < subsets; s++) {
    if (W[(anchors >> (8u * s)) & 255u] <= (maxIndex >> 1u)) { continue; }
    for (var c = 0u; c < 4u; c++) { let v = Q[s * 8u + c]; Q[s * 8u + c] = Q[s * 8u + 4u + c]; Q[s * 8u + 4u + c] = v; }
    let p = PB[s * 2u]; PB[s * 2u] = PB[s * 2u + 1u]; PB[s * 2u + 1u] = p;
    for (var t = 0u; t < 16u; t++) { if (((map >> (2u * t)) & 3u) == s) { W[t] = maxIndex - W[t]; } }
  }

  put(1u << bc7, bc7 + 1u);
  if (subsets > 1u) { put(bc7Pattern, select(6u, 4u, bc7 == 0u)); }
  var cbits = 7u;
  if (bc7 == 2u || bc7 == 7u) { cbits = 5u; } else if (bc7 == 1u) { cbits = 6u; }
  for (var c = 0u; c < comps; c++) {
    for (var s = 0u; s < subsets; s++) { put(Q[s * 8u + c], cbits); put(Q[s * 8u + 4u + c], cbits); }
  }
  if (bc7 == 1u) {
    for (var s = 0u; s < subsets; s++) { put(PB[s * 2u], 1u); }
  } else if (bc7 != 2u) {
    for (var s = 0u; s < subsets; s++) { put(PB[s * 2u], 1u); put(PB[s * 2u + 1u], 1u); }
  }
  for (var t = 0u; t < 16u; t++) { put(W[t], bits - ((mask >> t) & 1u)); }
}

// ---- ASTC 4x4 (scene/uastc.js astcBlock)
fn putISE(count: u32, range: u32) {
  let bits = T(RANGE_BITS_AT + range);
  let trits = T(RANGE_TRITS_AT + range);
  let quints = T(RANGE_QUINTS_AT + range);
  let mask = (1u << bits) - 1u;
  if (trits + quints == 0u) {
    for (var i = 0u; i < count; i++) { put(raw[i], bits); }
    return;
  }
  var per = 3u; var base = 5u; var splitAt = QUINT_SPLIT_AT; var codes = QUINT_CODES_AT; var packedBits = 7u;
  if (trits == 1u) { per = 5u; base = 3u; splitAt = TRIT_SPLIT_AT; codes = TRIT_CODES_AT; packedBits = 8u; }
  let end = bitpos + count * bits + (count * packedBits + per - 1u) / per;
  for (var g = 0u; g < count; g += per) {
    var digits = 0u;
    for (var i = i32(min(per, count - g)) - 1; i >= 0; i--) { digits = digits * base + (raw[g + u32(i)] >> bits); }
    var packed = T(codes + digits);
    for (var i = 0u; i < per && bitpos < end; i++) {
      if (g + i < count) { put(raw[g + i] & mask, bits); }
      let split = T(splitAt + i);
      let n = min(split, end - bitpos);
      put(packed & ((1u << n) - 1u), n);
      packed = packed >> split;
    }
  }
}

fn astcBlock() {
  if (mode == 8u) {
    put(0xfcu, 8u); put(0xfdu, 8u);
    for (var i = 2u; i < 8u; i++) { put(0xffu, 8u); }
    for (var c = 0u; c < 4u; c++) { put(solid[c], 8u); put(solid[c], 8u); }
    return;
  }
  let comps = T(COMPS_AT + mode);
  let subsets = T(SUBSETS_AT + mode);
  let planes = T(PLANES_AT + mode);
  let wbits = T(WEIGHT_BITS_AT + mode);
  if (comps >= 3u) {
    let maxIndex = (1u << wbits) - 1u;
    let map = uastcPattern();
    for (var s = 0u; s < subsets; s++) {
      let o = s * comps * 2u;
      if (ends[o] + ends[o + 2u] + ends[o + 4u] <= ends[o + 1u] + ends[o + 3u] + ends[o + 5u]) { continue; }
      for (var c = 0u; c < comps; c++) { let lo = raw[o + c * 2u]; raw[o + c * 2u] = raw[o + c * 2u + 1u]; raw[o + c * 2u + 1u] = lo; }
      for (var t = 0u; t < 16u; t++) {
        if (subsets > 1u && ((map >> (2u * t)) & 3u) != s) { continue; }
        for (var p = 0u; p < planes; p++) { wts[t * planes + p] = maxIndex - wts[t * planes + p]; }
      }
    }
  }
  let blockMode = T(BLOCK_MODE_AT + wbits) | select(0u, 0x400u, planes == 2u);
  put(blockMode & 255u, 8u);
  put(blockMode >> 8u, 3u);
  put(subsets - 1u, 2u);
  let cem = T(CEM_AT + comps);
  if (subsets == 1u) {
    put(cem, 4u);
  } else {
    var seed = T(SEEDS2_AT + pattern);
    if (mode == 3u) { seed = T(SEEDS3_AT + pattern); } else if (mode == 7u) { seed = T(SEEDS7_AT + pattern); }
    put(seed & 255u, 8u);
    put(seed >> 8u, 2u);
    put(cem << 2u, 6u);
  }
  putISE(comps * 2u * subsets, T(ENDPOINT_RANGE_AT + mode));
  let weightBits = 16u * planes * wbits;
  if (planes == 2u) { bitpos = 126u - weightBits; put(selector, 2u); }
  // The weights from the block's top bit down: packed forwards, then each word bit-reversed into place.
  var stream: array<u32, 5>;
  var at = 0u;
  for (var i = 0u; i < 16u * planes; i++) {
    let w = at >> 5u;
    let s = at & 31u;
    stream[w] = stream[w] | (wts[i] << s);
    if (s + wbits > 32u) { stream[w + 1u] = stream[w + 1u] | (wts[i] >> (32u - s)); }
    at += wbits;
  }
  for (var k = 0u; k < 4u; k++) { acc[3u - k] = acc[3u - k] | reverseBits(stream[k]); }
}

// ---- entry points: one invocation a block
fn blockAt(id: vec3<u32>) -> u32 { return id.y * levelInfo[2] + id.x; }

fn outputWord(b: u32) -> u32 {
  var l = 0u;
  loop {
    if (l + 1u >= levelInfo[1] || b < levelInfo[4u + (l + 1u) * 4u]) { break; }
    l++;
  }
  let local = b - levelInfo[4u + l * 4u];
  let across = levelInfo[5u + l * 4u];
  return levelInfo[6u + l * 4u] + (local / across) * levelInfo[7u + l * 4u] + (local % across) * 4u;
}

fn load(b: u32) {
  blk[0] = src[b * 4u]; blk[1] = src[b * 4u + 1u]; blk[2] = src[b * 4u + 2u]; blk[3] = src[b * 4u + 3u]; blk[4] = 0u;
}

fn store(o: u32) {
  dst[o] = acc[0]; dst[o + 1u] = acc[1]; dst[o + 2u] = acc[2]; dst[o + 3u] = acc[3];
}

@compute @workgroup_size(${WORKGROUP})
fn bc7(@builtin(global_invocation_id) id: vec3<u32>) {
  let b = blockAt(id);
  if (b >= levelInfo[0]) { return; }
  load(b);
  begin();
  if (unpack()) { bc7Block(); }
  store(outputWord(b));
}

@compute @workgroup_size(${WORKGROUP})
fn astc(@builtin(global_invocation_id) id: vec3<u32>) {
  let b = blockAt(id);
  if (b >= levelInfo[0]) { return; }
  load(b);
  begin();
  if (unpack()) { astcBlock(); }
  store(outputWord(b));
}
`;

/** Per device: the tables uploaded and the pipelines building, once. */
const transcoders = new WeakMap();

export class UastcGPU {
  /** The device's transcoder, made on first use (off means off: a page with no KTX2 never builds it). */
  static for(rhi) {
    let t = transcoders.get(rhi.device);
    if (!t) transcoders.set(rhi.device, (t = new UastcGPU(rhi)));
    return t;
  }

  constructor(rhi) {
    this.rhi = rhi;
    const device = rhi.device;
    const { words, at } = packedTables();
    this.tables = createBuffer(rhi, { label: 'uastc-tables', usage: GPUBufferUsage.STORAGE, data: words });
    this.layout = device.createBindGroupLayout({
      label: 'uastc',
      entries: [0, 1, 2, 3].map((binding) => ({
        binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: binding === 1 ? 'storage' : 'read-only-storage' },
      })),
    });
    const layout = createPipelineLayout(device, { 0: this.layout }, 'uastc');
    const shader = compileShader(device, shaderSource(at), 'uastc.wgsl');
    const pipelines = sharedPipelines(device);
    this.pipelines = {
      bc7: shader.then((s) => pipelines.computeAsync({ label: 'uastc-bc7', layout, shader: s, entry: 'bc7' })),
      astc: shader.then((s) => pipelines.computeAsync({ label: 'uastc-astc', layout, shader: s, entry: 'astc' })),
    };
  }

  /**
   * An image's UASTC levels ({ width, height, levels: [{ width, height, data }] })
   * transcoded to `target` ('bc7' or 'astc'). Resolves, once the work is on the
   * queue, to { buffer, levels }: copy(texture) puts every level into a texture
   * of the image's size, format and level count; destroy() frees the buffer.
   * `buffer` and `levels` (each one's offset and bytesPerRow) are the blocks as
   * written, for a test to read.
   */
  async transcode(image, target) {
    const pipeline = await this.pipelines[target];
    const { rhi } = this;
    const device = rhi.device;
    const levels = image.levels;
    // Input: the levels' blocks one after another. Output: each level's rows 256-byte aligned.
    let blocks = 0, out = 0;
    const header = [0, levels.length, 0, 0];
    const placed = levels.map(({ width, height }) => {
      const across = Math.ceil(width / 4), down = Math.ceil(height / 4);
      const bytesPerRow = Math.ceil((across * 16) / 256) * 256;
      const level = { width: across * 4, height: down * 4, offset: out, bytesPerRow };
      header.push(blocks, across, out / 4, bytesPerRow / 4);
      blocks += across * down;
      out += bytesPerRow * down;
      return level;
    });
    const groups = Math.ceil(blocks / WORKGROUP);
    const wide = Math.min(groups, rhi.limits.maxComputeWorkgroupsPerDimension ?? 65535);
    header[0] = blocks;
    header[2] = wide * WORKGROUP;

    const input = new Uint8Array(blocks * 16);
    let at = 0;
    for (const level of levels) { input.set(level.data, at); at += level.data.length; }
    const src = createBuffer(rhi, { label: 'uastc-in', usage: GPUBufferUsage.STORAGE, data: input });
    const dst = createBuffer(rhi, { label: 'uastc-out', size: out, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const layout = createBuffer(rhi, { label: 'uastc-layout', usage: GPUBufferUsage.STORAGE, data: Uint32Array.from(header) });
    const group = device.createBindGroup({
      layout: this.layout,
      entries: [src, dst, this.tables, layout].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const encoder = device.createCommandEncoder({ label: 'uastc' });
    const pass = encoder.beginComputePass({ label: `uastc-${target}` });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(wide, Math.ceil(groups / wide));
    pass.end();
    rhi.queue.submit([encoder.finish()]);
    // Destroyed now, freed once the dispatch above is done with them.
    src.destroy();
    layout.destroy();
    return {
      buffer: dst,
      levels: placed,
      copy(texture) {
        const copies = device.createCommandEncoder({ label: 'uastc-copy' });
        placed.forEach(({ width, height, offset, bytesPerRow }, mipLevel) => {
          copies.copyBufferToTexture({ buffer: dst, offset, bytesPerRow }, { texture, mipLevel }, [width, height]);
        });
        rhi.queue.submit([copies.finish()]);
      },
      destroy() { dst.destroy(); },
    };
  }
}
