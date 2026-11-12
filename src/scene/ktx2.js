// The KTX2 container (Khronos KTX File Format 2.0): a texture's mip levels and
// what they hold, and (transcodeKTX2) its Basis levels in a format a GPU
// samples. No GPU here -- which format, and the upload, are the loader's.
//
// A level is kept as it will be transcoded or uploaded: zstd-packed levels
// unpacked (core/zstd.js), BasisLZ ones (ETC1S) left as they are with the
// global data their decoding needs. The Data Format Descriptor says which
// Basis mode a VK_FORMAT_UNDEFINED file holds, whether it is sRGB, and which
// channels it carries.

import { zstdDecompress } from '../core/zstd.js';
import { uastcToRGBA, uastcToBC7, uastcToASTC } from './uastc.js';

const IDENTIFIER = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A];

/** supercompressionScheme values. */
export const KTX2_NONE = 0;
export const KTX2_BASISLZ = 1;
export const KTX2_ZSTD = 2;
export const KTX2_ZLIB = 3;

/** Data Format Descriptor colour models: the two Basis modes. */
const MODEL_ETC1S = 163;
const MODEL_UASTC = 166;
/** transferFunction: sRGB. */
const TRANSFER_SRGB = 2;
/** UASTC channel ids (KHR_DF_CHANNEL_UASTC_*). */
const UASTC_CHANNELS = { 0: 'rgb', 3: 'rgba', 4: 'rrr', 5: 'rrrg', 6: 'rg' };

const fail = (why) => { throw new Error(`ktx2: ${why}`); };

/**
 * Read a KTX2 file. Returns { vkFormat, width, height, depth, layers, faces,
 * levels: [{ width, height, data }] (largest first), basis: 'uastc' | 'etc1s'
 * | null, srgb, channels (UASTC: 'rgb' | 'rgba' | 'rrr' | 'rrrg' | 'rg'),
 * premultiplied, supercompression, globalData (BasisLZ's, else null) }.
 */
export function readKTX2(bytes) {
  if (!(bytes instanceof Uint8Array)) bytes = new Uint8Array(bytes);
  if (bytes.length < 80 || IDENTIFIER.some((b, i) => bytes[i] !== b)) fail('not a KTX2 file');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const u32 = (at) => view.getUint32(at, true);
  const u64 = (at) => {
    const high = view.getUint32(at + 4, true);
    if (high !== 0) fail('an offset past 4 GiB');
    return view.getUint32(at, true);
  };
  const vkFormat = u32(12);
  const width = u32(20), height = u32(24), depth = u32(28);
  const layers = u32(32), faces = u32(36), levelCount = Math.max(1, u32(40));
  const supercompression = u32(44);
  if (width === 0) fail('a width of 0');
  if (faces !== 1 && faces !== 6) fail(`${faces} faces; a texture has 1, a cube 6`);
  if (supercompression > KTX2_ZLIB) fail(`supercompression scheme ${supercompression}, which is not one KTX2 defines`);
  if (supercompression === KTX2_ZLIB) fail('ZLIB supercompression, which is not read; zstd is');

  const dfdOffset = u32(48), dfdLength = u32(52);
  const sgdOffset = u64(64), sgdLength = u64(72);
  const levelIndex = 80;
  if (levelIndex + levelCount * 24 > bytes.length) fail('a level index past the end of the file');

  // The basic Data Format Descriptor block: its colour model, transfer, flags and first sample.
  let basis = null, srgb = false, premultiplied = false, channels = null;
  if (dfdLength >= 28 && dfdOffset + dfdLength <= bytes.length) {
    const block = dfdOffset + 4;   // after dfdTotalSize
    const model = bytes[block + 8], transfer = bytes[block + 10], flags = bytes[block + 11];
    srgb = transfer === TRANSFER_SRGB;
    premultiplied = (flags & 1) !== 0;
    if (model === MODEL_UASTC) {
      basis = 'uastc';
      const channel = bytes[block + 24 + 3] & 15;
      channels = UASTC_CHANNELS[channel] ?? fail(`UASTC channel id ${channel}, which is not one the spec defines`);
    } else if (model === MODEL_ETC1S) {
      basis = 'etc1s';
    }
  }
  if (vkFormat === 0 && basis === null) fail('VK_FORMAT_UNDEFINED with neither Basis UASTC nor ETC1S in its descriptor');
  if (basis === 'etc1s' && supercompression !== KTX2_BASISLZ) fail('ETC1S without BasisLZ supercompression');
  if (supercompression === KTX2_BASISLZ && basis !== 'etc1s') fail('BasisLZ supercompression on something other than ETC1S');

  const levels = [];
  for (let i = 0; i < levelCount; i++) {
    const at = levelIndex + i * 24;
    const offset = u64(at), length = u64(at + 8), uncompressed = u64(at + 16);
    if (offset + length > bytes.length) fail(`level ${i} past the end of the file`);
    const stored = bytes.subarray(offset, offset + length);
    const data = supercompression === KTX2_ZSTD ? zstdDecompress(stored, uncompressed) : stored;
    levels.push({ width: Math.max(1, width >> i), height: Math.max(1, (height || 1) >> i), data });
  }
  const globalData = supercompression === KTX2_BASISLZ && sgdLength > 0
    ? bytes.subarray(sgdOffset, sgdOffset + sgdLength) : null;
  return {
    vkFormat, width, height: Math.max(1, height), depth, layers, faces, levels,
    basis, srgb, channels, premultiplied, supercompression, globalData,
  };
}

/**
 * A Basis UASTC KTX2 file's levels for the GPU: `target` 'astc' (ASTC 4x4,
 * nothing lost), 'bc7', or 'rgba' (RGBA8). With `gpu`, a file that would
 * become ASTC or BC7 is left as UASTC (`uastc: true`) for the GPU to
 * transcode (render/uastcGPU.js); only its zstd is undone here. Returns { width, height, target,
 * levels: [{ width, height, data }] }, `target` being what was written:
 * RGBA8 instead of a block format when the file has a single level, since a
 * block-compressed texture's mips cannot be rendered on the GPU (RGBA8 can),
 * or when its size is not whole 4x4 blocks, which WebGPU asks of one.
 * Every level the file has comes back; with one, the caller makes the mips.
 */
export function transcodeKTX2(bytes, target, { gpu = false } = {}) {
  const file = readKTX2(bytes);
  if (file.basis === 'etc1s') fail('ETC1S (BasisLZ) is not read yet, only UASTC');
  if (file.basis !== 'uastc') fail('not Basis Universal: KHR_texture_basisu holds UASTC or ETC1S');
  if (file.depth > 1 || file.layers > 1 || file.faces !== 1) fail('not a 2D image (a volume, an array or a cube)');
  const { width, height, levels } = file;
  if (levels.length === 1 || width % 4 || height % 4) target = 'rgba';
  if (gpu && target !== 'rgba') return { width, height, target, uastc: true, levels };
  const transcode = target === 'astc' ? uastcToASTC : target === 'bc7' ? uastcToBC7 : uastcToRGBA;
  return { width, height, target, levels: levels.map((l) => ({ width: l.width, height: l.height, data: transcode(l.data, l.width, l.height) })) };
}
