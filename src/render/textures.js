// Uploading decoded glTF images to the GPU.
//
// The layer boundary: scene/gltf/images.js produces ImageBitmaps and knows
// nothing about a GPU; this file turns them into textures and knows nothing
// about glTF's container format. Either half can be replaced without the other
// noticing.

import {
  createTexture, createTexture2D, uploadImage, generateMipmaps, linearSampler,
} from '../rhi/texture.js';
import {
  materialTextureSlots, samplerDescriptor, textureImageIndex, textureSamplerIndex, imageColorSpaces,
} from '../scene/gltf/images.js';
// The GPU transcoder and its tables (render/uastcGPU.js, scene/uastc.js, about
// 60 KB) are loaded by the first UASTC image, not by every page: off means off.
let uastcGPU = null;
const gpuTranscoder = () => (uastcGPU ??= import('./uastcGPU.js'));

/** WebGPU's formats for transcodeKTX2's targets, sRGB by appending '-srgb'. */
const KTX2_FORMATS = { astc: 'astc-4x4-unorm', bc7: 'bc7-rgba-unorm', rgba: 'rgba8unorm' };

/**
 * A transcoded KTX2 image's levels, as the file made them. A block format's
 * rows are rows of 4x4 blocks, 16 bytes each, and a level smaller than a block
 * is still one: the copy is the level's size in whole blocks. Both kinds come
 * to 4 bytes a texel of that size a row.
 */
function levelsTexture(rhi, image, srgb, label) {
  return createTexture(rhi, {
    label,
    size: [image.width, image.height],
    format: KTX2_FORMATS[image.target] + (srgb ? '-srgb' : ''),
    mipLevelCount: image.levels.length,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
}
function uploadLevels(rhi, image, srgb, label) {
  const block = image.target !== 'rgba';
  const texture = levelsTexture(rhi, image, srgb, label);
  image.levels.forEach(({ width, height, data }, mipLevel) => {
    if (block) { width = Math.ceil(width / 4) * 4; height = Math.ceil(height / 4) * 4; }
    rhi.queue.writeTexture({ texture, mipLevel }, data, { bytesPerRow: width * 4 }, [width, height]);
  });
  return texture;
}

/** A one-level KTX2 image, as RGBA8 (transcodeKTX2), with mips made here as a PNG's are. */
function uploadOneLevel(rhi, image, srgb, label, mipOptions) {
  const texture = createTexture2D(rhi, { label, width: image.width, height: image.height, srgb, mipmapped: true });
  rhi.queue.writeTexture({ texture }, image.levels[0].data, { bytesPerRow: image.width * 4 }, [image.width, image.height]);
  generateMipmaps(rhi, texture, mipOptions);
  return texture;
}

/**
 * A texture from a transcoded KTX2 image (transcodeKTX2's result, with `gpu`):
 * UASTC transcoded here on the GPU, block levels uploaded as the file made
 * them, one RGBA8 level mipped. Resolves once its copies are on the queue.
 */
export async function ktx2Texture(rhi, image, { srgb, label, mipOptions = {} }) {
  if (image.uastc) {
    const texture = levelsTexture(rhi, image, srgb, label);
    const { UastcGPU } = await gpuTranscoder();
    const made = await UastcGPU.for(rhi).transcode(image, image.target);
    made.copy(texture);
    made.destroy();
    return texture;
  }
  if (image.levels.length > 1) return uploadLevels(rhi, image, srgb, label);
  return uploadOneLevel(rhi, image, srgb, label, mipOptions);
}

/**
 * Uploads on demand and caches, so an image shared by twenty materials is
 * uploaded once.
 *
 * The cache key is `imageIndex:srgb`, not just the image index. The same file
 * used as both a base colour map and a roughness map genuinely needs two GPU
 * textures with different formats -- one that decodes on sample and one that
 * does not.
 */
export class GLTFTextures {
  constructor(rhi, json, bitmaps) {
    this.rhi = rhi;
    this.json = json;
    this.bitmaps = bitmaps;
    this._textures = new Map();
    this._samplers = new Map();
    this.uploaded = 0;
    /** KTX2 images transcoding on the GPU, until their levels are copied in. */
    this._pending = [];
    /** Their transcoded blocks on the GPU, freed with the images or with this. */
    this._transcodes = [];
    /** Images whose alpha is coverage, which their mips average by (see imageColorSpaces). */
    this._coverage = new Set([...imageColorSpaces(json)].filter(([, spaces]) => spaces.coverage).map(([image]) => image));
  }

  /** {baseColor, normal, metallicRoughness, occlusion, emissive, sampler} */
  texturesFor(material) {
    const result = {};
    for (const { slot, texture, srgb } of materialTextureSlots(material)) {
      const gpu = this._texture(texture, srgb);
      if (gpu) result[slot] = gpu;
    }
    result.sampler = this._samplerFor(material);
    return result;
  }

  /**
   * Make an image's textures now, as its decode lands, in every colour space
   * `spaces` says it is read in -- so they upload while other images are
   * still arriving, not all at once after the last.
   */
  upload(imageIndex, bitmap, spaces) {
    if (spaces?.srgb) this._make(imageIndex, bitmap, true);
    if (spaces?.linear) this._make(imageIndex, bitmap, false);
  }

  _texture(textureIndex, srgb) {
    const imageIndex = textureImageIndex(this.json, textureIndex);
    if (imageIndex < 0) return null;

    const key = `${imageIndex}:${srgb ? 1 : 0}`;
    const made = this._textures.get(key);
    if (made) return made;

    const bitmap = this.bitmaps[imageIndex];
    if (!bitmap) return null;            // decode failed; the factor stands in
    return this._make(imageIndex, bitmap, srgb);
  }

  _make(imageIndex, bitmap, srgb) {
    const key = `${imageIndex}:${srgb ? 1 : 0}`;
    let texture = this._textures.get(key);
    if (texture) return texture;

    const label = `gltf-image-${imageIndex}${srgb ? '-srgb' : ''}`;
    if (bitmap.uastc) {
      // Transcoded once on the GPU, however many textures (sRGB, linear) take it.
      texture = levelsTexture(this.rhi, bitmap, srgb, label);
      if (!bitmap.transcoded) {
        bitmap.transcoded = gpuTranscoder().then(({ UastcGPU }) => UastcGPU.for(this.rhi).transcode(bitmap, bitmap.target));
        this._transcodes.push(bitmap.transcoded);
      }
      const made = texture;
      this._pending.push(bitmap.transcoded.then((gpu) => { if (!this._destroyed) gpu.copy(made); }));
    } else if (bitmap.levels?.length > 1) {
      texture = uploadLevels(this.rhi, bitmap, srgb, label);
    } else {
      // Mips are averaged through an -srgb view when the format is sRGB, so the
      // filtering happens in linear light without any manual conversion. By
      // coverage only where alpha is coverage: elsewhere a clear block went black.
      const mipOptions = { coverage: srgb && this._coverage.has(imageIndex) };
      if (bitmap.levels) texture = uploadOneLevel(this.rhi, bitmap, srgb, label, mipOptions);
      else {
        texture = createTexture2D(this.rhi, { label, width: bitmap.width, height: bitmap.height, srgb, mipmapped: true });
        uploadImage(this.rhi, texture, bitmap);
        generateMipmaps(this.rhi, texture, mipOptions);
      }
    }

    this._textures.set(key, texture);
    this.uploaded++;
    return texture;
  }

  /**
   * One sampler per material, taken from its base colour texture.
   *
   * ponytail: glTF allows a different sampler per texture. Honouring that means
   * five sampler bindings in group 2 instead of one. Assets that mix wrap modes
   * across a single material's maps are rare enough that this has not been
   * worth the extra slots -- widen the bind group layout if one shows up.
   */
  _samplerFor(material) {
    const baseColorTexture = material.textures?.baseColor ?? -1;
    const samplerIndex = textureSamplerIndex(this.json, baseColorTexture);
    if (samplerIndex < 0) return linearSampler(this.rhi);

    let sampler = this._samplers.get(samplerIndex);
    if (!sampler) {
      const descriptor = samplerDescriptor(this.json.samplers?.[samplerIndex], this.rhi.anisotropy);
      sampler = this.rhi.device.createSampler({ label: `gltf-sampler-${samplerIndex}`, ...descriptor });
      this._samplers.set(samplerIndex, sampler);
    }
    return sampler;
  }

  /**
   * Drop the decoded images once everything that wanted one has been uploaded.
   *
   * The GPU textures are unaffected -- they already hold their own copy. This
   * only releases the ImageBitmaps, whose native memory the collector would
   * otherwise free whenever it felt like it.
   */
  /** Once every KTX2 image transcoding on the GPU has its copies on the queue. */
  async ready() {
    await Promise.all(this._pending);
  }

  releaseBitmaps() {
    for (const bitmap of this.bitmaps) bitmap?.close?.();
    this._freeTranscodes();
    this.bitmaps = [];
  }

  /** The GPU's transcoded blocks, once copied in: the copies keep them until they run. */
  _freeTranscodes() {
    for (const transcode of this._transcodes) transcode.then((gpu) => gpu.destroy(), () => {});
    this._transcodes = [];
  }

  destroy() {
    this._destroyed = true;
    this._freeTranscodes();
    for (const texture of this._textures.values()) texture.destroy();
    this._textures.clear();
  }
}
