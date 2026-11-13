// Outlines (node.setOutline): a line round what is under a node, drawn in
// screen space so it holds on flat-shaded low-poly meshes, whose split normals
// tear an inverted hull open at every edge.
//
// Two passes. The mask: each outlined mesh drawn alone into an r8uint
// texture as its outline's slot, by the shadow shader's vertex paths (which
// skin and morph as the forward pass does), tested against the scene's depth
// unless the outline shows through. The composite: over the scene's colour,
// a pixel outside every mask that has a masked one within that outline's
// width takes its colour, fading over the last half pixel. Scissored to the
// screen rectangle of what is outlined.

import { compileShader } from '../rhi/shader.js';
import { createBuffer } from '../rhi/buffer.js';
import { createPipelineLayout, GROUP_FRAME, GROUP_DRAW } from '../rhi/bindgroups.js';
import { DEPTH_FORMAT } from '../rhi/device.js';
import { POSITION_BUFFER_LAYOUT, SKIN_BUFFER_LAYOUT } from './vertex.js';
import { NO_PARENT } from '../scene/transform.js';
import { BATCH_BYTES } from './gpudriven.js';

/** Distinct outline looks a frame can draw: the mask is 8 bits, and 0 is none. */
export const OUTLINE_STYLES = 255;
/** The widest outline, in pixels: the composite reads (2w + 1)^2 of the mask a pixel. */
export const MAX_OUTLINE_WIDTH = 8;
const MASK_FORMAT = 'r8uint';
const STYLE_FLOATS = 8;

const COMPOSITE = /* wgsl */ `
struct Style {
  color : vec4<f32>,
  // x: width, in this view's pixels.
  width : vec4<f32>,
};
struct Styles {
  // x: how far any outline reaches, in whole pixels.
  reach : vec4<f32>,
  style : array<Style, ${OUTLINE_STYLES + 1}>,
};
@group(0) @binding(0) var mask : texture_2d<u32>;
@group(0) @binding(1) var<uniform> styles : Styles;

@vertex
fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4<f32> {
  return vec4<f32>(vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) p : vec4<f32>) -> @location(0) vec4<f32> {
  let c = vec2<i32>(p.xy);
  if (textureLoad(mask, c, 0).r != 0u) { discard; }
  let last = vec2<i32>(textureDimensions(mask)) - 1;
  let r = i32(styles.reach.x);
  var best = 0u;
  var nearest = 1e9;
  for (var y = -r; y <= r; y++) {
    for (var x = -r; x <= r; x++) {
      let s = textureLoad(mask, clamp(c + vec2<i32>(x, y), vec2<i32>(0), last), 0).r;
      if (s == 0u) { continue; }
      let d = f32(x * x + y * y);
      let w = styles.style[s].width.x + 0.5;
      if (d < nearest && d <= w * w) { best = s; nearest = d; }
    }
  }
  if (best == 0u) { discard; }
  let style = styles.style[best];
  return vec4<f32>(style.color.rgb, style.color.a * clamp(style.width.x + 0.5 - sqrt(nearest), 0.0, 1.0));
}
`;

/**
 * Each renderable under an outlined node, with the nearest such node's look:
 * [renderable, style] pairs into `out`, and the looks, deduplicated, into
 * `styles` (a Map from key to slot, from 1). Hidden ones are left out.
 */
export function outlinedRenderables(scene, out, styles) {
  const outlines = scene.outlines, parent = scene.transforms.parent;
  let n = 0;
  for (let i = 0; i < scene.renderableCount; i++) {
    if (scene.renderableHidden[i] === 1) continue;
    let slot = scene.renderableMatrixSlot[i], look;
    while (slot !== NO_PARENT && (look = outlines.get(slot)) === undefined) slot = parent[slot];
    if (look === undefined) continue;
    let style = styles.get(look.key);
    if (style === undefined) {
      if (styles.size === OUTLINE_STYLES) continue;
      styles.set(look.key, (style = { slot: styles.size + 1, look }));
    }
    out[n++] = i;
    out[n++] = style.slot;
  }
  return n / 2;
}

export class OutlinePass {
  static async create(rhi, pipelines, shadows) {
    const device = rhi.device;
    const descriptors = {};
    for (const skinned of [false, true]) {
      for (const through of [false, true]) {
        descriptors[`${skinned ? 1 : 0}${through ? 1 : 0}`] = {
          label: `outline-mask${skinned ? '-skinned' : ''}${through ? '-through' : ''}`,
          layout: shadows.pipelineLayout,
          shader: shadows.shader,
          vertexEntry: skinned ? 'vsOutlineSkinned' : 'vsOutline',
          fragmentEntry: 'fsOutline',
          buffers: skinned ? [POSITION_BUFFER_LAYOUT, SKIN_BUFFER_LAYOUT] : [POSITION_BUFFER_LAYOUT],
          targets: [{ format: MASK_FORMAT }],
          primitive: { topology: 'triangle-list', cullMode: 'none' },
          // Reverse-Z: nearer is greater. A little bias toward the eye, so the
          // surface the forward pass drew passes against its own depth.
          depth: through
            ? { format: DEPTH_FORMAT, depthCompare: 'always', depthWriteEnabled: false }
            : { format: DEPTH_FORMAT, depthCompare: 'greater-equal', depthWriteEnabled: false, depthBias: 4, depthBiasSlopeScale: 1 },
        };
      }
    }
    const shader = await compileShader(device, COMPOSITE, 'outline.wgsl');
    const compositeLayout = device.createBindGroupLayout({
      label: 'outline',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'uint' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    return new OutlinePass(rhi, pipelines, shadows, descriptors, { shader, compositeLayout });
  }

  constructor(rhi, pipelines, shadows, descriptors, { shader, compositeLayout }) {
    this.rhi = rhi;
    this.pipelines = pipelines;
    this.shadows = shadows;
    this.descriptors = descriptors;
    this.compositeLayout = compositeLayout;
    this.compositeDescriptors = new Map();
    this._compositeShader = shader;
    this.alignment = rhi.limits.minUniformBufferOffsetAlignment;
    this._pairs = new Uint32Array(0);
    this._styles = new Map();
    this._looks = [];
    this._styleData = new Float32Array(4 + (OUTLINE_STYLES + 1) * STYLE_FLOATS);
    this._styleBuffer = createBuffer(rhi, { label: 'outline-styles', size: this._styleData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._camera = createBuffer(rhi, { label: 'outline-camera', size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._items = null;
    this._batches = null;
    this._batchData = new Uint32Array(0);
    this._capacity = 0;
    this._draws = [];
    this._rect = [0, 0, 0, 0];
    this._mask = (pass) => this._encodeMask(pass);
    /** Meshes outlined this frame. */
    this.count = 0;
  }

  /**
   * What is outlined this frame, and where on screen, from the scene and the
   * view-projection the forward pass draws with. `scale` is the view's pixels
   * a canvas pixel, for a frame drawn at a lower resolution. Returns how many
   * meshes are outlined.
   */
  prepare(scene, viewProjection, width, height, scale) {
    this.count = 0;
    if (scene.outlines === undefined || scene.outlines.size === 0) return 0;
    if (this._pairs.length < scene.renderableCount * 2) this._pairs = new Uint32Array(scene.renderableCount * 2);
    this._styles.clear();
    const n = outlinedRenderables(scene, this._pairs, this._styles);
    if (n === 0) return 0;

    // The looks, in this view's pixels, and how far the widest reaches.
    const data = this._styleData, looks = this._looks;
    let widest = 0;
    for (const { slot, look } of this._styles.values()) {
      looks[slot] = look;
      const o = 4 + slot * STYLE_FLOATS, w = look.width * scale;
      data.set(look.color, o);
      data[o + 4] = w;
      widest = Math.max(widest, w);
    }
    const reach = Math.ceil(widest + 0.5);
    data[0] = reach;
    this.rhi.queue.writeBuffer(this._styleBuffer, 0, data, 0, 4 + (this._styles.size + 1) * STYLE_FLOATS);
    this.rhi.queue.writeBuffer(this._camera, 0, viewProjection);

    // A draw a mesh: its renderable in the list, its slot where a batch keeps its shift.
    if (n > this._capacity) {
      this._capacity = Math.max(n, this._capacity * 2, 16);
      this._items?.destroy();
      this._batches?.destroy();
      this._items = createBuffer(this.rhi, { label: 'outline-items', size: this._capacity * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this._batches = createBuffer(this.rhi, { label: 'outline-batches', size: this._capacity * this.alignment, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this._batchData = new Uint32Array(this._capacity * this.alignment / 4);
      this._drawGroup = this.rhi.device.createBindGroup({
        label: 'outline-batch', layout: this.shadows._drawLayout,
        entries: [{ binding: 0, resource: { buffer: this._batches, size: BATCH_BYTES } }],
      });
    }
    const items = new Uint32Array(n), words = this.alignment / 4, draws = this._draws;
    draws.length = 0;
    const box = [Infinity, Infinity, -Infinity, -Infinity];
    let whole = false;
    for (let k = 0; k < n; k++) {
      const i = this._pairs[k * 2], slot = this._pairs[k * 2 + 1];
      items[k] = i;
      this._batchData[k * words] = k;
      this._batchData[k * words + 1] = slot;
      draws.push({ k, primitive: scene.renderablePrimitive[i], skinned: scene.renderableSkin[i] >= 0, through: looks[slot].through });
      whole = whole || !screenBox(box, viewProjection, scene.worldMin, scene.worldMax, i * 3);
    }
    // Pipelines changed least: by how they are drawn.
    draws.sort((a, b) => (a.skinned - b.skinned) || (a.through - b.through));
    this.rhi.queue.writeBuffer(this._items, 0, items);
    this.rhi.queue.writeBuffer(this._batches, 0, this._batchData, 0, n * words);

    // The composite's scissor: the outlined boxes on screen, grown by the reach.
    const r = this._rect;
    if (whole) { r[0] = 0; r[1] = 0; r[2] = width; r[3] = height; } else {
      const x0 = Math.max(0, Math.floor((box[0] * 0.5 + 0.5) * width) - reach);
      const x1 = Math.min(width, Math.ceil((box[2] * 0.5 + 0.5) * width) + reach);
      const y0 = Math.max(0, Math.floor((0.5 - box[3] * 0.5) * height) - reach);
      const y1 = Math.min(height, Math.ceil((0.5 - box[1] * 0.5) * height) + reach);
      r[0] = x0; r[1] = y0; r[2] = Math.max(0, x1 - x0); r[3] = Math.max(0, y1 - y0);
    }
    this.count = n;
    return n;
  }

  addPasses(graph, { sceneColor, depth, width, height, gpu, palette, morph, colorFormat }) {
    if (this.count === 0 || this._rect[2] === 0 || this._rect[3] === 0) return;
    const mask = graph.createTexture('outline-mask', {
      width, height, format: MASK_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this._frameGroup = this.shadows._makeCascadeBindGroup(gpu, palette, morph, this._camera, this._items);
    graph.addPass({
      name: 'outline-mask',
      color: [{ resource: mask, clear: { r: 0, g: 0, b: 0, a: 0 } }],
      depth: { resource: depth },
      execute: this._mask,
    });
    graph.addPass({
      name: 'outline',
      reads: [mask],
      color: [{ resource: sceneColor }],
      execute: (pass) => this._encodeComposite(pass, graph.viewOf(mask), colorFormat),
    });
  }

  _encodeMask(pass) {
    const geometry = this.shadows.geometry;
    geometry.bindPositions(pass);
    pass.setBindGroup(GROUP_FRAME, this._frameGroup, [0]);
    this.shadows.pipelineLayout.bindEmptyGroups(pass);
    let bound = '';
    for (const { k, primitive, skinned, through } of this._draws) {
      const key = `${skinned ? 1 : 0}${through ? 1 : 0}`;
      if (key !== bound) {
        pass.setPipeline(this.pipelines.get(this.descriptors[key]));
        if (skinned) pass.setVertexBuffer(1, geometry.skinBuffer);
        bound = key;
      }
      pass.setBindGroup(GROUP_DRAW, this._drawGroup, [k * this.alignment]);
      pass.drawIndexed(primitive.indexCount, 1, primitive.firstIndex, primitive.baseVertex);
    }
  }

  _encodeComposite(pass, maskView, colorFormat) {
    let descriptor = this.compositeDescriptors.get(colorFormat);
    if (!descriptor) {
      descriptor = {
        label: 'outline',
        layout: createPipelineLayout(this.rhi.device, { 0: this.compositeLayout }, 'outline'),
        shader: this._compositeShader,
        buffers: [],
        targets: [{ format: colorFormat, blend: {
          color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
        } }],
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depth: null,
      };
      this.compositeDescriptors.set(colorFormat, descriptor);
    }
    pass.setPipeline(this.pipelines.get(descriptor));
    pass.setBindGroup(0, this.rhi.device.createBindGroup({
      label: 'outline', layout: this.compositeLayout,
      entries: [{ binding: 0, resource: maskView }, { binding: 1, resource: { buffer: this._styleBuffer } }],
    }));
    pass.setScissorRect(...this._rect);
    pass.draw(3);
  }

  destroy() {
    this._items?.destroy();
    this._batches?.destroy();
    this._styleBuffer.destroy();
    this._camera.destroy();
  }
}

/**
 * Widen `box` ([x0, y0, x1, y1] in clip space) by renderable `o / 3`'s world
 * box through `m`. False when a corner is behind the eye: then only the whole
 * screen is sure to hold it.
 */
function screenBox(box, m, min, max, o) {
  for (let c = 0; c < 8; c++) {
    const x = c & 1 ? max[o] : min[o], y = c & 2 ? max[o + 1] : min[o + 1], z = c & 4 ? max[o + 2] : min[o + 2];
    const w = m[3] * x + m[7] * y + m[11] * z + m[15];
    if (!(w > 1e-6)) return false;
    const px = (m[0] * x + m[4] * y + m[8] * z + m[12]) / w, py = (m[1] * x + m[5] * y + m[9] * z + m[13]) / w;
    box[0] = Math.min(box[0], px); box[1] = Math.min(box[1], py);
    box[2] = Math.max(box[2], px); box[3] = Math.max(box[3], py);
  }
  return true;
}
