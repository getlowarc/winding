// Motion blur (renderer.motionBlur): the frame smeared along how far each
// pixel moved since the last one, as a camera's shutter, open for a share of
// the frame, smears it.
//
// Each opaque surface writes its own motion beside its colour (fsVelocity in
// shaders/pbr.js): where it is now against where it was, from last frame's
// draw data and joints, copied at the end of that frame -- so a car blurs as
// it passes a still camera, and a still car does not. The sky, which writes
// none, moves as the camera turns, worked out here from the camera alone.
// This pass then averages the colour along that motion, about the pixel.
// ponytail: the centre's motion alone, so a fast object's edge smears what is
// behind it into itself a little; per-tile maximum motion (McGuire 2012) if a
// game's look needs the background to stay crisp round a moving edge.

import { compileShader } from '../rhi/shader.js';
import { createPipelineLayout } from '../rhi/bindgroups.js';
import { createBuffer } from '../rhi/buffer.js';
import { clampSampler } from '../rhi/texture.js';
import { checkOptions } from '../core/options.js';

/** Each opaque pixel's motion since last frame, in UV: two halves. */
export const VELOCITY_FORMAT = 'rg16float';

/** motionBlur's defaults: the shutter open half the frame, as film's 180 degrees. */
export const MOTION_BLUR_DEFAULTS = Object.freeze({ strength: 0.5, samples: 12, camera: true, maxPixels: 48 });

/** renderer.motionBlur, checked once a frame: settings, or null for off. */
export function motionBlurSettings(value) {
  if (value === null || value === undefined || value === false) return null;
  if (value === true) return MOTION_BLUR_DEFAULTS;
  if (typeof value !== 'object') throw new Error(`motionBlur: true, { strength, samples, camera, maxPixels }, or null to turn it off, got ${value}`);
  checkOptions('motionBlur', value);
  const s = { ...MOTION_BLUR_DEFAULTS, ...value };
  if (!(s.strength > 0 && s.strength <= 2)) throw new Error(`motionBlur: strength must be above 0 and at most 2, got ${s.strength}`);
  if (!(Number.isInteger(s.samples) && s.samples >= 2 && s.samples <= 32)) throw new Error(`motionBlur: samples must be a whole number from 2 to 32, got ${s.samples}`);
  if (!(s.maxPixels > 0 && Number.isFinite(s.maxPixels))) throw new Error(`motionBlur: maxPixels must be a positive number, got ${s.maxPixels}`);
  return s;
}

const SHADER = /* wgsl */ `
struct Params {
  // This frame's inverse view-projection and last frame's, unjittered: the sky's motion.
  inverse  : mat4x4<f32>,
  previous : mat4x4<f32>,
  // x strength, y samples, z maxPixels, w 1 to count the camera's motion.
  settings : vec4<f32>,
  // xy the view's size in pixels.
  size     : vec4<f32>,
};
@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var colour : texture_2d<f32>;
@group(0) @binding(2) var velocity : texture_2d<f32>;
@group(0) @binding(3) var depth : texture_depth_2d;
@group(0) @binding(4) var linearSampler : sampler;

@vertex
fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4<f32> {
  return vec4<f32>(vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) p : vec4<f32>) -> @location(0) vec4<f32> {
  let c = vec2<i32>(p.xy);
  let size = params.size.xy;
  var motion = textureLoad(velocity, c, 0).xy;
  // The sky (reverse-Z's cleared depth): no surface wrote its motion. A
  // direction, not a point, so w = 0 and the camera's turn alone moves it.
  if (textureLoad(depth, c, 0) == 0.0 && params.settings.w > 0.5) {
    let ndc = vec2<f32>(p.x / size.x * 2.0 - 1.0, 1.0 - p.y / size.y * 2.0);
    let direction = params.inverse * vec4<f32>(ndc, 0.0, 1.0);
    let before = params.previous * vec4<f32>(direction.xyz, 0.0);
    if (before.w > 1e-6) {
      motion = (ndc - before.xy / before.w) * vec2<f32>(0.5, -0.5);
    }
  }
  var pixels = motion * size * params.settings.x;
  let length = length(pixels);
  let centre = textureLoad(colour, c, 0);
  if (length < 0.5) { return centre; }
  if (length > params.settings.z) { pixels = pixels * (params.settings.z / length); }
  // Interleaved gradient noise turns each pixel's taps a little, so too few
  // of them over a long smear is grain, not bands.
  let noise = fract(52.9829189 * fract(dot(p.xy, vec2<f32>(0.06711056, 0.00583715))));
  let n = i32(params.settings.y);
  var sum = vec3<f32>(0.0);
  for (var i = 0; i < n; i++) {
    let t = (f32(i) + noise) / f32(n) - 0.5;
    sum += textureSampleLevel(colour, linearSampler, (p.xy + pixels * t) / size, 0.0).rgb;
  }
  return vec4<f32>(sum / f32(n), centre.a);
}
`;

export class MotionBlurPass {
  static async create(rhi, pipelines, format) {
    const shader = await compileShader(rhi.device, SHADER, 'motionblur.wgsl');
    const layout = rhi.device.createBindGroupLayout({
      label: 'motion-blur',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    const descriptor = {
      label: 'motion-blur', layout: createPipelineLayout(rhi.device, { 0: layout }, 'motion-blur'), shader,
      targets: [{ format }], primitive: { topology: 'triangle-list', cullMode: 'none' }, depth: null,
    };
    await pipelines.warm([descriptor]);
    return new MotionBlurPass(rhi, pipelines, descriptor, layout, format);
  }

  constructor(rhi, pipelines, descriptor, layout, format) {
    this.rhi = rhi;
    this.pipelines = pipelines;
    this._descriptor = descriptor;
    this._layout = layout;
    this._format = format;
    this._data = new Float32Array(40);
    this._buffer = createBuffer(rhi, { label: 'motion-blur', size: this._data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  /** The blurred frame, for the passes after to take in place of `color`. */
  addPass(graph, { color, velocity, depth, width, height, settings, inverse, previous }) {
    const d = this._data;
    d.set(inverse, 0);
    d.set(previous, 16);
    d.set([settings.strength, settings.samples, settings.maxPixels, settings.camera ? 1 : 0, width, height, 0, 0], 32);
    this.rhi.queue.writeBuffer(this._buffer, 0, d);
    const out = graph.createTexture('motion-blur', {
      width, height, format: this._format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    graph.addPass({
      name: 'motion-blur',
      reads: [color, velocity, depth],
      color: [{ resource: out, clear: { r: 0, g: 0, b: 0, a: 1 } }],
      execute: (pass) => {
        pass.setPipeline(this.pipelines.get(this._descriptor));
        pass.setBindGroup(0, this.rhi.device.createBindGroup({
          label: 'motion-blur', layout: this._layout,
          entries: [
            { binding: 0, resource: { buffer: this._buffer } },
            { binding: 1, resource: graph.viewOf(color) },
            { binding: 2, resource: graph.viewOf(velocity) },
            { binding: 3, resource: graph.viewOf(depth) },
            { binding: 4, resource: clampSampler(this.rhi) },
          ],
        }));
        pass.draw(3);
      },
    });
    return out;
  }

  destroy() {
    this._buffer.destroy();
  }
}
