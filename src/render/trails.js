// Trails (scene.addTrail): a ribbon behind a moving node -- a sword's swing,
// a shell's streak, a plane's contrail. Where the node has been is sampled
// here, once its transforms are composed, and aged by the scene's time; each
// frame the ribbon is built on the CPU as a strip facing the camera, smoothed
// through the samples (Catmull-Rom) so a fast swing at 60 frames a second is
// an arc, not a fan of straight cuts.
// ponytail: built on the CPU a frame, ~40 bytes a strip vertex; a compute
// pass that expands the samples, if hundreds of long trails show up in a profile.

import { compileShader } from '../rhi/shader.js';
import { createPipelineLayout } from '../rhi/bindgroups.js';
import { createBuffer } from '../rhi/buffer.js';
import { DEPTH_FORMAT, DEPTH_COMPARE } from '../rhi/device.js';
import { clampSampler, defaultTextures } from '../rhi/texture.js';
import { grownCapacity } from '../core/grow.js';
import { handleIndex } from '../core/handle.js';
import { FRAME_WGSL } from './shaders/pbr.js';
import { FOG_WGSL } from './fog.js';

/** position, uv, colour: floats a strip vertex. */
const VERTEX_FLOATS = 9;
/** Strip points between two samples, the first of them included. */
const SMOOTHING = 4;
/** The most samples a trail keeps: the oldest go first. */
export const TRAIL_SAMPLES = 256;
/** Below this far from the last sample, the node has not moved. */
const STILL = 1e-4;

const SHADER = /* wgsl */ `
${FRAME_WGSL}
${FOG_WGSL}

@group(0) @binding(0) var<uniform> frame      : Frame;
@group(0) @binding(1) var          irradiance : texture_cube<f32>;
@group(0) @binding(2) var          envSampler : sampler;
@group(1) @binding(0) var          image      : texture_2d<f32>;
@group(1) @binding(1) var          imageSampler : sampler;

override ADDITIVE : bool = true;
override TEXTURED : bool = false;

struct Out {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv    : vec2<f32>,
  @location(1) color : vec4<f32>,
  @location(2) world : vec3<f32>,
};

@vertex
fn vs(@location(0) position : vec3<f32>, @location(1) uv : vec2<f32>, @location(2) color : vec4<f32>) -> Out {
  var out : Out;
  out.clip = frame.viewProjection * vec4<f32>(position, 1.0);
  out.uv = uv;
  out.color = color;
  out.world = position;
  return out;
}

@fragment
fn fs(v : Out) -> @location(0) vec4<f32> {
  // Without a texture, soft at the edges: full down the middle, nothing at the rim.
  let across = v.uv.y * 2.0 - 1.0;
  var colour = v.color * select(vec4<f32>(1.0, 1.0, 1.0, 1.0 - across * across), textureSample(image, imageSampler, v.uv), TEXTURED);
  if (frame.fog.x > 0.0) {
    let toPoint = v.world - frame.cameraPosition.xyz;
    let distance = length(toPoint);
    let through = exp(-fogDepth(frame.fog, frame.cameraPosition.xyz, toPoint / max(distance, 1e-6), distance));
    let inscatter = frame.fogAlbedo.rgb * fogMeanRadiance(irradiance, envSampler) + frame.fogLight.rgb;
    colour = vec4<f32>(select(colour.rgb * through + inscatter * (1.0 - through), colour.rgb * through, ADDITIVE), colour.a);
  }
  if (ADDITIVE) { return vec4<f32>(min(colour.rgb * colour.a, vec3<f32>(65504.0)), 0.0); }
  return vec4<f32>(min(colour.rgb, vec3<f32>(65504.0)), colour.a);
}
`;

const BLENDS = {
  additive: {
    color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
  },
  alpha: {
    color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  },
};

/**
 * Where a trail's node is now, added to its samples if it moved, and the
 * samples older than its lifetime dropped. `record` is the scene's
 * (trailRecord): samples as x, y, z, time from `start`, oldest first.
 */
export function sampleTrail(record, x, y, z, now) {
  const s = (record.samples ??= new Float32Array(TRAIL_SAMPLES * 4));
  // Aged out from the oldest.
  while (record.count > 0 && now - s[record.start * 4 + 3] > record.lifetime) {
    record.start++;
    record.count--;
  }
  if (record.count === 0) record.start = 0;
  if (!record.emitting) return;
  if (record.count > 0) {
    const last = (record.start + record.count - 1) * 4;
    if (Math.hypot(x - s[last], y - s[last + 1], z - s[last + 2]) < STILL) return;
  }
  // Full: the oldest goes. At the end of the array: the live ones moved to its start.
  if (record.count === TRAIL_SAMPLES) { record.start++; record.count--; }
  if (record.start + record.count === TRAIL_SAMPLES) {
    s.copyWithin(0, record.start * 4, (record.start + record.count) * 4);
    record.start = 0;
  }
  const at = (record.start + record.count) * 4;
  s[at] = x; s[at + 1] = y; s[at + 2] = z; s[at + 3] = now;
  record.count++;
}

/**
 * A trail's strip into `out` from float `o`: two vertices a point, through
 * its samples smoothed, each pair across the line from the eye. Returns how
 * many vertices.
 */
export function buildStrip(record, now, eye, out, o) {
  const s = record.samples, n = record.count, base = record.start * 4;
  if (n < 2) return 0;
  const points = (n - 1) * SMOOTHING + 1;
  let vertices = 0;
  let sx = 0, sy = 0, sz = 1;
  const { width, widthEnd, lifetime, color, colorEnd } = record;
  const p = [0, 0, 0, 0], next = [0, 0, 0, 0], prev = [0, 0, 0, 0];
  const at = (k, into) => {
    const i = Math.min(Math.floor(k / SMOOTHING), n - 2), t = k / SMOOTHING - i;
    // Catmull-Rom through samples i-1, i, i+1, i+2, the ends repeated.
    const a = base + Math.max(i - 1, 0) * 4, b = base + i * 4, c = base + (i + 1) * 4, d = base + Math.min(i + 2, n - 1) * 4;
    const t2 = t * t, t3 = t2 * t;
    for (let j = 0; j < 3; j++) {
      into[j] = 0.5 * (2 * s[b + j] + (-s[a + j] + s[c + j]) * t + (2 * s[a + j] - 5 * s[b + j] + 4 * s[c + j] - s[d + j]) * t2
        + (-s[a + j] + 3 * s[b + j] - 3 * s[c + j] + s[d + j]) * t3);
    }
    into[3] = s[b + 3] + (s[c + 3] - s[b + 3]) * t;
    return into;
  };
  for (let k = 0; k < points; k++) {
    at(k, p);
    at(Math.min(k + 1, points - 1), next);
    at(Math.max(k - 1, 0), prev);
    const tx = next[0] - prev[0], ty = next[1] - prev[1], tz = next[2] - prev[2];
    const ex = eye[0] - p[0], ey = eye[1] - p[1], ez = eye[2] - p[2];
    // Across: square to the line and to the eye. Edge-on, the last good one.
    const cx = ty * ez - tz * ey, cy = tz * ex - tx * ez, cz = tx * ey - ty * ex;
    const length = Math.hypot(cx, cy, cz);
    if (length > 1e-12) { sx = cx / length; sy = cy / length; sz = cz / length; }
    const age = Math.min(Math.max((now - p[3]) / lifetime, 0), 1);
    const half = (width + (widthEnd - width) * age) * 0.5;
    for (let side = 0; side < 2; side++) {
      const sign = side === 0 ? 1 : -1, v = o + vertices * VERTEX_FLOATS;
      out[v] = p[0] + sx * half * sign;
      out[v + 1] = p[1] + sy * half * sign;
      out[v + 2] = p[2] + sz * half * sign;
      out[v + 3] = age;
      out[v + 4] = side;
      for (let j = 0; j < 4; j++) out[v + 5 + j] = color[j] + (colorEnd[j] - color[j]) * age;
      vertices++;
    }
  }
  return vertices;
}

export class TrailPass {
  static async create(rhi, pipelines, frameBuffer, colorFormat) {
    const device = rhi.device;
    const shader = await compileShader(device, SHADER, 'trails.wgsl');
    const frameLayout = device.createBindGroupLayout({
      label: 'trails-frame',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: 'cube' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    const imageLayout = device.createBindGroupLayout({
      label: 'trails-image',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    const layout = createPipelineLayout(device, { 0: frameLayout, 1: imageLayout }, 'trails');
    const descriptors = {};
    for (const blend of ['alpha', 'additive']) {
      for (const textured of [false, true]) {
        descriptors[`${blend}${textured ? ':textured' : ''}`] = {
          label: `trails:${blend}${textured ? ':textured' : ''}`,
          layout,
          shader,
          buffers: [{
            arrayStride: VERTEX_FLOATS * 4,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x2' },
              { shaderLocation: 2, offset: 20, format: 'float32x4' },
            ],
          }],
          targets: [{ format: colorFormat, blend: BLENDS[blend] }],
          primitive: { topology: 'triangle-strip', cullMode: 'none' },
          depth: { format: DEPTH_FORMAT, depthCompare: DEPTH_COMPARE, depthWriteEnabled: false },
          constants: { ADDITIVE: blend === 'additive' ? 1 : 0, TEXTURED: textured ? 1 : 0 },
        };
      }
    }
    await pipelines.warm(Object.values(descriptors));
    return new TrailPass(rhi, pipelines, { descriptors, frameLayout, imageLayout }, frameBuffer);
  }

  constructor(rhi, pipelines, built, frameBuffer) {
    this.rhi = rhi;
    this.pipelines = pipelines;
    Object.assign(this, built);
    this._frameBuffer = frameBuffer;
    this._staging = new Float32Array(0);
    this._buffer = null;
    this._frameGroups = new WeakMap();
    this._imageGroups = new WeakMap();
    /** This frame's strips: { record, first, vertices, depth }. */
    this._list = [];
    this._environment = null;
    this._draw = (pass) => this._encode(pass);
    /** Trails drawn this frame. */
    this.count = 0;
  }

  /**
   * Sample where every trail's node is, unless `frozen` (a probe's capture,
   * which sees them as they are), and build this camera's strips. Returns
   * how many trails draw.
   */
  prepare(scene, camera, environment, frozen = false) {
    const list = this._list;
    list.length = 0;
    this.count = 0;
    if (scene.trails === undefined || scene.trails.size === 0) return 0;
    const world = scene.transforms.world, now = scene.time, eye = camera.position;
    let floats = 0;
    for (const [entity, record] of scene.trails) {
      const slot = handleIndex(entity);
      // Hidden: nothing drawn and nothing sampled, so it does not streak from
      // where it was hidden to where it is shown.
      if (scene.hidden?.[slot] === 1) { record.count = 0; continue; }
      if (!frozen) sampleTrail(record, world[slot * 16 + 12], world[slot * 16 + 13], world[slot * 16 + 14], now);
      if (record.count < 2) continue;
      const vertices = ((record.count - 1) * SMOOTHING + 1) * 2;
      const head = (record.start + record.count - 1) * 4;
      const depth = Math.hypot(record.samples[head] - eye[0], record.samples[head + 1] - eye[1], record.samples[head + 2] - eye[2]);
      list.push({ record, first: floats / VERTEX_FLOATS, vertices, depth });
      floats += vertices * VERTEX_FLOATS;
    }
    if (list.length === 0) return 0;
    if (this._staging.length < floats) this._staging = new Float32Array(grownCapacity(this._staging.length, floats, Infinity, 'trail vertices'));
    for (const item of list) item.vertices = buildStrip(item.record, now, eye, this._staging, item.first * VERTEX_FLOATS);
    // Alpha first, far to near; then additive, whose order does not matter.
    list.sort((a, b) => (a.record.blend === 'additive') - (b.record.blend === 'additive') || b.depth - a.depth);
    if (!this._buffer || this._buffer.size < floats * 4) {
      this._buffer?.destroy();
      this._buffer = createBuffer(this.rhi, {
        label: 'trails', size: this._staging.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
    }
    this.rhi.queue.writeBuffer(this._buffer, 0, this._staging, 0, floats);
    this._environment = environment;
    this.count = list.length;
    return this.count;
  }

  addPass(graph, { sceneColor, depth }) {
    if (this.count === 0) return;
    graph.addPass({ name: 'trails', color: [{ resource: sceneColor }], depth: { resource: depth }, execute: this._draw });
  }

  _encode(pass) {
    const environment = this._environment;
    let frameGroup = this._frameGroups.get(environment);
    if (!frameGroup) {
      frameGroup = this.rhi.device.createBindGroup({
        label: 'trails-frame',
        layout: this.frameLayout,
        entries: [
          { binding: 0, resource: { buffer: this._frameBuffer } },
          { binding: 1, resource: environment.irradianceView },
          { binding: 2, resource: environment.sampler },
        ],
      });
      this._frameGroups.set(environment, frameGroup);
    }
    pass.setBindGroup(0, frameGroup);
    pass.setVertexBuffer(0, this._buffer);
    let bound = null;
    for (const item of this._list) {
      const key = `${item.record.blend}${item.record.texture ? ':textured' : ''}`;
      if (key !== bound) {
        pass.setPipeline(this.pipelines.get(this.descriptors[key]));
        bound = key;
      }
      pass.setBindGroup(1, this._imageGroup(item.record.texture));
      pass.draw(item.vertices, 1, item.first);
    }
  }

  /** A texture's bind group; without one, a white texel the shader ignores. */
  _imageGroup(texture) {
    const key = texture ?? this;
    let group = this._imageGroups.get(key);
    if (!group) {
      group = this.rhi.device.createBindGroup({
        label: 'trails-image',
        layout: this.imageLayout,
        entries: [
          { binding: 0, resource: texture?.view ?? defaultTextures(this.rhi).white.createView() },
          { binding: 1, resource: clampSampler(this.rhi) },
        ],
      });
      this._imageGroups.set(key, group);
    }
    return group;
  }

  destroy() {
    this._buffer?.destroy();
  }
}
