// The geometry arena: every primitive's vertices in one buffer, every
// primitive's indices in another.
//
// A buffer per primitive meant every batch rebound its own vertex and index
// buffers before drawing, a command apiece, for a hundred batches a pass. In
// one arena a pass binds both once, and a primitive is where it starts:
// `baseVertex` and `firstIndex`, which every draw call already takes. It is
// also what lets one draw reach the geometry of many primitives.
//
// The buffers are STORAGE as well as VERTEX and INDEX, for shaders that read
// them directly, and COPY_SRC, so growing can carry what is there across on
// the GPU.
//
// Every vertex's position is kept a second time, alone, in a buffer of its
// own at the same vertex numbering: a shadow pass writes only depth, and
// fetching the whole 60-byte vertex for its 12 bytes of position is five
// times the memory traffic on a GPU that is short of it.
//
// And every index a second time, in a buffer of its own at the same place,
// for the same depth-only passes: each one naming the first vertex of its
// primitive with the same position (and joints and weights) -- the vertex a
// depth-only pass cannot tell from it. A flat-shaded mesh repeats each corner
// once for every face that meets there, each with its own normal; drawn from
// these, the same triangles come from a quarter of the vertices, and the
// GPU's cache of transformed vertices finds the rest. 300 low-poly walkers:
// the sun's four cascades from 3.4 to 2.4 ms on Iris Xe. A morphed
// primitive keeps its own indices, its targets moving each vertex on its
// own, and so does one whose shadow takes its shape from alpha (Winding._sharesCorners).
//
// A skinned vertex's joints and weights go in a third buffer, also at the
// same numbering, made when the first skinned primitive arrives. They have to
// share it: a draw's base vertex moves every vertex buffer it reads, and a
// skin buffer of the primitive's own, counted from 0, was read from the
// primitive's place in the arena -- another mesh's influences, and a walking
// fox in pieces.

import { grownCapacity, RangeAllocator } from '../core/grow.js';
import { storageCapacity, createBuffer } from '../rhi/buffer.js';
import { VERTEX_STRIDE_BYTES, SKIN_STRIDE_BYTES } from './vertex.js';

const INDEX_BYTES = 4;
const POSITION_BYTES = 12;

export class GeometryArena {
  constructor(rhi, { vertexCapacity = 1 << 16, indexCapacity = 1 << 18 } = {}) {
    this.rhi = rhi;
    this.vertexCapacity = vertexCapacity;
    this.indexCapacity = indexCapacity;
    this.vertexBuffer = this._create('geometry-vertices', vertexCapacity * VERTEX_STRIDE_BYTES,
      GPUBufferUsage.VERTEX);
    this.positionBuffer = this._create('geometry-positions', vertexCapacity * POSITION_BYTES, GPUBufferUsage.VERTEX);
    this.indexBuffer = this._create('geometry-indices', indexCapacity * INDEX_BYTES, GPUBufferUsage.INDEX);
    this.depthIndexBuffer = this._create('geometry-depth-indices', indexCapacity * INDEX_BYTES, GPUBufferUsage.INDEX);
    this._vertices = new RangeAllocator();
    this._indices = new RangeAllocator();
    /** Joints and weights, SKIN_STRIDE_BYTES a vertex; null until something is skinned. */
    this.skinBuffer = null;
    /** Bumped when either buffer is replaced, so bind groups naming them rebuild. */
    this.revision = 0;
  }

  _create(label, size, usage) {
    return createBuffer(this.rhi, {
      label,
      size,
      usage: usage | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }

  /**
   * One primitive's interleaved vertices (VERTEX_STRIDE_BYTES each) and its
   * indices, relative to its own first vertex, and for a skinned one its
   * packed joints and weights (packSkinVertices). `shared`: whether its
   * depth-only draws may share vertices by position -- false for a morphed one.
   * Returns where they landed. Called at load, never during a frame.
   */
  allocate(vertices, indices, skin = null, shared = true) {
    const vertexCount = vertices.byteLength / VERTEX_STRIDE_BYTES;
    const baseVertex = this._place('vertex', vertexCount, VERTEX_STRIDE_BYTES);
    const firstIndex = this._place('index', indices.length, INDEX_BYTES);
    this.writeAt(baseVertex, firstIndex, vertices, indices, skin, shared);
    return { baseVertex, firstIndex };
  }

  /**
   * Write a primitive where allocate() placed it before: after a lost device,
   * the same file read again goes back to the same place, so every primitive
   * object and every draw that names its baseVertex and firstIndex is right.
   */
  writeAt(baseVertex, firstIndex, vertices, indices, skin = null, shared = true) {
    const vertexCount = vertices.byteLength / VERTEX_STRIDE_BYTES;
    const queue = this.rhi.queue;
    queue.writeBuffer(this.vertexBuffer, baseVertex * VERTEX_STRIDE_BYTES, vertices.buffer ?? vertices, vertices.byteOffset ?? 0, vertices.byteLength);
    queue.writeBuffer(this.indexBuffer, firstIndex * INDEX_BYTES, indices.buffer, indices.byteOffset, indices.byteLength);
    const depth = shared ? depthIndices(vertices, indices, skin) : indices;
    queue.writeBuffer(this.depthIndexBuffer, firstIndex * INDEX_BYTES, depth.buffer, depth.byteOffset, depth.byteLength);
    // The positions: the first three floats of each vertex.
    const floats = new Float32Array(vertices.buffer ?? vertices, vertices.byteOffset ?? 0, vertices.byteLength / 4);
    const positions = new Float32Array(vertexCount * 3);
    for (let v = 0; v < vertexCount; v++) {
      positions[v * 3] = floats[v * 15];
      positions[v * 3 + 1] = floats[v * 15 + 1];
      positions[v * 3 + 2] = floats[v * 15 + 2];
    }
    queue.writeBuffer(this.positionBuffer, baseVertex * POSITION_BYTES, positions);
    if (skin !== null) {
      this.skinBuffer ??= this._create('geometry-skin', this.vertexCapacity * SKIN_STRIDE_BYTES, GPUBufferUsage.VERTEX);
      queue.writeBuffer(this.skinBuffer, baseVertex * SKIN_STRIDE_BYTES, skin);
    }
  }

  /**
   * New buffers on a new device, at the capacities these had and empty. The
   * allocator is kept, so what was placed stays placed; writeAt refills it.
   */
  restore() {
    this.vertexBuffer = this._create('geometry-vertices', this.vertexCapacity * VERTEX_STRIDE_BYTES, GPUBufferUsage.VERTEX);
    this.positionBuffer = this._create('geometry-positions', this.vertexCapacity * POSITION_BYTES, GPUBufferUsage.VERTEX);
    this.indexBuffer = this._create('geometry-indices', this.indexCapacity * INDEX_BYTES, GPUBufferUsage.INDEX);
    this.depthIndexBuffer = this._create('geometry-depth-indices', this.indexCapacity * INDEX_BYTES, GPUBufferUsage.INDEX);
    this.skinBuffer = this.skinBuffer === null ? null
      : this._create('geometry-skin', this.vertexCapacity * SKIN_STRIDE_BYTES, GPUBufferUsage.VERTEX);
    this.revision++;
  }

  /** Give back what allocate() returned, for `vertexCount` vertices and `indexCount` indices. */
  free(baseVertex, vertexCount, firstIndex, indexCount) {
    this._vertices.free(baseVertex, vertexCount);
    this._indices.free(firstIndex, indexCount);
  }

  /** Room for `count` more vertices or indices, growing the buffer if they pass its end. */
  _place(kind, count, bytes) {
    const ranges = kind === 'vertex' ? this._vertices : this._indices;
    const written = ranges.end;
    const base = ranges.alloc(count);
    const capacity = kind === 'vertex' ? this.vertexCapacity : this.indexCapacity;
    if (ranges.end > capacity) {
      const grown = grownCapacity(capacity, ranges.end, storageCapacity(this.rhi, bytes), `${kind}es`);
      const old = kind === 'vertex' ? this.vertexBuffer : this.indexBuffer;
      const buffer = this._create(old.label, grown * bytes, kind === 'vertex' ? GPUBufferUsage.VERTEX : GPUBufferUsage.INDEX);
      // Only what has been written: the rest of the old buffer was never set.
      if (written > 0) {
        const encoder = this.rhi.device.createCommandEncoder({ label: 'geometry-grow' });
        encoder.copyBufferToBuffer(old, 0, buffer, 0, written * bytes);
        this.rhi.queue.submit([encoder.finish()]);
      }
      if (kind === 'vertex') {
        // The positions grow with the vertices: one numbering for both.
        const positions = this._create('geometry-positions', grown * POSITION_BYTES, GPUBufferUsage.VERTEX);
        if (written > 0) {
          const encoder = this.rhi.device.createCommandEncoder({ label: 'geometry-grow' });
          encoder.copyBufferToBuffer(this.positionBuffer, 0, positions, 0, written * POSITION_BYTES);
          this.rhi.queue.submit([encoder.finish()]);
        }
        this.positionBuffer.destroy();
        this.positionBuffer = positions;
        if (this.skinBuffer !== null) {
          const skin = this._create('geometry-skin', grown * SKIN_STRIDE_BYTES, GPUBufferUsage.VERTEX);
          if (written > 0) {
            const encoder = this.rhi.device.createCommandEncoder({ label: 'geometry-grow' });
            encoder.copyBufferToBuffer(this.skinBuffer, 0, skin, 0, written * SKIN_STRIDE_BYTES);
            this.rhi.queue.submit([encoder.finish()]);
          }
          this.skinBuffer.destroy();
          this.skinBuffer = skin;
        }
      }
      if (kind === 'index') {
        // The depth-only indices grow with the indices: one placement for both.
        const depth = this._create('geometry-depth-indices', grown * INDEX_BYTES, GPUBufferUsage.INDEX);
        if (written > 0) {
          const encoder = this.rhi.device.createCommandEncoder({ label: 'geometry-grow' });
          encoder.copyBufferToBuffer(this.depthIndexBuffer, 0, depth, 0, written * INDEX_BYTES);
          this.rhi.queue.submit([encoder.finish()]);
        }
        this.depthIndexBuffer.destroy();
        this.depthIndexBuffer = depth;
      }
      old.destroy();
      if (kind === 'vertex') { this.vertexBuffer = buffer; this.vertexCapacity = grown; }
      else { this.indexBuffer = buffer; this.indexCapacity = grown; }
      this.revision++;
    }
    return base;
  }

  /** Both buffers, bound for a pass's draws. */
  bind(pass) {
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint32');
  }

  /** The positions and the depth-only indices: for a pass that writes only depth. */
  bindPositions(pass) {
    pass.setVertexBuffer(0, this.positionBuffer);
    pass.setIndexBuffer(this.depthIndexBuffer, 'uint32');
  }

  destroy() {
    this.skinBuffer?.destroy();
    this.positionBuffer.destroy();
    this.vertexBuffer.destroy();
    this.indexBuffer.destroy();
    this.depthIndexBuffer.destroy();
  }
}

/**
 * A primitive's indices for a depth-only pass: each naming the first of its
 * vertices with the same position bits and, when skinned, the same joints and
 * weights -- everything such a pass reads of a vertex. `indices` itself when
 * no two vertices share them. Bits, not values: -0 and 0 stay apart, which
 * only ever shares less.
 */
export function depthIndices(vertices, indices, skin = null) {
  const count = vertices.byteLength / VERTEX_STRIDE_BYTES;
  const words = new Uint32Array(vertices.buffer ?? vertices, vertices.byteOffset ?? 0, count * (VERTEX_STRIDE_BYTES / 4));
  const skinWords = skin === null ? null : new Uint32Array(skin.buffer, skin.byteOffset, count * (SKIN_STRIDE_BYTES / 4));
  const stride = VERTEX_STRIDE_BYTES / 4, skinStride = SKIN_STRIDE_BYTES / 4;
  const same = (a, b) => {
    for (let k = 0; k < 3; k++) if (words[a * stride + k] !== words[b * stride + k]) return false;
    if (skinWords !== null) for (let k = 0; k < skinStride; k++) if (skinWords[a * skinStride + k] !== skinWords[b * skinStride + k]) return false;
    return true;
  };
  // Open addressing: a slot holds a vertex + 1, zero for empty.
  let size = 1;
  while (size < count * 2) size <<= 1;
  const table = new Uint32Array(size), first = new Uint32Array(count);
  let shared = 0;
  for (let v = 0; v < count; v++) {
    let h = 0x811c9dc5;
    for (let k = 0; k < 3; k++) h = Math.imul(h ^ words[v * stride + k], 0x01000193);
    if (skinWords !== null) for (let k = 0; k < skinStride; k++) h = Math.imul(h ^ skinWords[v * skinStride + k], 0x01000193);
    let slot = (h ^ (h >>> 15)) & (size - 1);
    for (;;) {
      const held = table[slot];
      if (held === 0) { table[slot] = v + 1; first[v] = v; break; }
      if (same(held - 1, v)) { first[v] = held - 1; shared++; break; }
      slot = (slot + 1) & (size - 1);
    }
  }
  if (shared === 0) return indices;
  const out = new Uint32Array(indices.length);
  for (let i = 0; i < indices.length; i++) out[i] = first[indices[i]];
  return out;
}
