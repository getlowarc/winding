// A frame's per-renderable work for what moved, as standalone functions over
// shared columns: each renderable's world box, and its matrix block in the
// draw data the GPU reads. As transformJob.js is for the hierarchy: the serial
// path and the parallel one run this same code, so they cannot disagree.
//
// Both are a renderable at a time, reading its node's world matrix and
// writing only its own entries, so a dispatch's items can go to any thread in
// any order. Run after the transforms have composed.

import { aabbTransform } from '../core/math/aabb.js';
import { mat4NormalMatrix } from '../core/math/mat4.js';

/** Floats of one renderable's draw data: DRAW_DATA_BYTES / 4 in gpudriven.js. */
const DRAW_FLOATS = 32;

/**
 * World boxes from local ones, through each renderable's world matrix: for
 * items[start..end) -- or with `all`, for renderables start..end whose node
 * moved this frame (transforms.moved), when too many moved to list.
 */
export function boundsRange(columns, start, end, all) {
  const { items, localMin, localMax, worldMin, worldMax, world, matrixSlot, moved } = columns;
  for (let k = start; k < end; k++) {
    const i = all ? k : items[k];
    const slot = matrixSlot[i];
    if (all && moved[slot] === 0) continue;
    // Offsets rather than subarray views: no allocation per renderable.
    aabbTransform(worldMin, worldMax, localMin, localMax, world, slot * 16, i * 3, i * 3);
  }
}

/**
 * The model matrix and its normal matrix of items[start..end) into their draw
 * data, and the tint and glow words that ride in the normal matrix's padding
 * (see DrawData in shaders/pbr.js). The rest of a renderable's draw data --
 * its joints' and morph targets' places -- is the renderer's to write.
 */
export function matricesRange(columns, start, end) {
  const { items, world, matrixSlot, drawData, drawWords, tint } = columns;
  for (let k = start; k < end; k++) {
    const i = items[k];
    const w = matrixSlot[i] * 16, o = i * DRAW_FLOATS;
    for (let j = 0; j < 16; j++) drawData[o + j] = world[w + j];
    mat4NormalMatrix(drawData, world, o + 16, w);
    // After the normal matrix, whose padding it writes as zeros.
    drawWords[o + 19] = tint[i * 3];
    drawWords[o + 23] = tint[i * 3 + 1];
    drawWords[o + 27] = tint[i * 3 + 2];
  }
}

/** Rebuild the views a worker needs from the raw shared buffers. */
export function renderableColumnsFromBuffers(buffers) {
  return {
    items: new Uint32Array(buffers.items),
    localMin: new Float32Array(buffers.localMin),
    localMax: new Float32Array(buffers.localMax),
    worldMin: new Float32Array(buffers.worldMin),
    worldMax: new Float32Array(buffers.worldMax),
    matrixSlot: new Uint32Array(buffers.matrixSlot),
    tint: new Uint32Array(buffers.tint),
    world: new Float32Array(buffers.world),
    moved: new Uint8Array(buffers.moved),
    drawData: new Float32Array(buffers.drawData),
    drawWords: new Uint32Array(buffers.drawData),
  };
}

/** The buffers of a set of columns, to publish to the workers: the inverse of the above. */
export function renderableBuffers(columns) {
  return {
    items: columns.items.buffer,
    localMin: columns.localMin.buffer,
    localMax: columns.localMax.buffer,
    worldMin: columns.worldMin.buffer,
    worldMax: columns.worldMax.buffer,
    matrixSlot: columns.matrixSlot.buffer,
    tint: columns.tint.buffer,
    world: columns.world.buffer,
    moved: columns.moved.buffer,
    drawData: columns.drawData.buffer,
  };
}
