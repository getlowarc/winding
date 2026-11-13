// A frame's per-renderable work for what moved, as standalone functions over
// shared columns: each renderable's world box, and its draw data and cull
// bounds for the GPU. As transformJob.js is for the hierarchy: the serial
// path and the parallel one run this same code, so they cannot disagree.
//
// Both are a renderable at a time, reading its node's world matrix and
// writing only its own entries, so a dispatch's items can go to any thread in
// any order. Run after the transforms have composed.

import { aabbTransform } from '../core/math/aabb.js';

/** Floats of one renderable's draw data: DRAW_DATA_BYTES / 4 in gpudriven.js. */
const DRAW_FLOATS = 24;
/** Floats of one renderable's cull bounds: BOUNDS_FLOATS in gpudriven.js. */
const BOUNDS_FLOATS = 12;

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
 * The draw data of items[start..end): the model matrix, the tint and glow
 * words (see DrawData in shaders/pbr.js), where its joints begin, and zero
 * morph words -- the
 * renderer writes them over for what is morphed, from its primitives. And its
 * cull bounds: the world box, the coverage it draws between, and its LOD
 * group's sphere through that group's world matrix, or a negative radius to
 * keep it out of every view when hidden.
 */
export function drawDataRange(columns, start, end) {
  const { items, world, matrixSlot, drawData, drawWords, tint, skin, paletteOffsets, boundsData,
    worldMin, worldMax, coverage, lodSlot, lodSphere, hidden } = columns;
  for (let k = start; k < end; k++) {
    const i = items[k];
    const w = matrixSlot[i] * 16, o = i * DRAW_FLOATS;
    for (let j = 0; j < 16; j++) drawData[o + j] = world[w + j];
    drawWords[o + 16] = tint[i * 3];
    drawWords[o + 17] = tint[i * 3 + 1];
    drawWords[o + 18] = tint[i * 3 + 2];
    // Zero for anything unskinned, which the unskinned vertex shader never reads anyway.
    const s = skin[i];
    drawWords[o + 19] = s >= 0 ? paletteOffsets[s] : 0;
    drawWords[o + 20] = 0;
    drawWords[o + 21] = 0;
    drawWords[o + 22] = 0;

    const b = i * BOUNDS_FLOATS, v = i * 3;
    boundsData[b] = worldMin[v];
    boundsData[b + 1] = worldMin[v + 1];
    boundsData[b + 2] = worldMin[v + 2];
    boundsData[b + 3] = coverage[i * 2];
    boundsData[b + 4] = worldMax[v];
    boundsData[b + 5] = worldMax[v + 1];
    boundsData[b + 6] = worldMax[v + 2];
    boundsData[b + 7] = coverage[i * 2 + 1];
    // Every level of a group measures the same sphere, so no distance shows two levels or none.
    const l = lodSlot[i];
    if (l >= 0) {
      const m = l * 16, x = lodSphere[i * 4], y = lodSphere[i * 4 + 1], z = lodSphere[i * 4 + 2];
      boundsData[b + 8] = world[m] * x + world[m + 4] * y + world[m + 8] * z + world[m + 12];
      boundsData[b + 9] = world[m + 1] * x + world[m + 5] * y + world[m + 9] * z + world[m + 13];
      boundsData[b + 10] = world[m + 2] * x + world[m + 6] * y + world[m + 10] * z + world[m + 14];
      boundsData[b + 11] = lodSphere[i * 4 + 3] * Math.sqrt(Math.max(
        world[m] * world[m] + world[m + 1] * world[m + 1] + world[m + 2] * world[m + 2],
        world[m + 4] * world[m + 4] + world[m + 5] * world[m + 5] + world[m + 6] * world[m + 6],
        world[m + 8] * world[m + 8] + world[m + 9] * world[m + 9] + world[m + 10] * world[m + 10],
      ));
    } else {
      boundsData[b + 11] = 0;
    }
    // Hidden: a negative radius, which lodSelected keeps out of every view.
    if (hidden[i] === 1) boundsData[b + 11] = -1;
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
    skin: new Int32Array(buffers.skin),
    paletteOffsets: new Uint32Array(buffers.paletteOffsets),
    boundsData: new Float32Array(buffers.boundsData),
    coverage: new Float32Array(buffers.coverage),
    lodSlot: new Int32Array(buffers.lodSlot),
    lodSphere: new Float32Array(buffers.lodSphere),
    hidden: new Uint8Array(buffers.hidden),
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
    skin: columns.skin.buffer,
    paletteOffsets: columns.paletteOffsets.buffer,
    boundsData: columns.boundsData.buffer,
    coverage: columns.coverage.buffer,
    lodSlot: columns.lodSlot.buffer,
    lodSphere: columns.lodSphere.buffer,
    hidden: columns.hidden.buffer,
  };
}
