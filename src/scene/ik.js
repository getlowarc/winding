// Inverse kinematics (scene.addIK): a limb reaching for a point, a head or a
// turret turned to look at one. Solved in scene.update once the transforms
// have composed -- so over whatever pose the animation left -- by turning
// the bones' local rotations, then composing again what they moved.
//
// Each frame starts from the pose underneath: a bone still holding what the
// solve wrote last frame is put back to what it held before, so a weight
// below 1 does not creep to full strength over the frames, and nothing is
// solved again when nothing it depends on moved.

import { handleIndex } from '../core/handle.js';
import { mat4Decompose } from '../core/math/mat4.js';
import { quatMultiply, quatConjugate, quatSetAxisAngle, quatFromTo, quatSlerp, quatNormalize } from '../core/math/quat.js';

const EPSILON = 1e-6;
const scratch = { pos: new Float64Array(3), scale: new Float64Array(3), rot: new Float64Array(4) };
const q0 = new Float64Array(4), q1 = new Float64Array(4), q2 = new Float64Array(4), q3 = new Float64Array(4), q4 = new Float64Array(4);
const IDENTITY = Float64Array.of(0, 0, 0, 1);

const point = (v, what, caller) => {
  if (!(v?.length === 3 && [...v].every(Number.isFinite))) throw new Error(`${caller}: ${what} must be a node or 3 finite numbers, got ${v}`);
  return Float64Array.from(v);
};

/** A node's parent, or null. */
function parentOf(scene, node) {
  const p = scene.transforms.parent[handleIndex(node.entity)];
  return p < 0 ? null : scene.node(scene.entities.handleAt(p));
}

/** addIK's options, checked: the record update() solves. */
export function ikRecord(scene, options, caller = 'addIK') {
  const { type, target, weight = 1 } = options;
  if (type !== 'twoBone' && type !== 'aim') throw new Error(`${caller}: type is 'twoBone' or 'aim', got ${type}`);
  if (!(weight >= 0 && weight <= 1)) throw new RangeError(`${caller}: weight must be from 0 to 1, got ${weight}`);
  if (target === undefined) throw new Error(`${caller}: target is required: a node, or a point in the world`);
  const where = (v, what) => (v?.entity !== undefined ? v : point(v, what, caller));
  const record = {
    options: { ...options }, type, weight, target: where(target, 'target'), bones: null, input: null, wrote: null,
    sig: new Float64Array(24), last: new Float64Array(24).fill(NaN),
  };
  if (type === 'twoBone') {
    const { end, pole = null } = options;
    const mid = end?.alive ? parentOf(scene, end) : null, root = mid === null ? null : parentOf(scene, mid);
    if (!end?.alive || root === null) throw new Error(`${caller}: end must be a live node with a parent and a grandparent: the foot, its knee and its hip`);
    record.bones = [root, mid];
    record.end = end;
    record.pole = pole === null ? null : where(pole, 'pole');
  } else {
    const { node, axis = [0, 0, 1], limit = Math.PI } = options;
    if (!node?.alive) throw new Error(`${caller}: node must be a live node: the head, the turret`);
    const a = point(axis, 'axis', caller);
    const length = Math.hypot(a[0], a[1], a[2]);
    if (!(length > 0)) throw new Error(`${caller}: axis must not be zero`);
    if (!(limit > 0 && limit <= Math.PI)) throw new RangeError(`${caller}: limit must be from above 0 to pi radians, got ${limit}`);
    record.bones = [node];
    record.axis = a.map((c) => c / length);
    record.limit = limit;
  }
  record.input = new Float32Array(record.bones.length * 4);
  record.wrote = new Float32Array(record.bones.length * 4).fill(NaN);
  return record;
}

/** Put back the pose a record's bones held under it, where they still hold what it wrote. */
export function releaseIK(transforms, record) {
  for (let k = 0; k < record.bones.length; k++) {
    if (!record.bones[k].alive) continue;
    const o = handleIndex(record.bones[k].entity) * 4, w = record.wrote, r = transforms.rotation;
    if (r[o] === w[k * 4] && r[o + 1] === w[k * 4 + 1] && r[o + 2] === w[k * 4 + 2] && r[o + 3] === w[k * 4 + 3]) {
      transforms.setRotation(record.bones[k].entity, record.input.subarray(k * 4, k * 4 + 4));
    }
  }
}

/** Whether the record's nodes are all still alive. */
export function ikAlive(record) {
  return record.bones.every((b) => b.alive) && (record.end?.alive ?? true)
    && (record.target.entity === undefined || record.target.alive) && (record.pole?.entity === undefined || record.pole.alive);
}

const worldPos = (world, node, out, o = 0) => {
  const m = handleIndex(node.entity) * 16;
  out[o] = world[m + 12]; out[o + 1] = world[m + 13]; out[o + 2] = world[m + 14];
  return out;
};
const worldRot = (world, node, out) => {
  mat4Decompose(scratch.pos, out, scratch.scale, world, handleIndex(node.entity) * 16);
  return out;
};
const sub = (a, ao, b, bo) => [a[ao] - b[bo], a[ao + 1] - b[bo + 1], a[ao + 2] - b[bo + 2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const unit = (a) => { const l = len(a); return l > EPSILON ? [a[0] / l, a[1] / l, a[2] / l] : null; };
const angle = (a, b) => Math.acos(Math.min(1, Math.max(-1, dot(unit(a) ?? [1, 0, 0], unit(b) ?? [1, 0, 0]))));
const clampCos = (x) => Math.acos(Math.min(1, Math.max(-1, x)));

/**
 * Turn `node`'s local rotation by `delta`, a rotation in world space, given
 * its world rotation `global`: local * global^-1 * delta * global.
 */
function turn(transforms, node, global, delta, out) {
  const o = handleIndex(node.entity) * 4, r = transforms.rotation;
  quatConjugate(q3, global);
  quatMultiply(q4, q3, delta);
  quatMultiply(q4, q4, global);
  q0[0] = r[o]; q0[1] = r[o + 1]; q0[2] = r[o + 2]; q0[3] = r[o + 3];
  quatMultiply(out, q0, q4);
  quatNormalize(out, out);
  transforms.setRotation(node.entity, out);
}

/**
 * Solve one record against the composed world matrices: the bones' local
 * rotations turned, and true when any was. The caller composes again.
 */
export function solveIK(transforms, record) {
  const world = transforms.world, rotation = transforms.rotation, n = record.bones.length;
  // What the solve depends on besides the pose: the target, the pole, the
  // weight, and where the chain hangs from.
  const sig = record.sig;
  sig.fill(0);
  if (record.target.entity !== undefined) worldPos(world, record.target, sig, 0); else sig.set(record.target, 0);
  if (record.pole) { if (record.pole.entity !== undefined) worldPos(world, record.pole, sig, 3); else sig.set(record.pole, 3); }
  sig[6] = record.weight;
  const parent = transforms.parent[handleIndex(record.bones[0].entity)];
  if (parent >= 0) sig.set(world.subarray(parent * 16, parent * 16 + 16), 7);
  // A bone still as the last solve left it holds no new pose.
  let fresh = false;
  for (let k = 0; k < n && !fresh; k++) {
    const o = handleIndex(record.bones[k].entity) * 4, w = record.wrote;
    fresh = rotation[o] !== w[k * 4] || rotation[o + 1] !== w[k * 4 + 1] || rotation[o + 2] !== w[k * 4 + 2] || rotation[o + 3] !== w[k * 4 + 3];
  }
  // Nothing new: what was written stands, untouched, so a still character stays still.
  if (!fresh && sig.every((v, c) => v === record.last[c])) return false;
  record.last.set(sig);
  // The pose underneath: a bone the animation or the app wrote is it; one
  // still as the solve left it goes back to what it held before.
  let restored = false;
  for (let k = 0; k < n; k++) {
    const o = handleIndex(record.bones[k].entity) * 4, w = record.wrote, i = record.input;
    if (rotation[o] === w[k * 4] && rotation[o + 1] === w[k * 4 + 1] && rotation[o + 2] === w[k * 4 + 2] && rotation[o + 3] === w[k * 4 + 3]) {
      transforms.setRotation(record.bones[k].entity, i.subarray(k * 4, k * 4 + 4));
      restored = true;
    } else {
      for (let c = 0; c < 4; c++) i[k * 4 + c] = rotation[o + c];
    }
  }
  if (restored) transforms.update();

  if (record.type === 'twoBone') solveTwoBone(transforms, record, sig);
  else solveAim(transforms, record, sig);
  for (let k = 0; k < n; k++) {
    const o = handleIndex(record.bones[k].entity) * 4;
    for (let c = 0; c < 4; c++) record.wrote[k * 4 + c] = rotation[o + c];
  }
  return true;
}

/**
 * Two bones reaching for a point: the hip and knee turned so the foot lands
 * on it, by the law of cosines (after Holden, "Simple Two Joint IK"), then
 * the whole leg turned about the hip-to-target line so the knee points at
 * the pole. Out of reach, the leg straightens toward it.
 */
function solveTwoBone(transforms, record, sig) {
  const world = transforms.world, [root, mid] = record.bones, w = record.weight;
  const a = worldPos(world, root, new Float64Array(3)), b = worldPos(world, mid, new Float64Array(3)), c = worldPos(world, record.end, new Float64Array(3));
  const t = [c[0] + (sig[0] - c[0]) * w, c[1] + (sig[1] - c[1]) * w, c[2] + (sig[2] - c[2]) * w];
  const ab = sub(b, 0, a, 0), cb = sub(c, 0, b, 0), ac = sub(c, 0, a, 0), at = sub(t, 0, a, 0);
  const lab = len(ab), lcb = len(cb);
  if (lab < EPSILON || lcb < EPSILON) return false;
  const lat = Math.min(Math.max(len(at), Math.abs(lab - lcb) + EPSILON, EPSILON), lab + lcb - EPSILON);
  const acab0 = angle(ac, ab), babc0 = angle([-ab[0], -ab[1], -ab[2]], cb), acat0 = angle(ac, at);
  const acab1 = clampCos((lcb * lcb - lab * lab - lat * lat) / (-2 * lab * lat));
  const babc1 = clampCos((lat * lat - lab * lab - lcb * lcb) / (-2 * lab * lcb));
  // The bend's axis: square to the leg's plane. A leg dead straight has no
  // plane; the pole's, or any, then.
  const pole = record.pole ? [sig[3] - a[0], sig[4] - a[1], sig[5] - a[2]] : null;
  const axis0 = unit(cross(ac, ab)) ?? (pole && unit(cross(ac, pole))) ?? unit(cross(ac, Math.abs(ac[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0])) ?? [1, 0, 0];
  const axis1 = unit(cross(ac, at));
  const r0 = quatSetAxisAngle(new Float64Array(4), axis0, acab1 - acab0);
  const r1 = quatSetAxisAngle(new Float64Array(4), axis0, babc1 - babc0);
  const r2 = axis1 ? quatSetAxisAngle(new Float64Array(4), axis1, acat0) : IDENTITY;
  const turnRoot = quatMultiply(new Float64Array(4), r2, r0);
  if (pole) {
    // Where the knee ends up, and the turn about the reach that takes it to the pole.
    const knee = rotateBy(turnRoot, ab), reach = unit(at);
    if (reach) {
      const u = unit(sub(knee, 0, scaled(reach, dot(knee, reach)), 0)), v = unit(sub(pole, 0, scaled(reach, dot(pole, reach)), 0));
      if (u && v) {
        const twist = quatSetAxisAngle(new Float64Array(4), reach, Math.atan2(dot(cross(u, v), reach), dot(u, v)) * w);
        quatMultiply(turnRoot, twist, turnRoot);
      }
    }
  }
  const aGlobal = worldRot(world, root, new Float64Array(4)), bGlobal = worldRot(world, mid, new Float64Array(4));
  turn(transforms, root, aGlobal, turnRoot, q1);
  turn(transforms, mid, bGlobal, r1, q2);
  return true;
}

/**
 * One bone turned so its `axis` points at the target, by no more than
 * `limit`, and by `weight` of that.
 */
function solveAim(transforms, record, sig) {
  const world = transforms.world, [node] = record.bones;
  const p = worldPos(world, node, new Float64Array(3));
  const global = worldRot(world, node, new Float64Array(4));
  const want = unit([sig[0] - p[0], sig[1] - p[1], sig[2] - p[2]]);
  if (!want) return false;
  const facing = rotateBy(global, record.axis);
  const delta = quatFromTo(new Float64Array(4), facing, want);
  const turnAngle = 2 * Math.acos(Math.min(1, Math.abs(delta[3])));
  const share = record.weight * (turnAngle > record.limit ? record.limit / turnAngle : 1);
  if (share < 1) quatSlerp(delta, IDENTITY, delta, share);
  turn(transforms, node, global, delta, q1);
  return true;
}

const scaled = (v, s) => [v[0] * s, v[1] * s, v[2] * s];

/** `v` turned by quaternion `q`. */
function rotateBy(q, v) {
  const [x, y, z, w] = q;
  const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + y * tz - z * ty, v[1] + w * ty + z * tx - x * tz, v[2] + w * tz + x * ty - y * tx];
}
