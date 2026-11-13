// Scene snapshots (scene.snapshot, scene.restore): the state of every node
// under a root, as plain data that JSON.stringify keeps -- where each is,
// whether it is shown, its tint, glow, hook params and outline, and what its
// model is playing -- and that state put back, by where each node sits under
// the root and what it is called, in this session or after a reload that
// loads the same models into the same places. For checkpoints, replays,
// rewind and an editor's undo. A game's own state -- health, inventory, the
// wave -- is the game's to save.

import { handleIndex } from '../core/handle.js';
import { NO_PARENT } from './transform.js';

export const SNAPSHOT_VERSION = 1;

/** Each slot's children, in the order they were made: one pass, not a scan per node. */
function childrenBySlot(transforms) {
  const children = new Map();
  for (let c = 0; c < transforms._high; c++) {
    if (!transforms.used[c]) continue;
    const p = transforms.parent[c];
    let list = children.get(p);
    if (!list) children.set(p, (list = []));
    list.push(c);
  }
  for (const list of children.values()) list.sort((a, b) => transforms.born[a] - transforms.born[b]);
  return children;
}

/** Path segments for siblings: the name, '#' for none, and '@k' on the k-th repeat of either. */
function segments(scene, slots) {
  const seen = new Map();
  return slots.map((slot) => {
    const name = scene._names[slot] || '#';
    const k = seen.get(name) ?? 0;
    seen.set(name, k + 1);
    return k === 0 ? name : `${name}@${k}`;
  });
}

const rounded = (array, from, n) => Array.from(array.subarray(from, from + n));

export function snapshotScene(scene, root) {
  const t = scene.transforms, children = childrenBySlot(t), nodes = [];
  const visit = (slot, path) => {
    const entry = { path, position: rounded(t.position, slot * 3, 3), rotation: rounded(t.rotation, slot * 4, 4), scale: rounded(t.scale, slot * 3, 3) };
    if (scene._hiddenSelf[slot] === 1) entry.visible = false;
    const own = scene._tints.get(slot);
    if (own) {
      if (own[0] !== 1 || own[1] !== 1 || own[2] !== 1) entry.tint = rounded(own, 0, 3);
      if (own[3] !== 0 || own[4] !== 0 || own[5] !== 0) entry.glow = rounded(own, 3, 3);
      if (own[6] !== 0 || own[7] !== 0) entry.hookParams = rounded(own, 6, 2);
    }
    const outline = scene.outlines.get(slot);
    if (outline) entry.outline = { color: Array.from(outline.color), width: outline.width, through: outline.through };
    const player = scene._players.get(scene.entities.handleAt(slot));
    if (player) entry.animation = player._snapshot();
    nodes.push(entry);
    const kids = children.get(slot) ?? [];
    const names = segments(scene, kids);
    kids.forEach((kid, k) => visit(kid, path === '' ? names[k] : `${path}/${names[k]}`));
  };
  if (root) {
    visit(handleIndex(root.entity), '');
  } else {
    const tops = children.get(NO_PARENT) ?? [];
    const names = segments(scene, tops);
    tops.forEach((slot, k) => visit(slot, names[k]));
  }
  return { version: SNAPSHOT_VERSION, nodes };
}

export function restoreScene(scene, snapshot, root) {
  if (snapshot?.version !== SNAPSHOT_VERSION || !Array.isArray(snapshot.nodes)) {
    throw new Error(`restore: not a snapshot scene.snapshot made (version ${SNAPSHOT_VERSION})`);
  }
  const t = scene.transforms, children = childrenBySlot(t);
  // Every node under the root by its path, as snapshot names them.
  const bySlot = new Map();
  const walk = (slot, path) => {
    bySlot.set(path, slot);
    const kids = children.get(slot) ?? [];
    const names = segments(scene, kids);
    kids.forEach((kid, k) => walk(kid, path === '' ? names[k] : `${path}/${names[k]}`));
  };
  if (root) walk(handleIndex(root.entity), '');
  else {
    const tops = children.get(NO_PARENT) ?? [];
    const names = segments(scene, tops);
    tops.forEach((slot, k) => walk(slot, names[k]));
  }
  const missing = [];
  let restored = 0;
  for (const entry of snapshot.nodes) {
    const slot = bySlot.get(entry.path);
    if (slot === undefined) { missing.push(entry.path); continue; }
    const node = scene.node(scene.entities.handleAt(slot));
    node.setPosition(...entry.position).setRotation(entry.rotation).setScale(...entry.scale);
    node.setVisible(entry.visible !== false);
    node.setTint(entry.tint ?? [1, 1, 1]).setGlow(entry.glow ?? [0, 0, 0]).setHookParams(...(entry.hookParams ?? [0, 0]));
    if (entry.outline) node.setOutline(entry.outline.color, { width: entry.outline.width, through: entry.outline.through });
    else node.setOutline(null);
    const player = scene._players.get(node.entity);
    if (player && entry.animation) player._restore(entry.animation);
    restored++;
  }
  return { restored, missing };
}
