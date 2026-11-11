// Job system self-check. Run: node test/jobs.test.js
//
// Real threads, not a simulation. node:worker_threads gives genuine parallelism
// over a SharedArrayBuffer, which is the only way to catch the failure this
// design can actually have: an item processed twice, or none.
//
// THIS SUITE SPENT ITS WHOLE LIFE TESTING THE SERIAL PATH. It spawned workers
// correctly and then dispatched in the same synchronous run, so `readyCount`
// -- which a worker raises by POSTING A MESSAGE -- was still zero every time,
// and every dispatch took the inline branch. Fourteen dispatches, none
// parallel. Nothing failed, because the fallback is bit-identical to the
// parallel path by design; that identity is what makes results useless as
// evidence here.
//
// So two things are asserted now that cannot be inferred from output: that
// `jobs.stats` says the parallel branch ran, and that more than one thread
// signed the work. Every test that means to be parallel awaits ready() first.

import assert from 'node:assert/strict';
import { Worker, threadId } from 'node:worker_threads';
import { JobSystem, runChunks, epochTag, CONTROL } from '../src/core/jobs.js';
import { sharedInt32Array, sharedMemoryAvailable } from '../src/core/shared.js';

let passed = 0;
async function test(name, fn) {
  await fn();
  passed++;
  console.log(`  ok  ${name}`);
}

const JOB_DOUBLE = 99;
const JOB_TOUCH = 98;
const JOB_SLOW = 97;
const WORKER_URL = new URL('./fixtures/jobWorker.js', import.meta.url);

/** This thread's signature, by the same rule the worker fixture uses. */
const WHO = threadId + 1;

function makeSystem(workerCount, count) {
  const input = sharedInt32Array(count);
  const output = sharedInt32Array(count);
  const touches = sharedInt32Array(count);
  const who = sharedInt32Array(count);
  for (let i = 0; i < count; i++) input[i] = i;

  const jobs = new JobSystem({
    parallelThreshold: 1,
    workerCount,
    createWorker: workerCount > 0 ? () => new Worker(WORKER_URL) : null,
  });

  // The dispatching thread runs the same handlers as the workers.
  jobs.register(JOB_DOUBLE, (start, end) => {
    for (let i = start; i < end; i++) output[i] = input[i] * 2;
  });
  jobs.register(JOB_TOUCH, (start, end) => {
    for (let i = start; i < end; i++) Atomics.add(touches, i, 1);
  });
  // Must match the worker fixture exactly -- it is the same job.
  jobs.register(JOB_SLOW, (start, end) => {
    for (let i = start; i < end; i++) {
      Atomics.add(touches, i, 1);
      who[i] = WHO;
    }
    const until = Date.now() + 2;
    while (Date.now() < until) { /* burn a chunk's worth of time */ }
  });

  jobs.setSharedData({
    input: input.buffer,
    output: output.buffer,
    touches: touches.buffer,
    who: who.buffer,
  });
  return { jobs, input, output, touches, who };
}

console.log('\nenvironment');

await test('shared memory is available under node', () => {
  assert.equal(sharedMemoryAvailable, true);
});

console.log('\nparallel execution');

await test('the workers actually check in, and more than one does the work', async () => {
  // The check the rest of this section rests on. Without it every assertion
  // below is about the serial path wearing the parallel path's name.
  const count = 8000;
  const { jobs, touches, who } = makeSystem(3, count);

  assert.equal(await jobs.ready(), true, 'all three workers checked in');

  // 16 chunks at roughly 2ms each: far more time than a thread takes to wake,
  // so the dispatching thread cannot possibly claim them all first.
  jobs.dispatch(JOB_SLOW, count, { chunkSize: 500 });

  assert.equal(jobs.stats.parallel, 1, 'the dispatch took the parallel branch');
  assert.equal(jobs.stats.inline, 0, 'and did not fall back');

  const threads = new Set(who);
  assert.ok(!threads.has(0), 'every item was claimed by some thread');
  assert.ok(threads.size > 1, `only thread ${[...threads]} ran any work`);
  assert.ok(threads.has(WHO), 'the dispatching thread participates too');

  for (let i = 0; i < count; i++) assert.equal(touches[i], 1, `item ${i}`);
  jobs.destroy();
});

await test('every item is processed exactly once', async () => {
  // THE property. Too few and there is a hole; too many and two threads wrote
  // the same slot, which for a transform buffer means a torn matrix.
  const count = 10_000;
  const { jobs, touches } = makeSystem(3, count);
  await jobs.ready();

  jobs.dispatch(JOB_TOUCH, count);
  assert.equal(jobs.stats.parallel, 1, 'ran on the workers, not inline');
  for (let i = 0; i < count; i++) {
    assert.equal(touches[i], 1, `item ${i} was touched ${touches[i]} times`);
  }
  jobs.destroy();
});

await test('results match the serial computation', async () => {
  const count = 5000;
  const { jobs, input, output } = makeSystem(3, count);
  await jobs.ready();

  jobs.dispatch(JOB_DOUBLE, count);
  assert.equal(jobs.stats.parallel, 1, 'ran on the workers, not inline');
  for (let i = 0; i < count; i++) assert.equal(output[i], input[i] * 2, `at ${i}`);
  jobs.destroy();
});

await test('repeated dispatches stay correct', async () => {
  // The counters are reset per dispatch. Forget one and the second dispatch
  // returns immediately with most of the work undone.
  const count = 2000;
  const { jobs, touches } = makeSystem(3, count);
  await jobs.ready();

  for (let round = 0; round < 5; round++) jobs.dispatch(JOB_TOUCH, count);
  assert.equal(jobs.stats.parallel, 5, 'every round ran on the workers');
  for (let i = 0; i < count; i++) assert.equal(touches[i], 5, `at ${i}`);
  jobs.destroy();
});

await test('workers pick up buffers published after the first', async () => {
  // What a growing scene does: new SharedArrayBuffers, published again. A
  // worker parked in its loop never read the second 'init' and went on
  // writing into the first buffers, so its share of the work vanished.
  const count = 8000;
  const { jobs } = makeSystem(3, count);
  await jobs.ready();

  const input = sharedInt32Array(count);
  const output = sharedInt32Array(count);
  const touches = sharedInt32Array(count);
  const who = sharedInt32Array(count);
  jobs.register(JOB_SLOW, (start, end) => {
    for (let i = start; i < end; i++) {
      Atomics.add(touches, i, 1);
      who[i] = WHO;
    }
    const until = Date.now() + 2;
    while (Date.now() < until) { /* burn a chunk's worth of time */ }
  });
  jobs.setSharedData({ input: input.buffer, output: output.buffer, touches: touches.buffer, who: who.buffer });

  // The first dispatch after a republish is how a waiting worker finds out;
  // it steps out to read the new buffers and the dispatching thread covers.
  jobs.dispatch(JOB_SLOW, count, { chunkSize: 500 });
  // Even this first one lands in the new buffers: a worker that wakes into it
  // checks the revision before claiming anything, and steps out instead.
  for (let i = 0; i < count; i++) assert.equal(touches[i], 1, `first dispatch, item ${i}`);
  await new Promise((resolve) => setTimeout(resolve, 50));
  touches.fill(0);
  who.fill(0);

  jobs.dispatch(JOB_SLOW, count, { chunkSize: 500 });
  for (let i = 0; i < count; i++) assert.equal(touches[i], 1, `item ${i} was touched ${touches[i]} times`);
  assert.ok(new Set(who).size > 1, 'the workers took part with the new buffers');
  jobs.destroy();
});

await test('more chunks than the cursor can count still covers every item once', async () => {
  // chunkSize 1 over 70,000 items is more chunk indices than the 16 bits
  // beside the epoch tag hold. The chunk is widened to fit rather than
  // letting the index run into the tag.
  const count = 70_000;
  const { jobs, touches } = makeSystem(3, count);
  await jobs.ready();
  jobs.dispatch(JOB_TOUCH, count, { chunkSize: 1 });
  for (let i = 0; i < count; i++) assert.equal(touches[i], 1, `item ${i}`);
  jobs.destroy();
});

await test('the epoch tag is the low 16 bits of the epoch', () => {
  assert.equal(epochTag(65536 + 5), epochTag(5));
  assert.notEqual(epochTag(6), epochTag(5));
});

await test('a tiny job is still correct', async () => {
  // Fewer items than threads: most threads claim nothing at all.
  const count = 3;
  const { jobs, touches } = makeSystem(3, count);
  await jobs.ready();

  jobs.dispatch(JOB_TOUCH, count);
  assert.equal(jobs.stats.parallel, 1, 'still the parallel branch');
  assert.deepEqual([...touches], [1, 1, 1]);
  jobs.destroy();
});

await test('an empty job does nothing and returns', async () => {
  const { jobs, touches } = makeSystem(2, 8);
  await jobs.ready();

  assert.equal(jobs.dispatch(JOB_TOUCH, 0), 0);
  assert.equal(jobs.stats.parallel + jobs.stats.inline, 0, 'no work, no dispatch');
  assert.equal(touches.reduce((a, b) => a + b, 0), 0);
  jobs.destroy();
});

console.log('\nserial fallback');

await test('no workers means the calling thread does everything', async () => {
  // The fallback is not a different algorithm -- identical results, one thread.
  const count = 1000;
  const { jobs, input, output } = makeSystem(0, count);
  assert.equal(jobs.parallel, false);
  assert.equal(await jobs.ready(), false, 'no workers is not readiness');

  jobs.dispatch(JOB_DOUBLE, count);
  assert.equal(jobs.stats.inline, 1, 'and the inline branch is what ran');
  for (let i = 0; i < count; i++) assert.equal(output[i], input[i] * 2, `at ${i}`);
  jobs.destroy();
});

console.log('\nchunk claiming');

await test('claiming stops at the item count, never past it', async () => {
  // runChunks in isolation: the cursor overshoots by design, and the guard is
  // what keeps a handler from being called with a range past the end.
  const control = sharedInt32Array(CONTROL.CONTROL_SLOTS);
  Atomics.store(control, CONTROL.ITEM_COUNT, 10);
  Atomics.store(control, CONTROL.CHUNK, 4);

  const ranges = [];
  runChunks(control, (start, end) => ranges.push([start, end]), 0, 0);

  assert.deepEqual(ranges, [[0, 4], [4, 8], [8, 10]], 'last chunk is clamped');
  assert.equal(Atomics.load(control, CONTROL.COMPLETED), 10);
});

await test('completion counts items, not chunks', async () => {
  // Waiting on the cursor instead would return while the final chunk was still
  // being written.
  const control = sharedInt32Array(CONTROL.CONTROL_SLOTS);
  Atomics.store(control, CONTROL.ITEM_COUNT, 7);
  Atomics.store(control, CONTROL.CHUNK, 3);
  runChunks(control, () => {}, 0, 0);

  assert.equal(Atomics.load(control, CONTROL.COMPLETED), 7);
  // The cursor counts CHUNKS and is claimed by compare-exchange, so unlike the
  // atomicAdd it replaced it can never be advanced past the end: 7 items in
  // threes is three claims, and the third is the one that runs out.
  assert.equal(Atomics.load(control, CONTROL.CURSOR), 3, 'three chunks claimed, none beyond');
});

await test('a thread carrying a stale dispatch claims nothing', async () => {
  // The race this cursor layout exists for, and the one that made the parallel
  // transform check fail three runs in eight once it started running for real.
  //
  // A worker reads its dispatch's arguments, is descheduled, and resumes after
  // the dispatching thread has finished that dispatch and started another. The
  // cursor it finds belongs to the NEW dispatch. With a plain counter it took
  // those chunks and ran them with the OLD arguments, then credited the new
  // dispatch's completion count -- so the dispatch returned believing work was
  // done that its handler never touched, and a range of transforms kept
  // whatever was in the buffer.
  const control = sharedInt32Array(CONTROL.CONTROL_SLOTS);
  Atomics.store(control, CONTROL.ITEM_COUNT, 10);
  Atomics.store(control, CONTROL.CHUNK, 4);
  Atomics.store(control, CONTROL.CURSOR, epochTag(5));      // dispatch 5 is live

  const stale = [];
  runChunks(control, (s, e) => stale.push([s, e]), 0, 0, 4);
  assert.deepEqual(stale, [], 'a thread still serving dispatch 4 takes nothing');
  assert.equal(Atomics.load(control, CONTROL.COMPLETED), 0, 'and credits nothing');

  // And the dispatch's own threads are untouched by the refusal.
  const live = [];
  runChunks(control, (s, e) => live.push([s, e]), 0, 0, 5);
  assert.deepEqual(live, [[0, 4], [4, 8], [8, 10]]);
  assert.equal(Atomics.load(control, CONTROL.COMPLETED), 10);
});


// ------------------------------------------- parallel transform composition

console.log('\nparallel transforms');

const { TransformStore } = await import('../src/scene/transform.js');
const { HandleAllocator } = await import('../src/core/handle.js');
const { composeRange } = await import('../src/scene/transformJob.js');
const { JOB_COMPOSE_TRANSFORMS } = await import('../src/core/jobs.js');

/** A wide, deep hierarchy: 4 levels, branching 6. */
function buildHierarchy(store, ids) {
  const roots = [];
  for (let r = 0; r < 6; r++) {
    const root = ids.alloc();
    store.add(root, { position: [r * 3, 0, 0] });
    roots.push(root);

    for (let a = 0; a < 6; a++) {
      const child = ids.alloc();
      store.add(child, { position: [0, 1, 0], parent: root });
      for (let b = 0; b < 6; b++) {
        const grand = ids.alloc();
        store.add(grand, { position: [0, 0, 1], parent: child });
        for (let c = 0; c < 3; c++) {
          const leaf = ids.alloc();
          store.add(leaf, { position: [0.5, 0, 0], parent: grand });
        }
      }
    }
  }
  return roots;
}

await test('depth levels are recorded and cover every node', () => {
  const ids = new HandleAllocator(2048);
  const store = new TransformStore(2048);
  buildHierarchy(store, ids);
  store.update();

  assert.equal(store.levelCount, 4, 'root, child, grandchild, leaf');
  assert.equal(store.levelStart[0], 0);
  assert.equal(store.levelStart[store.levelCount], store.orderCount, 'levels tile the order');
});

await test('parallel composition matches serial, matrix for matrix', async () => {
  // THE test. Depth ordering is what makes a level safe to split; if that
  // reasoning is wrong, a child composes against a stale parent and the two
  // paths diverge.
  const ids = new HandleAllocator(2048);
  const serialIds = new HandleAllocator(2048);

  const parallelStore = new TransformStore(2048);
  const serialStore = new TransformStore(2048);
  buildHierarchy(parallelStore, ids);
  buildHierarchy(serialStore, serialIds);

  serialStore.update();

  const jobs = new JobSystem({ parallelThreshold: 1,
    workerCount: 3,
    createWorker: () => new Worker(new URL('./fixtures/transformWorker.js', import.meta.url)),
  });
  jobs.register(JOB_COMPOSE_TRANSFORMS,
    (start, end, base) => composeRange(parallelStore, base, start, end));
  jobs.setSharedData(parallelStore.sharedBuffers());
  assert.equal(await jobs.ready(), true, 'the transform workers checked in');

  parallelStore.updateParallel(jobs);
  assert.ok(jobs.stats.parallel > 0, 'at least one level went wide');

  assert.equal(parallelStore.orderCount, serialStore.orderCount);
  for (let i = 0; i < parallelStore.world.length; i++) {
    assert.equal(parallelStore.world[i], serialStore.world[i], `world[${i}] diverged`);
  }
  jobs.destroy();
});

await test('world boxes, draw data and cull bounds on the workers match the same jobs run here, bit for bit', async () => {
  // 3,000 renderables on 1,000 nodes, a third of the nodes moved, some
  // skinned, some in LOD groups, some hidden: each job run over a listed set
  // and (for the boxes) over every renderable whose node moved, once across
  // three workers and once inline on a copy.
  const { sharedFloat32Array, sharedInt32Array, sharedUint32Array, sharedUint8Array } = await import('../src/core/shared.js');
  const { JOB_WORLD_BOUNDS, JOB_DRAW_DATA } = await import('../src/core/jobs.js');
  const { boundsRange, drawDataRange, renderableBuffers } = await import('../src/scene/renderableJob.js');
  const N = 3000, NODES = 1000;
  let seed = 7;
  const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const columns = () => ({
    items: sharedUint32Array(N), localMin: sharedFloat32Array(N * 3), localMax: sharedFloat32Array(N * 3),
    worldMin: sharedFloat32Array(N * 3), worldMax: sharedFloat32Array(N * 3), matrixSlot: sharedUint32Array(N),
    tint: sharedUint32Array(N * 3), world: sharedFloat32Array(NODES * 16), moved: sharedUint8Array(NODES),
    drawData: sharedFloat32Array(N * 32), skin: sharedInt32Array(N), paletteOffsets: sharedUint32Array(16),
    boundsData: sharedFloat32Array(N * 12), coverage: sharedFloat32Array(N * 2), lodSlot: sharedInt32Array(N),
    lodSphere: sharedFloat32Array(N * 4), hidden: sharedUint8Array(N),
  });
  const wide = columns();
  wide.drawWords = new Uint32Array(wide.drawData.buffer);
  for (let i = 0; i < N * 3; i++) {
    wide.localMin[i] = random() * 2 - 2; wide.localMax[i] = wide.localMin[i] + random() * 3;
    wide.tint[i] = (random() * 0xffffffff) >>> 0;
  }
  for (let i = 0; i < N; i++) {
    wide.matrixSlot[i] = Math.floor(random() * NODES);
    wide.skin[i] = random() < 0.2 ? Math.floor(random() * 16) : -1;
    wide.lodSlot[i] = random() < 0.3 ? Math.floor(random() * NODES) : -1;
    wide.hidden[i] = random() < 0.1 ? 1 : 0;
  }
  for (let i = 0; i < 16; i++) wide.paletteOffsets[i] = i * 31 + 1;
  for (let i = 0; i < N * 4; i++) wide.lodSphere[i] = random() * 3;
  for (let i = 0; i < N * 2; i++) wide.coverage[i] = random();
  for (let i = 0; i < NODES * 16; i++) wide.world[i] = random() * 4 - 2;
  for (let i = 0; i < NODES; i++) wide.moved[i] = random() < 0.33 ? 1 : 0;
  const here = columns();
  for (const key of Object.keys(here)) here[key].set(wide[key]);
  here.drawWords = new Uint32Array(here.drawData.buffer);
  let listed = 0;
  for (let i = 0; i < N; i++) if (random() < 0.5) wide.items[listed++] = i;
  here.items.set(wide.items);

  const jobs = new JobSystem({ parallelThreshold: 1, workerCount: 3,
    createWorker: () => new Worker(new URL('./fixtures/transformWorker.js', import.meta.url)),
  });
  jobs.register(JOB_WORLD_BOUNDS, (start, end, all) => boundsRange(wide, start, end, all === 1));
  jobs.register(JOB_DRAW_DATA, (start, end) => drawDataRange(wide, start, end));
  jobs.setSharedData(renderableBuffers(wide), null, 'renderables');
  assert.equal(await jobs.ready(), true, 'the workers checked in');

  jobs.dispatch(JOB_WORLD_BOUNDS, N, { arg0: 1 });
  boundsRange(here, 0, N, true);
  jobs.dispatch(JOB_WORLD_BOUNDS, listed);
  boundsRange(here, 0, listed, false);
  jobs.dispatch(JOB_DRAW_DATA, listed);
  drawDataRange(here, 0, listed);
  assert.equal(jobs.stats.parallel, 3, 'all three went wide');

  for (const key of ['worldMin', 'worldMax', 'drawData', 'boundsData']) {
    const a = new Uint32Array(wide[key].buffer), b = new Uint32Array(here[key].buffer);
    for (let i = 0; i < a.length; i++) assert.equal(a[i], b[i], `${key}[${i}] diverged`);
  }
  // And they did something: each listed renderable has draw data, and only those.
  const isListed = new Set(wide.items.subarray(0, listed));
  for (let i = 0; i < N; i++) assert.equal(wide.drawData[i * 32] !== 0, isListed.has(i), `renderable ${i}`);
  jobs.destroy();
});

console.log(`\n${passed} checks passed\n`);
