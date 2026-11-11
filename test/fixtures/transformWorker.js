// Worker side of the parallel job tests: the same shape as src/core/jobWorker.js.
import { parentPort } from 'node:worker_threads';
import { workerLoop, JOB_COMPOSE_TRANSFORMS, JOB_WORLD_BOUNDS, JOB_DRAW_DATA } from '../../src/core/jobs.js';
import { composeRange, columnsFromBuffers } from '../../src/scene/transformJob.js';
import { boundsRange, drawDataRange, renderableColumnsFromBuffers } from '../../src/scene/renderableJob.js';

const postReady = () => parentPort.postMessage({ type: 'ready' });
let reported = false;
const groups = {};

parentPort.on('message', (message) => {
  if (message.type !== 'init') return;
  const control = new Int32Array(message.control);
  const key = message.key ?? 'transforms';
  groups[key] = (key === 'renderables' ? renderableColumnsFromBuffers : columnsFromBuffers)(message.buffers);

  if (!reported) { reported = true; postReady(); }
  workerLoop(control, new Map([
    [JOB_COMPOSE_TRANSFORMS, (start, end, base) => composeRange(groups.transforms, base, start, end)],
    [JOB_WORLD_BOUNDS, (start, end, all) => boundsRange(groups.renderables, start, end, all === 1)],
    [JOB_DRAW_DATA, (start, end) => drawDataRange(groups.renderables, start, end)],
  ]), message.revision);
});
