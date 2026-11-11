// Browser worker entry for the job system.
//
// Deliberately identical in shape to the Node test fixture: receive the shared
// buffers, register the same handlers the main thread registers, sit in
// workerLoop. Any logic that lived here would be logic the serial path does not
// run, and the two would eventually disagree.

import { workerLoop, JOB_COMPOSE_TRANSFORMS, JOB_WORLD_BOUNDS, JOB_DRAW_DATA } from './jobs.js';
import { composeRange, columnsFromBuffers } from '../scene/transformJob.js';
import { boundsRange, drawDataRange, renderableColumnsFromBuffers } from '../scene/renderableJob.js';

const postReady = () => self.postMessage({ type: 'ready' });
// Once. A worker gets a fresh 'init' every time a scene republishes its
// buffers, and it is the same worker checking in.
let reported = false;
/** The columns each named set of buffers was last published as. */
const groups = {};

self.addEventListener('message', (event) => {
  const message = event.data;
  if (message.type !== 'init') return;

  const control = new Int32Array(message.control);
  // Each named set replaces only its own: the others stay as they were.
  const key = message.key ?? 'transforms';
  groups[key] = (key === 'renderables' ? renderableColumnsFromBuffers : columnsFromBuffers)(message.buffers);

  if (!reported) { reported = true; postReady(); }
  workerLoop(control, new Map([
    [JOB_COMPOSE_TRANSFORMS, (start, end, base) => composeRange(groups.transforms, base, start, end)],
    [JOB_WORLD_BOUNDS, (start, end, all) => boundsRange(groups.renderables, start, end, all === 1)],
    [JOB_DRAW_DATA, (start, end) => drawDataRange(groups.renderables, start, end)],
  ]), message.revision);
});
