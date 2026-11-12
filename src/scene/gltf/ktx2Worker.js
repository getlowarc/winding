// Browser worker that transcodes KTX2 images for the glTF loader (images.js):
// one file a message, its bytes transferred in and its levels transferred back.
// All the work is transcodeKTX2's, which the tests run directly.

import { transcodeKTX2 } from '../ktx2.js';

self.addEventListener('message', ({ data: { bytes, target, gpu } }) => {
  try {
    const image = transcodeKTX2(bytes, target, { gpu });
    // Levels left as UASTC with no zstd to undo share the file's one buffer: listed once.
    self.postMessage(image, [...new Set(image.levels.map((level) => level.data.buffer))]);
  } catch (error) {
    self.postMessage({ error: error.message });
  }
});
