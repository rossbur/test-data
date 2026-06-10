// --- MODULE INTEROPERABILITY BRIDGE ---
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Unified path resolver logic for ESM and CommonJS
const __dirname = typeof __dirname !== 'undefined' 
  ? __dirname 
  : path.dirname(fileURLToPath(import.meta.url));
// --------------------------------------

import tf from '@tensorflow/tfjs-node';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = typeof __dirname !== 'undefined' 
  ? __dirname 
  : path.dirname(fileURLToPath(import.meta.url));

// 1. COMPREHENSIVE FILE TYPE REGEX REGISTRY
const EXTENSIONS = {
  TEXT: /\.(txt|csv|tsv|json|xml|yaml|yml)$/i,
  IMAGE: /\.(png|jpe?g|jpg|webp|gif|bmp|tiff|svg)$/i
};

/**
 * Scans a folder and groups all file types into matching, naturally sorted pairs.
 */
async function ingestAllDataTypes(relativeFolder) {
  const targetDir = path.resolve(__dirname, relativeFolder);
  const rawItems = await fs.readdir(targetDir);

  // Filter into strict type buckets
  const dataFiles = rawItems.filter(file => EXTENSIONS.TEXT.test(file));
  const photoFiles = rawItems.filter(file => EXTENSIONS.IMAGE.test(file));

  // NATURAL SORTING: Forces 'img2.png' to sit before 'img10.png' despite mismatched names
  const naturalSort = (a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
  
  return {
    basePath: targetDir,
    data: dataFiles.sort(naturalSort),
    photos: photoFiles.sort(naturalSort)
  };
}

/**
 * Text Preprocessing Layer: Cleans string layout and returns fixed-length token IDs.
 */
function processTextToTokens(rawText, vocabMap, maxLen = 150) {
  const tokens = rawText.toLowerCase().replace(/[^a-z0-9 ]/g, '').split(/\s+/);
  const sequence = tokens.map(t => vocabMap[t] || 0); // 0 reserved for Out-Of-Vocabulary (OOV)
  
  return sequence.length < maxLen
    ? sequence.concat(new Array(maxLen - sequence.length).fill(0))
    : sequence.slice(0, maxLen);
}

/**
 * Image Preprocessing Layer: Converts any 2D binary buffer into normalized 3D/4D tensors.
 */
async function processImageToTensor(absolutePath, targetShape =) {
  const imageBuffer = await fs.readFile(absolutePath);
  
  // tf.node.decodeImage parses PNG, JPEG, BMP, and GIF directly from raw bytes
  let tensor = tf.node.decodeImage(imageBuffer, targetShape[2]); 
  
  // Resize and normalize tensor values between 0.0 and 1.0
  tensor = tf.image.resizeBilinear(tensor, [targetShape[0], targetShape[1]]);
  return tensor.div(255.0);
}

/**
 * Orchestrator: Combines inputs into unified, batched tensor arrays.
 */
export async function buildUnifiedPipeline(folderPath, vocabulary) {
  const manifest = await ingestAllDataTypes(folderPath);
  
  const imageTensorStack = [];
  const textTensorStack = [];

  const totalPairs = Math.min(manifest.data.length, manifest.photos.length);

  for (let i = 0; i < totalPairs; i++) {
    const photoPath = path.join(manifest.basePath, manifest.photos[i]);
    const dataPath = path.join(manifest.basePath, manifest.data[i]);

    // Process image/photo data channel
    const imgTensor = await processImageToTensor(photoPath);
    imageTensorStack.push(imgTensor);

    // Process matching text/data payload channel
    const rawText = await fs.readFile(dataPath, 'utf8');
    const tokenArray = processTextToTokens(rawText, vocabulary);
    textTensorStack.push(tf.tensor1d(tokenArray, 'int32'));
  }

  // Pack arrays into solid training blocks
  const batchedImages = tf.stack(imageTensorStack);
  const batchedTexts = tf.stack(textTensorStack);

  return {
    X_images: batchedImages,
    X_texts: batchedTexts,
    totalRecords: totalPairs
  };
}
