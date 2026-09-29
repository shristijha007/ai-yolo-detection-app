import cv from "@techstark/opencv-js";
import { Tensor } from "onnxruntime-web/webgpu";

/**
 * Pre-process input image.
 *
 * @param {cv.Mat} srcMat - input image Mat
 * @param {[Number, Number]} outputSize - Output size [width, height]
 * @param {String} imgszType - Processing type, "dynamic" or "zeroPad"
 * @returns {[ort.Tensor, Number, Number]} - return [inputTensor, xRatio, yRatio]
 */
const preProcessImage = (srcMat, outputSize, imgszType) => {
  let preProcessedMat, xRatio, yRatio, inputTensor, div_width, div_height;

  if (imgszType === "dynamic") {
    [preProcessedMat, xRatio, yRatio, div_width, div_height] =
      dynamicInputProcess(srcMat, outputSize);

    // create input tensor
    inputTensor = new Tensor(
      "float32",
      preProcessedMat.data32F,
      [1, 3, div_height, div_width] // [batch, channel, height, width]
    );
  } else if (imgszType === "zeroPad") {
    const modelDefaultInputSize = [640, 640]; // yolo model default input size
    [preProcessedMat, xRatio, yRatio] = zeroPadInputProcess(
      srcMat,
      modelDefaultInputSize,
      outputSize
    );

    // create input tensor
    inputTensor = new Tensor(
      "float32",
      preProcessedMat.data32F,
      [1, 3, modelDefaultInputSize[1], modelDefaultInputSize[0]] // [batch, channel, height, width]
    );
  }
  preProcessedMat.delete();

  return [inputTensor, xRatio, yRatio];
};

/**
 * Pre process input image.
 *
 * Zero padding to square and resize to input size.
 *
 * @param {cv.Mat} srcMat - Pre process yolo model input image.
 * @param {Number} modelSize - Yolo model image size input [width, height].
 * @param {Number} outputSize - Overlay image size [width, height].
 * @returns {[cv.Mat, Number, Number]} Processed input mat, xRatio, yRatio.
 */
const zeroPadInputProcess = (srcMat, modelSize, outputSize) => {
  cv.cvtColor(srcMat, srcMat, cv.COLOR_RGBA2RGB);

  // Resize to dimensions divisible by 32
  const [div_width, div_height] = divStride(32, srcMat.cols, srcMat.rows);
  cv.resize(srcMat, srcMat, new cv.Size(div_width, div_height));

  // Padding to square
  const max_dim = Math.max(div_width, div_height);
  const right_pad = max_dim - div_width;
  const bottom_pad = max_dim - div_height;
  cv.copyMakeBorder(
    srcMat,
    srcMat,
    0,
    bottom_pad,
    0,
    right_pad,
    cv.BORDER_CONSTANT,
    new cv.Scalar(0, 0, 0)
  ); // padding to square

  // Resize to input size and normalize to [0, 1]
  const preProcessed = cv.blobFromImage(
    srcMat,
    1 / 255.0,
    { width: modelSize[0], height: modelSize[1] },
    [0, 0, 0, 0],
    false,
    false
  );

  const xRatio = (outputSize[0] / div_width) * (max_dim / outputSize[0]);
  const yRatio = (outputSize[1] / div_height) * (max_dim / outputSize[1]);

  return [preProcessed, xRatio, yRatio];
};

/**
 * Pre process input image for dynamic input model.
 *
 * @param {cv.Mat} mat - Pre process yolo model input image.
 * @returns {[cv.mat, Number, Number ...]} Processed input mat, xRatio, yRatio, div_width, div_height.
 */
const dynamicInputProcess = (mat, outputSize) => {
  cv.cvtColor(mat, mat, cv.COLOR_RGBA2RGB);

  // resize image to divisible by 32
  const [div_width, div_height] = divStride(32, mat.cols, mat.rows);

  // resize, normalize to [0, 1]
  const preProcessedMat = cv.blobFromImage(
    mat,
    1 / 255.0,
    { width: div_width, height: div_height },
    [0, 0, 0, 0],
    false,
    false
  );
  const xRatio = outputSize[0] / div_width; // scale factor for overlay
  const yRatio = outputSize[1] / div_height;
  return [preProcessedMat, xRatio, yRatio, div_width, div_height];
};

/**
 * Return height and width are divisible by stride.
 * @param {Number} stride - Stride value.
 * @param {Number} width - Image width.
 * @param {Number} height - Image height.
 * @returns {[Number]}[width, height] divisible by stride.
 **/
const divStride = (stride, width, height) => {
  width =
    width % stride >= stride / 2
      ? (Math.floor(width / stride) + 1) * stride
      : Math.floor(width / stride) * stride;

  height =
    height % stride >= stride / 2
      ? (Math.floor(height / stride) + 1) * stride
      : Math.floor(height / stride) * stride;

  return [width, height];
};

/**
 * Ultralytics default color palette https://ultralytics.com/.
 *
 * This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
 * RGB values.
 */
class Colors {
  static palette = [
    "042AFF",
    "0BDBEB",
    "F3F3F3",
    "00DFB7",
    "111F68",
    "FF6FDD",
    "FF444F",
    "CCED00",
    "00F344",
    "BD00FF",
    "00B4FF",
    "DD00BA",
    "00FFFF",
    "26C000",
    "01FFB3",
    "7D24FF",
    "7B0068",
    "FF1B6C",
    "FC6D2F",
    "A2FF0B",
  ].map((c) => Colors.hex2rgba(`#${c}`));
  static n = Colors.palette.length;
  static cache = {}; // Cache for colors

  static hex2rgba(h, alpha = 1.0) {
    return [
      parseInt(h.slice(1, 3), 16),
      parseInt(h.slice(3, 5), 16),
      parseInt(h.slice(5, 7), 16),
      alpha,
    ];
  }

  static getColor(i, alpha = 1.0, bgr = false) {
    const key = `${i}-${alpha}-${bgr}`;
    if (Colors.cache[key]) {
      return Colors.cache[key];
    }
    const c = Colors.palette[i % Colors.n];
    const rgba = [...c.slice(0, 3), alpha];
    const result = bgr ? [rgba[2], rgba[1], rgba[0], rgba[3]] : rgba;
    Colors.cache[key] = result;
    return result;
  }
}

export { preProcessImage, Colors };
