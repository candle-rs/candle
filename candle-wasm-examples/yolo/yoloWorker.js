//load the candle yolo wasm module
import init, { Model } from "./build/m.js";

class Yolo {
  static instance = {};
  // Retrieve the YOLO model. When called for the first time,
  // this will load the model and save it for future use.
  static async getInstance(modelID, modelURL) {
    // load individual modelID only once
    if (!this.instance[modelID]) {
      await init();

      self.postMessage({ status: "loading model" });
      const modelRes = await fetch(modelURL);
      const yoloArrayBuffer = await modelRes.arrayBuffer();
      const weightsArrayU8 = new Uint8Array(yoloArrayBuffer);
      this.instance[modelID] = new Model(weightsArrayU8);
    } else {
      self.postMessage({ status: "model already loaded" });
    }
    return this.instance[modelID];
  }
}

self.addEventListener("message", async (event) => {
  const imageURL = event.data.imageURL;
  const modelURL = event.data.modelURL;
  const modelID = event.data.modelID;

  try {
    const yolo = await Yolo.getInstance(modelID, modelURL);

    self.postMessage({ status: "loading image" });
    const imgRes = await fetch(imageURL);
    const imgData = await imgRes.arrayBuffer();
    const imageArrayU8 = new Uint8Array(imgData);

    self.postMessage({ status: "running inference" });
    const bboxes = yolo.run(imageArrayU8);

    // Send the output back to the main thread as JSON
    self.postMessage({
      status: "complete",
      output: JSON.parse(bboxes),
    });
  } catch (e) {
    self.postMessage({ error: e });
  }
});
