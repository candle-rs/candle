export async function extractEmbeddings(
  worker,
  weightsURL,
  tokenizerURL,
  configURL,
  modelID,
  sentences,
  updateStatus,
  normalize_embeddings = true
) {
  return new Promise((resolve, reject) => {
    worker.postMessage({
      weightsURL,
      tokenizerURL,
      configURL,
      modelID,
      sentences,
      normalize_embeddings,
    });
    function messageHandler(event) {
      if ("error" in event.data) {
        worker.removeEventListener("message", messageHandler);
        reject(new Error(event.data.error));
      }
      if (event.data.status === "complete") {
        worker.removeEventListener("message", messageHandler);
        resolve(event.data);
      }
      if (updateStatus) updateStatus(event.data);
    }
    worker.addEventListener("message", messageHandler);
  });
}

export async function generateText(
  worker,
  weightsURL,
  tokenizerURL,
  configURL,
  modelID,
  prompt,
  params,
  updateStatus
) {
  return new Promise((resolve, reject) => {
    worker.postMessage({
      weightsURL,
      tokenizerURL,
      configURL,
      modelID,
      prompt,
      params,
    });
    function messageHandler(event) {
      if ("error" in event.data) {
        worker.removeEventListener("message", messageHandler);
        reject(new Error(event.data.error));
      }
      if (event.data.status === "complete") {
        worker.removeEventListener("message", messageHandler);
        resolve(event.data);
      }
      if (updateStatus) updateStatus(event.data);
    }
    worker.addEventListener("message", messageHandler);
  });
}

export const MODELS = {
  t5_small_quantized: {
    size: "64.4 MB",
    base_url: "https://huggingface.co/lmz/candle-quantized-t5/resolve/main/",
    model: "model.gguf",
    tokenizer: "tokenizer.json",
    config: "config.json",
    tasks: {
      translation_en_to_de: {
        prefix: "translate English to German: ",
        max_length: 300,
      },
      translation_en_to_fr: {
        prefix: "translate English to French: ",
        max_length: 300,
      },
      translation_en_to_ro: {
        prefix: "translate English to Romanian: ",
        max_length: 300,
      },
      summarization: { prefix: "summarize: ", max_length: 200 },
    },
  },
  t5_small: {
    size: "242 MB",
    base_url: "https://huggingface.co/t5-small/resolve/main/",
    model: "model.safetensors",
    tokenizer: "tokenizer.json",
    config: "config.json",
    tasks: {
      translation_en_to_de: {
        prefix: "translate English to German: ",
        max_length: 300,
      },
      translation_en_to_fr: {
        prefix: "translate English to French: ",
        max_length: 300,
      },
      translation_en_to_ro: {
        prefix: "translate English to Romanian: ",
        max_length: 300,
      },
      summarization: { prefix: "summarize: ", max_length: 200 },
    },
  },
  flan_t5_small: {
    size: "308 MB",
    base_url:
      "https://huggingface.co/google/flan-t5-small/resolve/refs%2Fpr%2F14/",
    model: "model.safetensors",
    tokenizer: "tokenizer.json",
    config: "config.json",
    tasks: {
      translation_en_to_de: {
        prefix: "translate English to German: ",
        max_length: 300,
      },
      translation_en_to_fr: {
        prefix: "translate English to French: ",
        max_length: 300,
      },
      translation_en_to_ro: {
        prefix: "translate English to Romanian: ",
        max_length: 300,
      },
      summarization: { prefix: "summarize: ", max_length: 200 },
    },
  },
  flan_t5_base_quantized: {
    size: "263 MB",
    base_url: "https://huggingface.co/lmz/candle-quantized-t5/resolve/main/",
    model: "model-flan-t5-base.gguf",
    tokenizer: "tokenizer.json",
    config: "config-flan-t5-base.json",
    tasks: {
      translation_en_to_de: {
        prefix: "translate English to German: ",
        max_length: 300,
      },
      translation_en_to_fr: {
        prefix: "translate English to French: ",
        max_length: 300,
      },
      translation_en_to_ro: {
        prefix: "translate English to Romanian: ",
        max_length: 300,
      },
      summarization: { prefix: "summarize: ", max_length: 200 },
    },
  },
  coedit_large_quantized: {
    size: "643 MB",
    base_url: "https://huggingface.co/jbochi/candle-coedit-quantized/resolve/main/",
    model: "model.gguf",
    tokenizer: "tokenizer.json",
    config: "config.json",
    tasks: {
      fluency: {
        prefix: "Fix the grammar: ",
        max_length: 300,
      },
      coherence: {
        prefix: "Rewrite to make this easier to understand: ",
        max_length: 300,
      },
      simplification: {
        prefix: "Paraphrase this: ",
        max_length: 300,
      },
      formalization: {
        prefix: "Write this more formally: ",
        max_length: 300,
      },
      neutralize: {
        prefix: "Write in a more neutral way: ",
        max_length: 300,
      },
    },
  },
};

export function getModelInfo(id, taskID) {
  const model = MODELS[id];
  return {
    modelURL: model.base_url + model.model,
    configURL: model.base_url + model.config,
    tokenizerURL: model.base_url + model.tokenizer,
    maxLength: model.tasks[taskID].max_length,
  };
}
