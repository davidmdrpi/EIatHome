// Model layers shared by Track A (frame prediction) and Track B (the token model), in their
// own module so tests can build them outside the engine worker. Loads TensorFlow.js itself, so
// importing this module is enough.

import '../vendor/tfjs/tf.min.js';
import { TOKEN_GRU_UNITS, TOKEN_WINDOW, matchedTokenWidth } from './models.js';
import { T } from './signals.js';

const tf = globalThis.tf;

// A learned position embedding, added to each time step (attention has no built-in order).
export class PositionEmbedding extends tf.layers.Layer {
  static className = 'EIatHomePositionEmbedding';

  constructor(config) {
    super(config);
    this.maxLength = config.maxLength;
    this.seed = config.seed;
  }

  build(inputShape) {
    const width = inputShape[inputShape.length - 1];
    this.table = this.addWeight('table', [this.maxLength, width], 'float32', tf.initializers.randomNormal({ stddev: 0.02, seed: this.seed }));
    this.built = true;
  }

  call(inputs) {
    return tf.tidy(() => {
      const x = Array.isArray(inputs) ? inputs[0] : inputs;
      return x.add(this.table.read().slice([0, 0], [x.shape[1], -1]));
    });
  }

  computeOutputShape(inputShape) {
    return inputShape;
  }

  getConfig() {
    return { ...super.getConfig(), maxLength: this.maxLength, seed: this.seed };
  }
}

// Layer normalization over the last axis, with a learned scale and offset. TF.js's own layer
// tiles them to the input's shape, and the gradient of a tile is one slice and one add per
// repetition: thousands of kernels per step for a batch of sequences. Broadcasting computes
// the same function with the same parameters (epsilon 0.001, as TF.js's default) far faster.
export class LayerNorm extends tf.layers.Layer {
  static className = 'EIatHomeLayerNorm';

  build(inputShape) {
    const d = inputShape[inputShape.length - 1];
    this.gamma = this.addWeight('gamma', [d], 'float32', tf.initializers.ones());
    this.beta = this.addWeight('beta', [d], 'float32', tf.initializers.zeros());
    this.built = true;
  }

  call(inputs) {
    return tf.tidy(() => {
      const x = Array.isArray(inputs) ? inputs[0] : inputs;
      const { mean, variance } = tf.moments(x, -1, true);
      return x.sub(mean).mul(tf.rsqrt(variance.add(1e-3))).mul(this.gamma.read()).add(this.beta.read());
    });
  }

  computeOutputShape(inputShape) {
    return inputShape;
  }
}

// Single-head causal self-attention: each time step attends only to itself and earlier steps.
export class CausalSelfAttention extends tf.layers.Layer {
  static className = 'EIatHomeCausalSelfAttention';

  constructor(config) {
    super(config);
    this.seed = config.seed;
  }

  build(inputShape) {
    const d = inputShape[inputShape.length - 1];
    for (const [i, name] of ['q', 'k', 'v', 'o'].entries()) {
      this[`w${name}`] = this.addWeight(`w${name}`, [d, d], 'float32', tf.initializers.glorotUniform({ seed: this.seed + i }));
      this[`b${name}`] = this.addWeight(`b${name}`, [d], 'float32', tf.initializers.zeros());
    }
    this.built = true;
  }

  call(inputs) {
    return tf.tidy(() => {
      const x = Array.isArray(inputs) ? inputs[0] : inputs;
      const [, steps, d] = x.shape;
      const flat = x.reshape([-1, d]);
      const project = (w, b) => flat.matMul(w.read()).add(b.read()).reshape([-1, steps, d]);
      const q = project(this.wq, this.bq);
      const k = project(this.wk, this.bk);
      const v = project(this.wv, this.bv);
      const future = tf.sub(1, tf.linalg.bandPart(tf.ones([steps, steps]), -1, 0)); // 1 above the diagonal
      const scores = tf.matMul(q, k, false, true).div(Math.sqrt(d)).add(future.mul(-1e9));
      const mixed = tf.matMul(tf.softmax(scores, -1), v).reshape([-1, d]);
      return mixed.matMul(this.wo.read()).add(this.bo.read()).reshape([-1, steps, d]);
    });
  }

  computeOutputShape(inputShape) {
    return inputShape;
  }

  getConfig() {
    return { ...super.getConfig(), seed: this.seed };
  }
}

// One pre-norm transformer block: attention then a feed-forward layer, each with a residual.
// Positions are learned for sequences of up to maxLength steps.
export function attentionStates(input, width, seed, maxLength = T) {
  const glorot = (i) => tf.initializers.glorotUniform({ seed: seed + i });
  let h = tf.layers.dense({ units: width, kernelInitializer: glorot(10) }).apply(input);
  h = new PositionEmbedding({ maxLength, seed: seed + 11 }).apply(h);
  const attended = new CausalSelfAttention({ seed: seed + 12 }).apply(new LayerNorm({}).apply(h));
  h = tf.layers.add().apply([h, attended]);
  const expanded = tf.layers.dense({ units: 2 * width, activation: 'relu', kernelInitializer: glorot(20) }).apply(new LayerNorm({}).apply(h));
  const projected = tf.layers.dense({ units: width, kernelInitializer: glorot(21) }).apply(expanded);
  return tf.layers.add().apply([h, projected]);
}

// A small causal model over radio tokens, predicting next-token logits from one-hot tokens.
// - 'gru': a bias-free dense embedding, then a GRU. The dense layer has the same parameters as
//   an embedding lookup, but its gradient is a matrix product, which every backend supports
//   (the lookup's gradient has no WebAssembly kernel).
// - 'transformer': one block of Track A's causal attention, with learned positions for the
//   TOKEN_WINDOW positions it trains on, sized to the GRU's parameter count.
export function buildTokenModel(k, seed, arch = 'gru') {
  const input = tf.input({ shape: [null, k] });
  let states;
  if (arch === 'gru') {
    const embedded = tf.layers
      .dense({ units: 16, useBias: false, kernelInitializer: tf.initializers.randomUniform({ minval: -0.05, maxval: 0.05, seed }) })
      .apply(input);
    states = tf.layers
      .gru({
        units: TOKEN_GRU_UNITS,
        returnSequences: true,
        kernelInitializer: tf.initializers.glorotUniform({ seed: seed + 1 }),
        recurrentInitializer: tf.initializers.orthogonal({ seed: seed + 2 }),
      })
      .apply(embedded);
  } else if (arch === 'transformer') {
    states = attentionStates(input, matchedTokenWidth(k, TOKEN_WINDOW), seed, TOKEN_WINDOW);
  } else {
    throw new Error(`Unknown token architecture "${arch}"`);
  }
  const logits = tf.layers.dense({ units: k, kernelInitializer: tf.initializers.glorotUniform({ seed: seed + 3 }) }).apply(states);
  return tf.model({ inputs: input, outputs: logits });
}
