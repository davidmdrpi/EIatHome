# Vendored TensorFlow.js 4.22.0

These files are copied unmodified from the official npm packages, so the Live Lab works without
any third-party CDN and always runs the version it was tested with.

| File | npm package | SHA-256 |
|---|---|---|
| `tf.min.js` | `@tensorflow/tfjs@4.22.0` (`dist/`) | `300dfae273d20b4046f46a06d735688f03675a807561e9bcb5f664eb2f3d2831` |
| `tf-backend-webgpu.min.js` | `@tensorflow/tfjs-backend-webgpu@4.22.0` (`dist/`) | `c02ae6ef5e7fff6dc9cec8b14b45e36e8894614f3ef006973e98f7568b590f3f` |
| `tf-backend-wasm.min.js` | `@tensorflow/tfjs-backend-wasm@4.22.0` (`dist/`) | `691bbec0ef8b5a7f17f212e9c5d85832030f90d030b46e99acb4dea1fb16f009` |
| `tfjs-backend-wasm-simd.wasm` | `@tensorflow/tfjs-backend-wasm@4.22.0` (`dist/`) | `77ebb28a6d34f371dbbf2086b7f2de8994acd8ea5a3cf1fa24d2c26c840cac7b` |
| `tfjs-backend-wasm-threaded-simd.wasm` | `@tensorflow/tfjs-backend-wasm@4.22.0` (`dist/`) | `c052228d4bef185c27bbe59a9e029570c78bbb9f08b3cb46b597851650373de2` |
| `tfjs-backend-wasm.wasm` | `@tensorflow/tfjs-backend-wasm@4.22.0` (`dist/`) | `70a5d516060464e5269f01c74bac1772d6b8ab6cb612acf16b5cdaf61f78d892` |

TensorFlow.js is Copyright Google LLC and licensed under the Apache License 2.0 (see
[`LICENSE`](LICENSE)).

To upgrade, download the new tarballs from `https://registry.npmjs.org/@tensorflow/<package>`,
copy the same files from each package's `dist/` folder, update the table above, and re-test the
Live Lab on every backend.
