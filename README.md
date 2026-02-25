# JAX Implementation of JiT

Unofficial JAX implementation of Just Image Transformers (JiT) from

Tianhong Li and Kaiming He. "Back to Basics: Let Denoising Generative Models Denoise." arXiv preprint [arXiv:2511.13720, 2025](https://arxiv.org/abs/2511.13720).

## Details
-   This code is for JAX on TPUs. To run the code, you should install `requirements.txt` with python 3.13.
-   To train the model, run `main.py`.
-   To sample from the model and evaluate FID-50K, run `main_fid.py`.
-   This code is based on the [official Pytorch implementation of JiT](https://github.com/LTH14/JiT).

## FID-50K Results

We verify that this JAX implementation gets similar FID-50K results as Table 4 of the JiT paper. The models were trained on TPU-v5p.

| Model | FID-50K of this repo | FID-50K from paper | Details |
|----------|----------|----------|----------|
| JiT-B/16 (baseline) | 7.285, 7.292, 7.384 | 7.48 | CFG 2.5, EMA decay 0.9996, with SwiGLU and RMSNorm and 5-ep warmup |
| JiT-B/16 (baseline w/o warmup) | 7.132 | N/A | CFG 2.5, EMA decay 0.9996, with SwiGLU and RMSNorm and no warmup |
| JiT-B/16 (+ RoPE, qk-norm, cls) | 5.341, 5.490 | 5.49 | CFG 2.1, EMA decay 0.9996 |
| JiT/L-16 (+ RoPE, qk-norm, cls) | 3.386 | 3.39 | CFG 1.8, EMA decay 0.9996 |

Note: When the FID-50K column lists multiple values, it means that the same model has been trained using several different random seeds for initializing the model weights. Some sampled images from JiT-L/16 are in the `assets` folder.