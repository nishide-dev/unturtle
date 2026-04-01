# Unturtle — Diffusion Language Model 対応 Unsloth

## プロジェクトの目的

このプロジェクトは [unslothai/unsloth](https://github.com/unslothai/unsloth) のフォーク。
**Diffusion Language Models (dLLMs)** のための Triton 最適化トレーニングフレームワークを構築する。

Autoregressive LLM (GPT系) がクロスエントロピーロスで次トークン予測を行うのに対し、
dLLM (LLaDA, MDLM 等) はトークンをランダムにマスクし、タイムステップ依存の損失で学習する。
本プロジェクトはその dLLM 学習を unsloth と同様に高速化・効率化することを目指す。

---

## ディレクトリ構成

```
unturtle/
├── unsloth/                             # upstream unsloth (変更は dLLM 追加のみ)
│   ├── kernels/
│   │   ├── cross_entropy_loss.py        # 既存: AR用Triton CEカーネル
│   │   └── masked_diffusion_loss.py     # ✅ Phase 1: dLLM用損失カーネル
│   ├── diffusion/                       # ✅ Phase 2-4: dLLM コアパッケージ
│   │   ├── __init__.py                  # 公開 API エクスポート
│   │   ├── collator.py                  # MaskedDiffusionDataCollator
│   │   ├── schedulers.py                # Linear/Cosine Alpha スケジューラ
│   │   ├── trainer.py                   # DiffusionTrainer, DiffusionTrainingArguments
│   │   └── grpo_trainer.py              # Diffu-GRPOトレーナー (Phase 6)
│   └── trainer.py                       # 既存: UnslothTrainer
├── unturtle/                            # シム: `import unturtle` を可能にする
│   ├── __init__.py                      # unsloth.* + dLLM シンボルを re-export
│   ├── diffusion/
│   │   └── __init__.py                  # unsloth.diffusion.* を re-export
│   └── models/                          # ✅ Phase 7: dLLM モデル実装
│       ├── __init__.py
│       ├── a2d/                         # AutoRegressive→Diffusion アダプタ
│       │   ├── __init__.py
│       │   ├── modeling_llama.py        # A2DLlamaLMHeadModel
│       │   ├── modeling_qwen2.py        # A2DQwen2LMHeadModel
│       │   └── modeling_qwen3.py        # A2DQwen3LMHeadModel
│       ├── llada/                       # LLaDA ネイティブ拡散モデル
│       │   ├── __init__.py
│       │   ├── configuration_llada.py
│       │   └── modeling_llada.py
│       └── dream/                       # Dream 拡散モデル
│           ├── __init__.py
│           ├── configuration_dream.py
│           ├── modeling_dream.py
│           └── generation_utils.py
├── tests/
│   ├── diffusion/                       # ✅ Phase 5: dLLM テスト (74テスト)
│   │   ├── test_masked_diffusion_loss.py
│   │   ├── test_integration.py
│   │   ├── test_gpu_accuracy.py
│   │   └── test_grpo_trainer.py
│   ├── models/                          # ✅ Phase 7: モデルテスト (31テスト)
│   │   ├── test_a2d.py
│   │   ├── test_llada.py
│   │   └── test_dream.py
│   └── test_fast_diffusion_model.py     # ✅ Phase C: FastDiffusionModel テスト (10テスト)
├── dev/                                 # 設計ドキュメント (必ず参照)
│   ├── repos/                           # 参照リポジトリ (gitignore済み、要 git clone)
│   │   ├── d1/                          # dllm-reasoning/d1
│   │   └── dllm/                        # zhziszz/dllm
│   ├── 01_overview.md                   # プロジェクト概要・フェーズ一覧
│   ├── 02_dllm_algorithms.md            # LLaDA/MDLM/BD3LM/d1 アルゴリズム詳細
│   ├── 03_loss_analysis.md              # 損失関数の解析・Triton拡張方針
│   ├── 04_triton_plan.md                # masked_diffusion_loss.py の設計
│   ├── 05_trainer_plan.md               # DiffusionTrainer の設計・使用例
│   └── 06_references.md                # 参考論文・リポジトリ・重要ファイル一覧
└── CLAUDE.md                            # このファイル
```

---

## パッケージ名移行方針

### 現状: Phase B — unturtle/ が正規実装

`unturtle/diffusion/` と `unturtle/kernels/` が正規の実装場所。
`unsloth/diffusion/` と `unsloth/kernels/masked_diffusion_loss.py` は後方互換 shim に格下げ済み。

```python
# どちらでも動作する
import unsloth             # upstream 互換
import unturtle            # unturtle 公開 API
from unturtle.diffusion import DiffusionTrainer   # 正規パス
from unsloth.diffusion import DiffusionTrainer    # shim (後方互換)
from unturtle import FastDiffusionModel           # Phase C 追加
```

**Phase B 移行の設計**:
- `unsloth/` のファイルを削除せず shim に格下げ → `git merge upstream/main` 引き続き可能
- `unturtle/diffusion/` / `unturtle/kernels/` が canonical source

### Phase C — FastDiffusionModel 完了

`FastDiffusionModel` を `unturtle/fast_diffusion_model.py` に実装。A2D モデルに対して:
- unsloth の Triton-fused LoRA カーネル (QKV/O) を適用
- `A2DAttention_fast_forward` (bidirectional, causal=False) を injection
- `TaskType.FEATURE_EXTRACTION` で PEFT ラップ (CAUSAL_LM guard 回避)

---

## 実装ロードマップ

| フェーズ | 対象ファイル | 内容 | 状態 |
|---------|------------|------|------|
| Phase 0 | `dev/`, `CLAUDE.md` | ドキュメント整備 | ✅ 完了 |
| Phase 1 | `unsloth/kernels/masked_diffusion_loss.py` | Tritonカーネル (CE再利用 + タイムステップ重み) | ✅ 完了 |
| Phase 2 | `unsloth/diffusion/collator.py` | MaskedDiffusionDataCollator | ✅ 完了 |
| Phase 3 | `unsloth/diffusion/schedulers.py` | Alpha スケジューラ (Linear/Cosine) | ✅ 完了 |
| Phase 4 | `unsloth/diffusion/trainer.py` | DiffusionTrainer, DiffusionTrainingArguments | ✅ 完了 |
| Phase 5 | `tests/diffusion/test_masked_diffusion_loss.py` | テスト・数値検証 (22テスト) | ✅ 完了 |
| Phase A | `unturtle/`, `pyproject.toml` | パッケージ名シム (`unturtle` as public API) | ✅ 完了 |
| Phase 6 | `unsloth/diffusion/grpo_trainer.py` | Diffu-GRPO RL ステージ | ✅ 完了 |
| Phase 7 | `unturtle/models/` + `tests/models/` | dLLM モデル実装 (A2D/LLaDA/Dream) + import rename | ✅ 完了 |
| Phase B | `unturtle/diffusion/`, `unturtle/kernels/` | dLLM コード canonical 移行、shim 格下げ | ✅ 完了 |
| Phase C | `unturtle/fast_diffusion_model.py`, `unturtle/models/a2d/_fast_forward.py` | FastDiffusionModel + 双方向 LoRA 適用 | ✅ 完了 |

---

## 重要な技術的ポイント

### dLLM 損失の核心

```python
# AR-LLM: 全非パディング位置で CE
loss = CE(logits, labels)  # labels=-100 でパディング無視

# dLLM: マスク位置のみ CE + タイムステップ重み
t = uniform(eps, 1)
masked_labels = labels.clone()
masked_labels[~diffusion_mask] = -100
loss = CE(logits, masked_labels) / t  # d1 SFT スタイル
# or
loss = CE(logits, masked_labels)      # MDLM/LLaDA スタイル
```

### 既存カーネルとの関係

`unsloth/kernels/cross_entropy_loss.py` の `Fast_CrossEntropyLoss` を**直接再利用**できる:
- `label == -100` → 損失ゼロ、の仕組みはそのまま流用
- dLLM 用ラベル: マスク位置 = `x_0[i]`、アンマスク位置 = `-100`
- タイムステップ重みは Python レベルで乗算 (Phase 1)
- 将来的に専用カーネルへ最適化 (Phase 1 → Phase 1b)

### 双方向 Attention

dLLM は causal mask なし (`is_causal=False`)。
LLaDA 系モデル (GSAI-ML/LLaDA-8B など) はアーキテクチャ的に双方向。
Unsloth の AR モデル最適化 (Flash Attention, causal mask 前提) との互換性を確認すること。

### Completion-only マスキング

プロンプト部分はマスクせず、completion (回答) 部分のみマスクする:
```python
maskable = completion_mask  # プロンプト = False, completion = True
diffusion_mask = (rand < p_mask) & maskable
```

---

## 参照リポジトリ (dev/repos/)

`dev/repos/` は `.gitignore` 済み。初回セットアップ時に以下を実行:

```bash
mkdir -p dev/repos
git clone https://github.com/dllm-reasoning/d1.git dev/repos/d1
git clone https://github.com/zhziszz/dllm.git dev/repos/dllm
```

| ディレクトリ | リポジトリ | 参照すべきファイル |
|-------------|-----------|-----------------|
| `dev/repos/d1/` | [dllm-reasoning/d1](https://github.com/dllm-reasoning/d1) | `SFT/sft_trainer.py`, `diffu-grpo/diffu_grpo_trainer.py` |
| `dev/repos/dllm/` | [zhziszz/dllm](https://github.com/zhziszz/dllm) | `dllm/core/trainers/mdlm.py`, `dllm/core/schedulers/alpha.py` |

詳細は `dev/06_references.md` を参照。

---

## 開発環境セットアップ

### 前提
- CUDA 12.4、NVIDIA RTX 6000 Ada (確認済み)
- uv インストール済み: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 初回セットアップ

> **注意**: `install.sh --local` は Python 3.13 を使うが xformers が 3.13 非対応のためビルド失敗する。
> Python 3.12 で手動セットアップを推奨。

```bash
# 1. Python 3.12 仮想環境を作成
uv venv .venv --python 3.12
source .venv/bin/activate

# 2. PyTorch を先にインストール (CUDA 12.4 用)
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# 3. unsloth をローカルソースから editable install
#    (setuptools==80.9.0 が必要 — pyproject.toml の build-system 要件に合わせる)
uv pip install "setuptools==80.9.0" "setuptools-scm==9.2.0"
uv pip install -e ".[huggingface]"

# 4. 開発ツール
uv pip install pytest ruff bitsandbytes
```

### 毎回の起動

```bash
source .venv/bin/activate
```

### テスト実行

```bash
# dLLM 実装のテスト (CPU + CUDA)
python -m pytest tests/diffusion/ -v

# 全テスト
python -m pytest tests/ -v
```

### 既知のパッケージ互換性問題

| 問題 | 原因 | 対処 |
|------|------|------|
| `from trl import GRPOTrainer` が失敗 | TRL が `llm_blender` を import しようとするが、`llm_blender` が古い transformers API (`TRANSFORMERS_CACHE`) に依存 | `llm_blender` と `mergekit` をインストールしない。TRL 0.29+ では修正済み |
| TRL 0.14〜0.19 で `from vllm import ...` エラー | `is_vllm_available()` が `(False, None)` という tuple を返し、if 文で truthy と評価されるバグ | TRL 0.29+ を使用 |
| `install.sh --local` 失敗 | Python 3.13 に xformers が非対応 | Python 3.12 で手動セットアップ（上記参照） |
| `tests/saving/` の INTERNALERROR | `soundfile` 等の未インストールパッケージで `sys.exit(1)` | upstream の既存問題。`pytest tests/diffusion/` で dLLM テストのみ実行 |

> **推奨 TRL バージョン**: `trl==0.29.1` 以降
> `llm_blender` および `mergekit` は **インストールしないこと**（依存競合が発生する）。

### Lint / Format

```bash
ruff check .      # lint
ruff format .     # format
```

### 動作確認

```bash
source .venv/bin/activate
python -c "
import torch; print('torch:', torch.__version__, '/ CUDA:', torch.cuda.is_available())
import unturtle; print('unturtle:', unturtle.__version__)
from unturtle.diffusion import DiffusionTrainer, LinearAlphaScheduler
print('DiffusionTrainer: OK')
"
```

---

## Git / Issue / PR ワークフロー

### Issue 作成ルール

実装開始前に必ず Issue を作成する。

**タイトル形式**: `[Phase N] <動詞> <対象>`
- 例: `[Phase 1] Implement fast_masked_diffusion_loss Triton kernel`

**本文に必須**:
- 背景・目的
- Acceptance Criteria (受け入れ条件)
- 関連する `dev/` ドキュメントへのリンク

**ラベル体系**:

| ラベル | 用途 |
|-------|------|
| `type: feat` | 新機能 |
| `type: fix` | バグ修正 |
| `type: docs` | ドキュメント |
| `type: test` | テスト |
| `type: perf` | 性能改善 (Tritonカーネル最適化等) |
| `type: refactor` | リファクタリング |
| `type: chore` | 依存更新など |
| `phase: 1` 〜 `phase: 6` | 実装フェーズ |
| `diffusion` | dLLM固有 |
| `triton` | Tritonカーネル関連 |

---

### ブランチ命名規則

```
<type>/<issue番号>-<短い説明>
```

例:
- `feat/42-masked-diffusion-loss`
- `fix/55-collator-masking-bug`
- `docs/12-update-claude-md`
- `test/67-triton-kernel-correctness`

**ルール**:
- `main` への直接 push 禁止
- upstream の取り込みは `git fetch upstream && git merge upstream/main`

---

### コミットメッセージ規則

```
<emoji> <type>(<scope>): <description> (#<issue番号>)

[オプション: なぜこの変更をしたか]
```

**Emoji + type 対応表**:

| emoji | type | 使いどころ |
|-------|------|----------|
| ✨ | `feat` | 新機能追加 |
| 🐛 | `fix` | バグ修正 |
| 📚 | `docs` | ドキュメント更新 |
| ✅ | `test` | テスト追加・修正 |
| ⚡ | `perf` | 性能改善 (Tritonカーネル最適化など) |
| ♻️ | `refactor` | リファクタリング |
| 🔧 | `chore` | 設定・依存更新 |
| 🚧 | `wip` | 作業中 (Draft PR 時のみ) |

scope は任意 (例: `kernel`, `trainer`, `collator`, `docs`)

**コミット例**:
```
✨ feat(kernel): add fast_masked_diffusion_loss Triton kernel (#42)
🐛 fix(collator): fix off-by-one in Bernoulli mask rate (#55)
⚡ perf(kernel): chunk masked diffusion loss for vocab>65K (#61)
📚 docs: update CLAUDE.md with Phase 1 completion (#12)
```

upstream unsloth のスタイル (`fix: description (#N)`) とも互換。emoji は unturtle 独自追加。

---

### PR ルール

**PR タイトル**: コミットと同形式
```
✨ feat(kernel): add fast_masked_diffusion_loss (#42)
```

**PR ワークフロー**:
1. 作業中は **Draft PR** で open (`🚧 wip` コミットを積む)
2. 完成したら **Ready for Review** に変更してレビュアーをアサイン
3. 1 PR = 1 Issue を原則とする
4. マージ前に `main` との差分を解消 (rebase or merge)
5. **Squash and merge** を標準とする (コミットタイトル = PR タイトル)

**Triton カーネル変更時の追加要件**:
- 数値テスト結果 (`pytest tests/` の出力) を PR コメントに貼ること

**コードレビュー時の必須確認事項**:
- `dev/repos/` の参照実装と照合すること。アルゴリズムの挙動について疑問がある場合は、
  推測や論文の解釈だけで判断せず、必ず以下の実装を直接確認する:
  - `dev/repos/d1/diffu-grpo/diffu_grpo_trainer.py` — Diffu-GRPO の参照実装
  - `dev/repos/d1/SFT/sft_trainer.py` — d1 SFT の参照実装
  - `dev/repos/dllm/dllm/core/trainers/mdlm.py` — MDLM/LLaDA の参照実装
- 参照実装と異なる挙動を「バグ」と判定する前に、意図的な設計変更か否かを確認する

**ドキュメント更新**:
- `dev/` や `CLAUDE.md` の更新が必要な変更は同一 PR に含める

---

### マージルール

| ケース | マージ方法 |
|-------|-----------|
| 通常の機能追加・バグ修正 | **Squash and merge** |
| upstream unsloth からの取り込み | **Merge commit** (履歴を残す) |

`main` は常にビルド可能な状態を維持する。

---

## 検証結果

### テスト状況

| テスト種別 | ファイル | 件数 | 状態 |
|-----------|---------|------|------|
| ユニットテスト (損失/スケジューラ/コレーター) | `tests/diffusion/test_masked_diffusion_loss.py` | 22 | ✅ 全通過 |
| インテグレーション (E2E forward/backward) | `tests/diffusion/test_integration.py` | 15 | ✅ 全通過 |
| GPU 数値精度 (Triton vs F.cross_entropy) | `tests/diffusion/test_gpu_accuracy.py` | 25 | ✅ 全通過 |
| DiffuGRPO (コンフィグ/静的メソッド/forward process) | `tests/diffusion/test_grpo_trainer.py` | 13 | ✅ 全通過 |
| A2D モデル (LLaMA/Qwen2/Qwen3 config・forward・AutoConfig登録) | `tests/models/test_a2d.py` | 15 | ✅ 全通過 |
| LLaDA モデル (config・forward・backward) | `tests/models/test_llada.py` | 7 | ✅ 全通過 |
| Dream モデル (config・forward・backward・generation utils) | `tests/models/test_dream.py` | 9 | ✅ 全通過 |
| FastDiffusionModel (stubs/LoRA patch/bidirectional/save-load) | `tests/test_fast_diffusion_model.py` | 23 | ✅ 全通過 |
| E2E パイプライン (CPU fast + GPU slow) | `tests/test_e2e_integration.py` | 4 | ✅ 全通過 (slow/GPU 2件は実 HF チェックポイントが必要) |

インテグレーションテストの確認内容:
- BERT (双方向 attention) / GPT-2 (causal) 両モデルで forward/backward 通過
- CPU・CUDA 両パス通過
- `loss_weight_type = uniform / timestep / scheduler` 全モード有効
- オプティマイザステップ後に損失が減少することを確認
- 双方向 attention が将来トークンに依存することを確認 (dLLM の前提条件)

### ベンチマーク結果

GPU: **NVIDIA RTX 6000 Ada Generation** / mask_ratio=0.5 / warmup=10 iters=100

比較対象:
- **Triton (unturtle)**: `fast_masked_diffusion_loss` — Triton CE カーネル
- **d1-style (reference)**: `F.cross_entropy` → `/t` — d1/LLaDA の実際の実装
- **PyTorch masked**: `F.cross_entropy` + label=-100 マスク — ナイーブな Python 実装

ベンチマークスクリプト: `dev/benchmark_loss.py`

| Batch | SeqLen | Vocab | Impl | Time (ms) | Mem (MB) | vs d1-ref | vs PyTorch |
|------:|-------:|------:|------|----------:|---------:|----------:|-----------:|
| 4 | 128 | 32,000 | **Triton (unturtle)** | 0.08 | 0.0 | **2.09x** | **2.07x** |
| | | | d1-style (reference) | 0.17 | 62.5 | — | — |
| | | | PyTorch masked | 0.17 | 62.5 | — | — |
| 4 | 512 | 32,000 | **Triton (unturtle)** | 0.30 | 0.0 | **2.08x** | **2.09x** |
| | | | d1-style (reference) | 0.63 | 250.0 | — | — |
| | | | PyTorch masked | 0.63 | 250.0 | — | — |
| 2 | 512 | 128,256 | **Triton (unturtle)** | 0.63 | 0.0 | **2.95x** | **2.94x** |
| | | | d1-style (reference) | 1.86 | 504.0 | — | — |
| | | | PyTorch masked | 1.86 | 504.0 | — | — |
| 4 | 512 | 126,464 | **Triton (unturtle)** | 1.20 | 0.1 | **3.05x** | **3.05x** |
| | | | d1-style (reference) | 3.65 | 992.0 | — | — |
| | | | PyTorch masked | 3.66 | 992.0 | — | — |

**要点**:
- Triton カーネルは d1/LLaDA の参照実装より **2.1〜3.1x 高速**
- メモリ使用量はほぼゼロ (参照実装: vocab=128K 時に最大 992 MB の一時 tensor を確保)
- シーケンス長・語彙サイズが大きいほど効果が大きい (seq=512, vocab=126K で 3x)
- d1-style と PyTorch masked はほぼ同等 → Triton カーネルの実装が効いている

---

## 開発ガイドライン

- **既存機能を壊さない**: AR-LLM 用の機能は一切変更しない。dLLM 機能は新規追加のみ。
- **Triton カーネルのテスト**: 数値正確性を `F.cross_entropy` との比較で必ず検証すること。
- **互換性**: LoRA/QLoRA/4-bit 量子化は dLLM でも動作するよう維持する。
- **ドキュメント更新**: フェーズ完了時に `dev/` の該当ファイルとこの `CLAUDE.md` を更新する。

---

## 開発中に発見したトリッキーな注意点

### 1. Triton カーネルは `model.device == "cuda"` のときのみ適用する

`_patch_a2d_peft` / `_patch_dream_peft` / `_patch_llada_peft` は、モデルパラメータが
CUDA 上にある場合のみ Triton カーネルをパッチする。
`torch.cuda.is_available()` ではなく `first_param.device.type == "cuda"` で判定すること。

```python
first_param = next(iter(model.parameters()), None)
on_cuda = first_param is not None and first_param.device.type == "cuda"
if not on_cuda:
    return n_qkv, n_o, n_mlp  # Triton カーネルをスキップ
```

**理由**: `unsloth_zoo` は `UnslothSFTTrainer` 経由で学習時に activations を bf16 に変換する。
モデルが float32 (CPU) のまま学習が始まると、bf16 activation × float32 LoRA weight の dtype
ミスマッチが `matmul_lora` で発生する (`RuntimeError: BFloat16 != Float`)。

### 2. `DiffusionTrainer` に `processing_class` を明示的に渡す

`DiffusionTrainer` (= UnslothSFTTrainer) は `processing_class` が渡されないと
`model.config.name_or_path` を使って HuggingFace Hub から `processor_config.json` を
取得しようとする。テスト用のランダムモデルは `name_or_path = ""` でこの呼び出しが
`OSError: Repo id must be alphanumeric` で失敗する。

```python
# NG: テスト用ランダムモデルでは name_or_path="" のため Hub 呼び出しが失敗
trainer = DiffusionTrainer(model=peft_model, args=args, train_dataset=dataset)

# OK: tokenizer を明示的に渡す
trainer = DiffusionTrainer(model=peft_model, args=args, train_dataset=dataset,
                            processing_class=tokenizer)
```

### 3. save/reload テストでは base model の weights を事前にスナップショットする

`A2DLlamaLMHeadModel(cfg)` はランダム初期化なので、LoRA adapter を保存して別の
インスタンスに再ロードしても base weights が異なり logits が一致しない。

```python
# PEFT ラップ前に base weights を保存しておく
base_state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}
peft_model = FastDiffusionModel.get_peft_model(base_model, ...)

# reload 時に同じ base weights を復元してから adapter を適用
fresh_base = A2DLlamaLMHeadModel(cfg)
fresh_base.load_state_dict(base_state_dict)
reloaded = PeftModel.from_pretrained(fresh_base, adapter_dir)
```

### 4. LoRA の `lora_dropout` と `bias` が Triton カーネルを無効化する

`lora_dropout != 0` または `bias != "none"` の場合、Triton LoRA カーネルは適用されない
(PEFT デフォルトのフルパスにフォールバック)。このとき `_warn_once()` でログが出る。

Dream モデルの `q/k/v_proj` は `bias=True` のため標準 `apply_lora_qkv` は使えず、
専用の `apply_lora_qkv_with_bias` (`unturtle/kernels/fast_lora.py`) が必要。

```
[WARNING] FastDiffusionModel: cannot patch QKV with Triton kernel
          (LoRA adapters not enabled or bias present — e.g. Dream q/k/v_proj).
```

### 5. GPU テストは `@pytest.mark.skipif(not cuda)` + `model.cuda()` で明示する

Triton カーネルの適用を検証するテストはモデルを明示的に CUDA に移してから実行する:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels require CUDA")
def test_apply_qkv_patched_to_lora(self, tiny_model):
    peft_model = FastDiffusionModel.get_peft_model(
        tiny_model.cuda(),  # ← CUDA に移してからパッチ
        r=4, target_modules=[...], lora_dropout=0,
    )
    for layer in peft_model.base_model.model.model.layers:
        assert layer.self_attn.apply_qkv is apply_lora_qkv
```

### 6. `LoRA_QKV_Bias` の backward では `dBias = dOutput.sum(0)` を使う

bias を含む LoRA カーネルのカスタム autograd 関数では、bias の勾配は
`saved_tensors` から `dQ/dK/dV` を参照して `dQ.sum(0)` で求める:

```python
# unturtle/kernels/fast_lora.py: LoRA_QKV_Bias.backward
dQ, dK, dV = ... # (通常の LoRA_QKV.backward と同じ)
d_QBias = dQ.sum(0) if ctx.needs_input_grad[6] else None
d_KBias = dK.sum(0) if ctx.needs_input_grad[12] else None
d_VBias = dV.sum(0) if ctx.needs_input_grad[18] else None
```

### 7. `LLaDA` の PEFT パス解決に注意

`LLaDAModelLM` は `LLaDAModel` を `self.model` に持つため、PEFT ラップ後のパスが
1段深くなる:

```
PeftModel
 └─ base_model (LoraModel)
     └─ model (LLaDAModelLM)      ← LlaMA は ここが LlamaForCausalLM
         └─ model (LLaDAModel)     ← LlaMA にはこの中間層がない
             └─ transformer
                 └─ blocks
```

`_patch_llada_peft` では両方のパスを試みるフォールバック付き解決が必要:

```python
inner = model.base_model.model
if hasattr(inner, "model") and hasattr(inner.model, "transformer"):
    transformer = inner.model.transformer
elif hasattr(inner, "transformer"):
    transformer = inner.transformer
```
