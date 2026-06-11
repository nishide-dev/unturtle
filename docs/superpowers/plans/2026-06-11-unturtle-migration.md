# Unturtle 移植実装計画（legacy → クリーン再構築）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `unturtle-legacy` から新リポジトリ `nishide-dev/unturtle` へ、Studio・lighteval・互換コードを捨てた単一パッケージとして全モジュールを移植し、unsloth クラス継承への内部リファクタを行う。

**Architecture:** 1フェーズ=1 issue=1 PR のモジュール順次移植。各 PR は legacy のテストを先に持ち込み「import パス修正のみでグリーン」を契約とし、その上で内部リファクタ（`UnslothTrainer` 継承化、`FastModel` 委譲）を別コミットで行う。依存順: kernels/utils → generation → backbones → conversion → trainer+diffusion+eval(smoke) → fast_diffusion_model → eval(harness) → cli。

**Tech Stack:** Python 3.12 / uv / pytest / ruff / unsloth(最新pin) / trl / transformers / Triton

**スペック:** `docs/superpowers/specs/2026-06-11-unturtle-migration-design.md`

---

## 共通事項（全タスクで前提）

```bash
LEGACY=/grouper/nishide.21066-1000003/projects/unturtle        # legacy クローン
NEW=/grouper/nishide.21066-1000003/projects/unturtle-new       # 新リポジトリ クローン
```

- **コピーは常に** `rsync -a --exclude='__pycache__' $LEGACY/<src> $NEW/<dst>` を使う。
- **テスト実行は GPU マシン必須**（unsloth は import 時に GPU を要求。CI はこのため lint-only）。
- fast テスト実行コマンド（新リポジトリ側）:
  `cd $NEW && uv run python -m pytest <対象> -m "not slow" -v`
- 各 PR のワークフロー（全タスク共通、以下「**issue→PR 手順**」と呼ぶ）:
  1. `gh issue create --repo nishide-dev/unturtle --title "[Migration N] <内容>" --body "<受け入れ基準>" --label "type: <type>"`（ラベル未作成なら `gh label create` で先に作る）
  2. `git -C $NEW switch -c <type>/<issue番号>-<short-desc>`
  3. コミットは `<emoji> <type>(<scope>): <description> (#<issue番号>)` 形式
  4. `git push -u origin <branch>` → `gh pr create --repo nishide-dev/unturtle --fill --base main`
  5. fast テスト＋`uv run ruff check . && uv run ruff format --check .` グリーン確認後、**Squash and merge** → `git switch main && git pull`
- **テスト import パス修正の原則:** 書き換えてよいのは import 文（`unturtle_cli` → `unturtle.cli`）のみ。assert やロジックは変更禁止。変更が必要に見えたら挙動差分なのでバグとして調査する。

### 移植しないファイル（確定。コピー対象から常に除外）

- `unturtle_studio/` 全部、`unturtle/eval/experimental/` 全部
- `unturtle/models/{llada,dream,modernbert}.py`（非推奨エイリアス）、`unturtle/models/a2d/`
- `tests/models/test_import_compat.py`、`tests/eval/test_lighteval_*.py`（3件）、`tests/install/`（3件）、`tests/examples/test_validate_studio_mdlm_chat.py`、`examples/validate_studio_mdlm_chat.py`
- legacy の `install.sh`（Python 3.13 前提の upstream ラッパー）、`.github/workflows/{deploy-docs,release,stale}.yml`（必要になったら個別に再導入）
  - ※ 新リポジトリには **uv 前提の薄い `install.sh` を新規作成**して置く（Task 1 で追加済み）。
    実行順序が重要: torch（CUDA ビルドを TORCH_INDEX で選択、既定 cu128）→ build/test ツール →
    `uv pip install -e ".[huggingface]"`。素の pip は unsloth の依存グラフで壊れた `regex` を
    解決する事故が確認されたため非サポートと明記。

---

### Task 0: ブートストラップ（リポジトリ・環境・資産）

**Files:**
- なし（環境準備のみ）

- [ ] **Step 0-1: origin/main の存在確認**

```bash
git -C $NEW ls-remote origin main
```

Expected: コミットハッシュが返る。**空なら停止**してユーザーに `! git -C $NEW push -u origin main` の実行を依頼（初回 push は main 直 push 制限のため人間が行う）。

- [ ] **Step 0-2: 新リポジトリ用 venv 構築（GPU マシン上）**

```bash
cd $NEW
uv venv .venv --python 3.12
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install "setuptools==80.9.0" "setuptools-scm==9.2.0"
uv pip install pytest ruff bitsandbytes
```

（`-e ".[huggingface]"` は Task 1 で pyproject 作成後にインストール）

- [ ] **Step 0-3: git 管理外資産のコピー**

```bash
rsync -a $LEGACY/dev/local/ $NEW/dev/local/
rsync -a $LEGACY/dev/papers/ $NEW/dev/papers/
rsync -a $LEGACY/.references/ $NEW/.references/
ls $NEW/dev/local $NEW/.references | head
```

Expected: ファイルが見える。`dev/repos/` は必要になった時に CLAUDE.md の手順で再クローン。

---

### Task 1: 骨格（pyproject / CI / NOTICE / README）— PR #1

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.github/workflows/ci.yml`, `NOTICE`, `README.md`, `unturtle/__init__.py`, `unturtle/_version.py`

- [ ] **Step 1-1: issue→PR 手順で issue 作成・ブランチ作成**（`chore/N-skeleton`）

- [ ] **Step 1-2: unsloth 最新版の確認と互換性プローブ（legacy 側）**

```bash
uv pip index versions unsloth 2>/dev/null || .venv/bin/python -m pip index versions unsloth
# legacy の venv で unsloth/unsloth_zoo だけ最新化して fast テストを回す
cd $LEGACY
uv pip install -U unsloth unsloth_zoo
uv run python -m pytest tests/diffusion/ tests/models/ tests/test_fast_diffusion_model.py tests/test_e2e_integration.py -m "not slow" -x -q 2>&1 | tail -30
```

Expected: 最新バージョン番号が判明する。テストは**失敗してもよい** — 失敗一覧を `$NEW/dev/local/2026-06-11-unsloth-pin-probe.md` に記録し、Task 6/7 のリファクタ対象として参照する。終了後 legacy venv を元に戻す: `uv pip install "unsloth>=2026.3.17,<2026.4" "unsloth_zoo>=2026.3.6,<2026.4"`（厳密な復元は `uv pip install unsloth==2026.4.8 unsloth_zoo==2026.4.9`）。

- [ ] **Step 1-3: pyproject.toml を作成**（legacy から studio/experimental を除去、単一パッケージ化、pin を Step 1-2 の最新に）

```toml
[build-system]
requires = ["setuptools==80.9.0", "setuptools-scm==9.2.0"]
build-backend = "setuptools.build_meta"

[project]
name = "unturtle"
dynamic = ["version"]
description = "dLLM method layer on top of unsloth — conversion, objectives, inference acceleration, and canonical evaluation for diffusion language models"
readme = "README.md"
requires-python = ">=3.9,<3.15"
license = "Apache-2.0"
keywords = ["ai", "llm", "diffusion", "dllm", "machine learning", "pytorch", "triton"]
authors = [{name = "nishide-dev"}]
classifiers = [
    "Programming Language :: Python",
    "Environment :: GPU",
    "Environment :: GPU :: NVIDIA CUDA",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "unsloth>=<Step1-2で確認した最新>",
    "typer>=0.12.0",
    "pydantic>=2.0",
    "pyyaml>=6.0",
]

[tool.setuptools.dynamic]
version = {attr = "unturtle._version.__version__"}

[tool.setuptools]
include-package-data = true

[project.scripts]
unturtle = "unturtle.cli:app"

[tool.setuptools.packages.find]
include = ["unturtle*"]

[project.optional-dependencies]
huggingface = [
    "transformers>=4.51.3",
    "datasets>=3.4.1",
    "accelerate>=0.34.1",
    "peft>=0.18.0",
    "unsloth_zoo>=<Step1-2で確認した最新>",
    "huggingface_hub>=0.34.0",
    "hf_transfer",
    "trl",
]
eval = ["lm-eval>=0.4.5"]
dev = ["ruff>=0.9.0", "ty>=0.0.0a1", "pytest>=8.0"]
```

`[tool.pytest.ini_options]`・`[tool.ruff]`〜`[tool.ruff.format]`・`[tool.ty.environment]` は legacy の `pyproject.toml` 95〜160 行目を**そのままコピー**する（markers 3種、ruff select/ignore リスト、per-file-ignores、format 設定）。ただし ruff の `extend-exclude` から studio 由来の `*mapper.py` 等が不要になっていないか確認し、unturtle に存在しないパターンは削る。

- [ ] **Step 1-4: 残りの骨格ファイルを作成**

```bash
# .gitignore は legacy をベースに studio 関連行を削除してコピー
rsync -a $LEGACY/.gitignore $NEW/.gitignore
# studio 関連行（"studio/backend/..." 等）を削除する
grep -n "studio" $NEW/.gitignore   # → 該当行を削除
```

`.gitignore` には以下が**すべて ignore 対象として含まれていること**を確認し、欠けていれば追加する（特に `.references/` は legacy では追跡対象だったが、新リポジトリでは「未整理のローカル調査メモ」としてローカル資産化する決定）:

```gitignore
.venv
dev/repos/
dev/papers/
dev/local/
.references/
```

確認コマンド: `grep -n "references\|dev/local\|dev/papers\|dev/repos\|^\.venv" $NEW/.gitignore`。
`.references/` が無ければ追記する。これにより Task 0 でコピーした調査資産（`dev/local`・`dev/papers`・`.references`）は untracked にならず `git status` がクリーンになる。
（`.references/` の有用部分を `docs/` へ整理して昇格させるのは移植完了後の別作業。本移植では追跡しない。）

`unturtle/_version.py` は legacy をそのままコピー（`__version__ = "0.1.0"`）。
`unturtle/__init__.py` は**最小**で作成（本格的な re-export は Task 7 で legacy 版を移植）:

```python
"""Unturtle — dLLM method layer on top of unsloth."""

from unturtle._version import __version__

__all__ = ["__version__"]
```

`NOTICE`（vendored 出典の記録。fork-point は確定済み: `git -C $LEGACY merge-base HEAD upstream/main` =
`a6c1f893fc87c0973f9c32e59ca3d7d54ffb9724`）:

```text
Unturtle
Copyright 2025-present nishide-dev

This product includes software developed by the Unsloth team
(https://github.com/unslothai/unsloth), licensed under the Apache License 2.0.

The following files are derived from unslothai/unsloth (fork point:
commit a6c1f893fc87c0973f9c32e59ca3d7d54ffb9724, 2026-03-28):
- unturtle/trainer.py
- unturtle/utils/attention_dispatch.py
- unturtle/utils/packing.py
- unturtle/kernels/__init__.py
```

`.github/workflows/ci.yml`（**lint-only**。unsloth が import 時に GPU を要求するため CPU ランナーでテスト不可 — legacy と同じ制約）:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    name: Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install ruff
        run: uv pip install --system "ruff>=0.9.0"
      - name: ruff check
        run: ruff check unturtle/ benchmarks/ tests/
      - name: ruff format --check
        run: ruff format --check unturtle/ benchmarks/ tests/

# NOTE: テストジョブは意図的に無し。unsloth が import 時に GPU を要求するため
# CPU ランナーではテストを collect できない。fast テストはローカル GPU で実行:
#   uv run python -m pytest tests/ -m "not slow" -v
```

`README.md`（位置付け文。最低限以下を含む）:

```markdown
# Unturtle

**The dLLM method layer on top of [unsloth](https://github.com/unslothai/unsloth).**

- transformers = モデル実装＋モデル内 loss
- TRL = objective 別 trainer
- unsloth = 高速ロード・QLoRA・カーネルパッチ
- **unturtle = dLLM 手法レイヤー**: AR→Diffusion 変換（Tiny-A2D）、diffusion objective
  （MDLM / BD3LM / Diffu-GRPO）、モデル横断の推論高速化（block decode / cache）、
  lm-eval-harness による canonical 評価

旧リポジトリ（unsloth フォーク時代）: [unturtle-legacy](https://github.com/nishide-dev/unturtle-legacy)
```

- [ ] **Step 1-5: インストールと lint 確認**

```bash
cd $NEW
uv pip install -e ".[huggingface]"
uv run python -c "import unturtle; print(unturtle.__version__)"
uv run ruff check . && uv run ruff format --check .
```

Expected: `0.1.0` が表示され、lint がグリーン。

- [ ] **Step 1-6: コミット・PR・マージ**（issue→PR 手順 3〜5）

```bash
git add -A && git commit -m "🔧 chore: repo skeleton — pyproject / CI / NOTICE / README (#N)"
```

---

### Task 2: kernels/ + utils/ — PR #2

**Files:**
- Create: `unturtle/kernels/`（legacy 全 7 ファイル相当）, `unturtle/utils/`（`__init__.py`, `attention_dispatch.py`, `packing.py`）
- Test: `tests/__init__.py`, `tests/utils/`（`__init__.py`, `os_utils.py`, `test_attention_masks.py`, `test_packing.py`）

- [ ] **Step 2-1: issue→PR 手順**（`feat/N-port-kernels-utils`）

- [ ] **Step 2-2: テストを先にコピーして失敗確認**

```bash
rsync -a --exclude='__pycache__' $LEGACY/tests/__init__.py $NEW/tests/__init__.py
rsync -a --exclude='__pycache__' $LEGACY/tests/utils/ $NEW/tests/utils/
cd $NEW && uv run python -m pytest tests/utils/ -m "not slow" -v 2>&1 | tail -5
```

Expected: FAIL（`ModuleNotFoundError: unturtle.utils` / `unturtle.kernels`）

- [ ] **Step 2-3: 本体をコピーしてグリーン確認**

```bash
rsync -a --exclude='__pycache__' $LEGACY/unturtle/kernels/ $NEW/unturtle/kernels/
rsync -a --exclude='__pycache__' $LEGACY/unturtle/utils/ $NEW/unturtle/utils/
cd $NEW && uv run python -m pytest tests/utils/ -m "not slow" -v
```

Expected: PASS（import パス修正不要のはず — パスは legacy と同一）

- [ ] **Step 2-4: unsloth_zoo 同等物チェック（置換判断）**

```bash
ZOO=$NEW/.venv/lib/python3.12/site-packages/unsloth_zoo
grep -rln "packed_seq\|cu_seqlens\|varlen" $ZOO --include="*.py" | head
grep -rln "attention_dispatch\|sdpa_packed" $ZOO --include="*.py" | head
```

判断ルール: **同名・同セマンティクスの関数が unsloth_zoo にあれば** vendored 側を削って import に置換し、テスト再実行。**なければ何もしない**（NOTICE 維持）。置換した場合は NOTICE から該当ファイル行を削除。
注意: `build_sdpa_packed_attention_mask()` は causal であり dLLM packed attention に流用不可（CLAUDE.md 既知の gotcha）— 置換候補に同名があってもセマンティクス確認を省略しない。

- [ ] **Step 2-5: lint・コミット・PR・マージ**

```bash
cd $NEW && uv run ruff check . && uv run ruff format --check .
git add -A && git commit -m "✨ feat(kernels,utils): port Triton kernels and packed/attention utils (#N)"
```

---

### Task 3: models/generation/ — PR #3

**Files:**
- Create: `unturtle/models/__init__.py`（最小）, `unturtle/models/generation/`（cache / cache_utils / block_decode_mixin / diffusion_generation_utils / masked_diffusion_block_mixin / sampler ほか legacy 全ファイル）
- Test: `tests/models/__init__.py`, `tests/models/test_cache.py`, `tests/models/test_sampler.py`

- [ ] **Step 3-1: issue→PR 手順**（`feat/N-port-generation`）

- [ ] **Step 3-2: テスト先行コピー → 失敗確認**

```bash
rsync -a --exclude='__pycache__' $LEGACY/tests/models/__init__.py $NEW/tests/models/__init__.py
rsync -a --exclude='__pycache__' $LEGACY/tests/models/test_cache.py $LEGACY/tests/models/test_sampler.py $NEW/tests/models/
cd $NEW && uv run python -m pytest tests/models/ -m "not slow" -v 2>&1 | tail -5
```

Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3-3: 本体コピー → グリーン確認**

```bash
rsync -a --exclude='__pycache__' $LEGACY/unturtle/models/generation/ $NEW/unturtle/models/generation/
# models/__init__.py は legacy 版から非推奨エイリアス（llada/dream/modernbert/a2d の re-export）を
# 取り除いた最小版を作る。まず legacy 版を確認:
sed -n '1,60p' $LEGACY/unturtle/models/__init__.py
cd $NEW && uv run python -m pytest tests/models/ -m "not slow" -v
```

Expected: PASS。`unturtle/models/__init__.py` にエイリアス・`AutoConfig.register` の legacy `a2d-*` 分岐が紛れ込んでいないことを `grep -n "a2d-\|llada\|dream\|modernbert" $NEW/unturtle/models/__init__.py` で確認（tiny-a2d-* のみ許容、ただし conversion 移植前なので register は Task 5 まで遅延 import になっているか確認）。

- [ ] **Step 3-4: lint・コミット・PR・マージ**

```bash
git add -A && git commit -m "✨ feat(generation): port shared cache / block-decode / sampler registry (#N)"
```

---

### Task 4: models/backbones/ — PR #4

**Files:**
- Create: `unturtle/models/backbones/`（llada / dream / modernbert、legacy 全ファイル）
- Test: `tests/models/test_dream.py`, `tests/models/test_llada.py`

- [ ] **Step 4-1: issue→PR 手順**（`feat/N-port-backbones`）

- [ ] **Step 4-2: テスト先行コピー → 失敗確認 → 本体コピー → グリーン確認**

```bash
rsync -a --exclude='__pycache__' $LEGACY/tests/models/test_dream.py $LEGACY/tests/models/test_llada.py $NEW/tests/models/
cd $NEW && uv run python -m pytest tests/models/test_dream.py tests/models/test_llada.py -m "not slow" -v 2>&1 | tail -3   # FAIL
rsync -a --exclude='__pycache__' $LEGACY/unturtle/models/backbones/ $NEW/unturtle/models/backbones/
cd $NEW && uv run python -m pytest tests/models/ -m "not slow" -v   # PASS（既存含め全部）
```

Expected: backbones 内の `from unturtle.models.modernbert import …`（エイリアス参照が1箇所ある）は `unturtle.models.backbones.modernbert` に書き換える。
`LLaDAModelLM` の HF 互換要件（`post_init()` / `tie_weights(**kwargs)` / tolerant `forward`）はテストが担保するので挙動変更しない。

- [ ] **Step 4-3: lint・コミット・PR・マージ**

```bash
git add -A && git commit -m "✨ feat(backbones): port llada / dream / modernbert native backbones (#N)"
```

---

### Task 5: models/conversion/（Tiny-A2D）— PR #5

**Files:**
- Create: `unturtle/models/conversion/`（a2d/tiny_a2d、legacy 全ファイル）
- Test: `tests/models/test_a2d.py`, `tests/models/test_block_decode.py`, `tests/models/test_parallel_decode.py`, `tests/models/test_block_decode_benchmark.py`

- [ ] **Step 5-1: issue→PR 手順**（`feat/N-port-tiny-a2d`）

- [ ] **Step 5-2: テスト先行 → 失敗確認 → 本体コピー**

```bash
rsync -a --exclude='__pycache__' $LEGACY/tests/models/test_a2d.py $LEGACY/tests/models/test_block_decode.py \
  $LEGACY/tests/models/test_parallel_decode.py $LEGACY/tests/models/test_block_decode_benchmark.py $NEW/tests/models/
cd $NEW && uv run python -m pytest tests/models/test_a2d.py -m "not slow" -v 2>&1 | tail -3   # FAIL
rsync -a --exclude='__pycache__' $LEGACY/unturtle/models/conversion/ $NEW/unturtle/models/conversion/
```

- [ ] **Step 5-3: legacy `a2d-*` load-compat の除去**

```bash
grep -rn "a2d-llama\"\|a2d-qwen2\"\|a2d-qwen3\"\|legacy_model_type\|legacy a2d" \
  $NEW/unturtle/models/conversion/ --include="*.py" | grep -v "tiny-a2d"
```

ヒットした load-compat 分岐（legacy #301/#302 で入れた旧 model_type 受理コード）を削除する。`tiny-a2d-*` の register・分岐は維持。dllm-hub 依存の名残（hub 固有の repo id 分岐等）も同 grep で洗い出して削除。

- [ ] **Step 5-4: グリーン確認・lint・コミット・PR・マージ**

```bash
cd $NEW && uv run python -m pytest tests/models/ -m "not slow" -v   # PASS
git add -A && git commit -m "✨ feat(conversion): port Tiny-A2D without legacy a2d-* load-compat (#N)"
```

---

### Task 6: trainer.py + diffusion/ + eval(smoke) — PR #6（リファクタ込み 2 段）

**Files:**
- Create: `unturtle/trainer.py`, `unturtle/diffusion/`（全ファイル: trainer / block_diffusion_trainer / collator / packed_collator / block_diffusion_collator / schedulers / grpo_trainer ほか）, `unturtle/eval/`（smoke のみ: `__init__.py`, `_answer_parser.py`, `base.py`, `diffusion.py`, `generation.py`, `gsm8k.py`）
- Test: `tests/diffusion/`（全 12 ファイル）, `tests/eval/__init__.py`, `tests/eval/test_evaluators.py`, `tests/eval/test_gsm8k.py`

依存メモ: `diffusion/trainer.py` は `unturtle.eval`（smoke 評価器）を import し、`eval/` は `diffusion` の collator/scheduler を import する**相互依存クラスタ**なので必ず同一 PR で移植する。harness は Task 8。

- [ ] **Step 6-1: issue→PR 手順**（`feat/N-port-diffusion-objectives`）

- [ ] **Step 6-2: テスト先行コピー → 失敗確認**

```bash
rsync -a --exclude='__pycache__' $LEGACY/tests/diffusion/ $NEW/tests/diffusion/
mkdir -p $NEW/tests/eval
rsync -a --exclude='__pycache__' $LEGACY/tests/eval/__init__.py $LEGACY/tests/eval/test_evaluators.py \
  $LEGACY/tests/eval/test_gsm8k.py $NEW/tests/eval/
cd $NEW && uv run python -m pytest tests/diffusion/ tests/eval/ -m "not slow" -v 2>&1 | tail -5   # FAIL
```

- [ ] **Step 6-3: 【移植コミット】本体コピー → グリーン確認**

```bash
rsync -a --exclude='__pycache__' $LEGACY/unturtle/trainer.py $NEW/unturtle/trainer.py
rsync -a --exclude='__pycache__' $LEGACY/unturtle/diffusion/ $NEW/unturtle/diffusion/
rsync -a --exclude='__pycache__' --exclude='experimental' $LEGACY/unturtle/eval/ $NEW/unturtle/eval/
rsync -a --exclude='__pycache__' --exclude='harness' $LEGACY/unturtle/eval/ $NEW/unturtle/eval/  # harness は Task 8
rm -rf $NEW/unturtle/eval/harness $NEW/unturtle/eval/experimental
# eval/__init__.py に harness の eager import があれば（lazy のはずだが）確認:
grep -n "harness" $NEW/unturtle/eval/__init__.py
cd $NEW && uv run python -m pytest tests/diffusion/ tests/eval/ -m "not slow" -v   # PASS
git add -A && git commit -m "✨ feat(diffusion): port DiffusionTrainer / BD3LM / GRPO and smoke evaluators (#N)"
```

注意: `processing_class=tokenizer` を明示する既存テスト規約、loss 正規化が `n_maskable` 基準である点（MDLM/d1 準拠）は挙動契約 — 変更禁止。

- [ ] **Step 6-4: 【リファクタコミット】vendored UnturtleTrainer → UnslothTrainer 継承化**

手順:

```bash
# (1) vendored 版と最新 unsloth 版の diff を取る
diff $NEW/unturtle/trainer.py $NEW/.venv/lib/python3.12/site-packages/unsloth/trainer.py | head -100
# (2) Task 1 Step 1-2 のプローブ記録（dev/local/2026-06-11-unsloth-pin-probe.md）を参照
```

リファクタ後の `unturtle/trainer.py` の目標構造（diff で判明した unturtle 独自差分のみ残す）:

```python
from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments

from unturtle.utils.packing import (...)  # packed-seq 配線（独自差分）


class UnturtleTrainingArguments(UnslothTrainingArguments):
    """unturtle 独自の引数（packed-seq 等）だけを追加する。"""
    ...


class UnturtleTrainer(UnslothTrainer):
    """UnslothTrainer に packed-seq / dLLM 向け差分のみを足す薄い拡張。

    複製していた UnslothTrainer 相当ロジック（optimizer 解決、
    bf16 判定、vision utils 配線等）はすべて継承で受ける。
    """
    ...  # 独自差分のメソッドオーバーライドのみ
```

判断ルール: diff の各ブロックについて「unsloth 由来（→削除して継承）」か「unturtle 独自（→オーバーライドとして残す）」かを分類してから書き換える。分類に迷う差分は**残す**（挙動優先）。`QGaloreConfig` など unturtle 独自クラスは維持。
あわせて dllm 参照実装由来の構造（collator / scheduler の境界、クラス名）を再点検する: 不自然な分割・dllm 固有の命名があれば是正候補としてメモするだけに留め、このコミットでは挙動契約（テスト）を変えない。是正は別 issue。

```bash
cd $NEW && uv run python -m pytest tests/diffusion/ tests/eval/ tests/models/ tests/utils/ -m "not slow" -v   # PASS 維持
git add -A && git commit -m "♻️ refactor(trainer): subclass UnslothTrainer, drop vendored duplicate logic (#N)"
```

リファクタで同一テストがグリーンにできない場合: リファクタコミットを revert し、移植コミットのみで PR を出す。継承化は別 issue に切り出す（退路）。成功した場合は NOTICE から `unturtle/trainer.py` 行を削除。

- [ ] **Step 6-5: GPU 手動チェックポイント①**

```bash
cd $NEW && uv run python -m pytest tests/diffusion/ -v 2>&1 | tail -10   # slow / gpu 含む全部
```

結果（pass/fail/skip 数）を `$NEW/dev/local/2026-06-11-migration-gpu-runs.md` に記録。

- [ ] **Step 6-6: lint・PR・マージ**

---

### Task 7: fast_diffusion_model.py + save.py + unturtle/__init__.py — PR #7（リファクタ込み 2 段）

**Files:**
- Create: `unturtle/fast_diffusion_model.py`, `unturtle/save.py`
- Modify: `unturtle/__init__.py`（legacy 版の re-export を移植）
- Test: `tests/test_fast_diffusion_model.py`, `tests/test_fast_diffusion_generate.py`, `tests/test_e2e_integration.py`, `tests/test_e2e_real_checkpoint.py`

- [ ] **Step 7-1: issue→PR 手順**（`feat/N-port-fast-diffusion-model`）

- [ ] **Step 7-2: テスト先行 → 失敗確認 → 【移植コミット】**

```bash
rsync -a --exclude='__pycache__' $LEGACY/tests/test_fast_diffusion_model.py $LEGACY/tests/test_fast_diffusion_generate.py \
  $LEGACY/tests/test_e2e_integration.py $LEGACY/tests/test_e2e_real_checkpoint.py $NEW/tests/
cd $NEW && uv run python -m pytest tests/test_fast_diffusion_model.py -m "not slow" -v 2>&1 | tail -3   # FAIL
rsync -a --exclude='__pycache__' $LEGACY/unturtle/fast_diffusion_model.py $NEW/unturtle/
rsync -a --exclude='__pycache__' $LEGACY/unturtle/save.py $NEW/unturtle/
# unturtle/__init__.py を legacy 版ベースで再構成（lighteval / studio / エイリアス参照を除去）
sed -n '1,80p' $LEGACY/unturtle/__init__.py   # 内容を確認して移植
cd $NEW && uv run python -m pytest tests/test_fast_diffusion_model.py tests/test_fast_diffusion_generate.py \
  tests/test_e2e_integration.py -m "not slow" -v   # PASS
git add -A && git commit -m "✨ feat(models): port FastDiffusionModel / save and package exports (#N)"
```

- [ ] **Step 7-3: 【リファクタコミット】loader の FastModel 委譲分離**

目標構造: `fast_diffusion_model.py` を2責務に分離する。

```python
# unturtle/fast_diffusion_model.py（リファクタ後の構造）

_NATIVE_MODEL_CLASSES = {...}  # model_type → unturtle native class（llada / dream / tiny-a2d-*）— 現状維持


def _load_native(model_name, config, **kwargs):
    """unturtle native クラスでのロード（現行ロジックを関数に切り出すだけ）。"""


def _load_via_unsloth(model_name, **kwargs):
    """HF 登録済み model_type（modernbert、将来の diffusion_gemma 等）は
    unsloth.FastModel.from_pretrained に委譲し、ロード・量子化・PEFT 適用を任せる。"""
    from unsloth import FastModel
    return FastModel.from_pretrained(model_name, **kwargs)


class FastDiffusionModel:
    @staticmethod
    def from_pretrained(model_name, ...):
        # 1. config を読み model_type を判定
        # 2. native にあれば _load_native、なければ _load_via_unsloth
        # 3. どちらの経路でも diffusion パッチ（bidirectional fast-forward / sampler 登録 /
        #    save パッチ）を適用 — パッチ適用部を _patch_for_diffusion(model) に集約
        ...
```

判断ルール: 委譲経路で `load_in_4bit` / `device_map` / LoRA の挙動が native 経路のテストと矛盾したら、委譲対象を「native 辞書にないモデルのみ」に限定して衝突を回避する。native クラス bypass（trust_remote_code 回避）と sampler registry は**現状維持が契約**。

```bash
cd $NEW && uv run python -m pytest tests/ -m "not slow" -v   # 全体 PASS 維持
git add -A && git commit -m "♻️ refactor(models): split FastDiffusionModel into native loader + FastModel delegation + diffusion patcher (#N)"
```

グリーンにできない場合は Task 6 と同じ退路（revert して別 issue 化）。

- [ ] **Step 7-4: GPU 手動チェックポイント②**

```bash
cd $NEW && uv run python -m pytest tests/test_e2e_real_checkpoint.py tests/models/ -v 2>&1 | tail -10
```

結果を `dev/local/2026-06-11-migration-gpu-runs.md` に追記。

- [ ] **Step 7-5: lint・PR・マージ**

---

### Task 8: eval/harness/ — PR #8

**Files:**
- Create: `unturtle/eval/harness/`（`__init__.py`, `configs.py`, `model_adapter.py`, `runner.py`）
- Test: `tests/eval/test_harness_adapter.py`, `tests/eval/test_harness_configs.py`, `tests/eval/test_harness_runner.py`, `tests/eval/test_eval_import_boundary.py`

- [ ] **Step 8-1: issue→PR 手順**（`feat/N-port-eval-harness`）

- [ ] **Step 8-2: テスト先行 → 失敗確認 → 本体コピー → グリーン確認**

```bash
rsync -a --exclude='__pycache__' $LEGACY/tests/eval/test_harness_adapter.py $LEGACY/tests/eval/test_harness_configs.py \
  $LEGACY/tests/eval/test_harness_runner.py $LEGACY/tests/eval/test_eval_import_boundary.py $NEW/tests/eval/
cd $NEW && uv run python -m pytest tests/eval/ -m "not slow" -v 2>&1 | tail -3   # FAIL
rsync -a --exclude='__pycache__' $LEGACY/unturtle/eval/harness/ $NEW/unturtle/eval/harness/
uv pip install -e ".[eval]"
cd $NEW && uv run python -m pytest tests/eval/ -m "not slow" -v   # PASS
```

不変条件（test_eval_import_boundary が担保）: `import unturtle.eval` は `lm_eval` を要求しない（adapter/runner が lazy import）。

- [ ] **Step 8-3: lint・コミット・PR・マージ**

```bash
git add -A && git commit -m "✨ feat(eval): port lm-eval-harness adapter / runner / DecodingConfig (#N)"
```

---

### Task 9: cli/ + benchmarks/ + examples/ + docs — PR #9（最終）

**Files:**
- Create: `unturtle/cli/`（legacy `unturtle_cli/` から。`commands/studio.py` は除外）, `benchmarks/`, `examples/`（`validate_studio_mdlm_chat.py` 除外）, `docs/dllm-gap-map.md`, `CLAUDE.md`, `AGENTS.md`
- Test: `tests/test_cli_smoke.py`, `tests/examples/test_benchmark_a2d_aligned.py`, `tests/examples/test_validate_a2d_real_inference.py`

- [ ] **Step 9-1: issue→PR 手順**（`feat/N-port-cli-bench-docs`）

- [ ] **Step 9-2: CLI を unturtle.cli として統合**

```bash
rsync -a --exclude='__pycache__' --exclude='commands/studio.py' $LEGACY/unturtle_cli/ $NEW/unturtle/cli/
# パッケージ内 import と studio コマンド登録を除去・書き換え
grep -rn "unturtle_cli\|studio" $NEW/unturtle/cli/ --include="*.py"
# → "from unturtle_cli" を "from unturtle.cli" に全置換、__init__.py の
#    "from unturtle.cli.commands.studio import studio" と "app.command()(studio)" の行を削除
```

- [ ] **Step 9-3: テスト移植（import 書き換え）→ グリーン確認**

```bash
rsync -a --exclude='__pycache__' $LEGACY/tests/test_cli_smoke.py $NEW/tests/
mkdir -p $NEW/tests/examples
rsync -a --exclude='__pycache__' $LEGACY/tests/examples/test_benchmark_a2d_aligned.py \
  $LEGACY/tests/examples/test_validate_a2d_real_inference.py $NEW/tests/examples/
rsync -a --exclude='__pycache__' --exclude='validate_studio_mdlm_chat.py' --exclude='__pycache__' $LEGACY/examples/ $NEW/examples/
# test_cli_smoke.py の import を unturtle_cli → unturtle.cli に書き換え（これのみ許容される修正）
sed -i 's/from unturtle_cli/from unturtle.cli/g; s/import unturtle_cli/import unturtle.cli/g' $NEW/tests/test_cli_smoke.py
cd $NEW && uv run python -m pytest tests/test_cli_smoke.py tests/examples/ -m "not slow" -v   # PASS
```

- [ ] **Step 9-4: CLI スモーク（entry point 動作確認）**

```bash
cd $NEW && uv pip install -e ".[huggingface]" && uv run unturtle --help
```

Expected: train / generate / export / eval が表示され、studio が**ない**こと。

- [ ] **Step 9-5: docs 移植**

```bash
rsync -a $LEGACY/docs/dllm-gap-map.md $NEW/docs/
```

- gap-map に2行追加: 「Encoder backbones」の下に DiffusionGemma（status ❌→候補, P1 相当, transformers ラッパー方式）と Nemotron-Labs-Diffusion tri-mode（3B/8B/14B, 候補）の行。Roadmap「移植後」節をスペックの「移植後の最初のロードマップ」と同期。
- `CLAUDE.md` は legacy 版をベースに作成し、以下を**削除**: Studio 関連、install.sh 注記、非推奨エイリアス節、`a2d-*` load-compat 言及、lighteval 節。以下を**書き換え**: リポジトリマップ（cli 統合後の構成）、unsloth pin、3層構図（transformers/TRL/unsloth/unturtle）の節を冒頭に追加。
- `AGENTS.md` も同様にスリム化して移植。

- [ ] **Step 9-6: GPU 手動チェックポイント③（全量）+ ベンチマーク比較**

```bash
cd $NEW && uv run python -m pytest tests/ -v 2>&1 | tail -15
# ベンチマーク: 新旧同条件で実行して結果を比較
cd $LEGACY && uv run python benchmarks/gsm8k.py > /tmp/bench-legacy.txt 2>&1
cd $NEW && uv run python benchmarks/gsm8k.py > /tmp/bench-new.txt 2>&1
diff /tmp/bench-legacy.txt /tmp/bench-new.txt
```

結果を `dev/local/2026-06-11-migration-gpu-runs.md` に追記。性能回帰（>10% 劣化）があれば unsloth pin 差由来かリファクタ由来かを切り分けてから PR を出す。

- [ ] **Step 9-7: lint・コミット・PR・マージ**

```bash
git add -A && git commit -m "✨ feat(cli,docs): integrate CLI as unturtle.cli, port benchmarks and docs (#N)"
```

---

## 完了定義（スペックと同一）

1. legacy の fast テスト相当（lighteval・studio・import_compat・install 除く）が新 repo で全グリーン
2. GPU 手動テスト（3回分の記録あり）＋ベンチマーク比較で回帰なし
3. `uv pip install -e ".[huggingface]"` → `unturtle --help` / train / generate / eval スモークが通る

## 移植後の最初の issue（本計画のスコープ外、忘備）

1. DiffusionGemma バックボーン（transformers ラッパー、ModernBERT パターン）
2. Nemotron-Labs-Diffusion 評価
3. unsloth CLI plugin 機構の upstream 提案
4. gap-map P1: dLLM-Cache
