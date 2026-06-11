# Unturtle 移植設計 — legacy リポジトリからのクリーン再構築

- 日付: 2026-06-11
- 移植元: `nishide-dev/unturtle-legacy`（アーカイブ済み、旧 unsloth フォーク）
- 移植先: `nishide-dev/unturtle`（本リポジトリ、独立リポジトリ）

## 背景と位置付け

unsloth 本体に DiffusionGemma 対応が入ったことを機に、unturtle の立ち位置を再定義した。

- **transformers** = モデル実装＋モデル内 loss（DiffusionGemma の diffusion 機構はここにある）
- **TRL** = objective 別 trainer（DPO / GRPO / PPO …）
- **unsloth** = 上記両方へのパッチによる高速化（学習ループは自作しない）
- **unturtle** = **dLLM 手法レイヤー**：AR→Diffusion 変換、diffusion objective（TRL と同じレイヤー）、モデル横断の推論高速化、canonical 評価

unturtle の `DiffusionTrainer` は TRL の objective 別 trainer と同格の存在であり、unsloth の
設計思想（trainer をパッチする）と対立しない。逆に、モデルが loss と生成を内蔵する場合
（DiffusionGemma）は標準パス（`SFTTrainer`/`UnslothTrainer`）で学習し、`DiffusionTrainer`
は objective を持たないモデル（LLaDA / Dream / Tiny-A2D 変換モデル）専用とする。

旧リポジトリは unsloth の git 履歴（約5,030コミット）を引き継いだフォークだったが、
実際の依存は pip（`unsloth>=…`）経由であり、git 履歴による upstream 同期は運用されて
いなかった。本移植で履歴・互換コード・Studio を捨て、上記の3層構図に合わせて作り直す。

## 決定事項サマリ

| 論点 | 決定 |
|---|---|
| リポジトリ | 新規独立リポジトリ `nishide-dev/unturtle`。旧は `unturtle-legacy` にリネームしアーカイブ（issue/PR 履歴は read-only で保存） |
| unsloth git 履歴 | 持ち込まない。依存は pip のみ |
| Studio（126ファイル/46k行、AGPL同梱） | 移植しない。upstream unsloth の Studio に任せる |
| lighteval（experimental） | 移植しない。lm-eval-harness に一本化 |
| 非推奨エイリアス（`unturtle.models.{llada,dream,modernbert}`） | 捨てる |
| `a2d-*` load-compat（legacy #301/#302） | 捨てる。checkpoint 移行も**行わない**（Nemotron-Labs-Diffusion 3B 等の小型公開モデル登場により dllm-hub レガシー checkpoint への依存が不要に。必要時は legacy リポジトリで読む） |
| Diffu-GRPO trainer | 移植する |
| benchmarks/ | 移植する（移植前後の性能回帰確認に使用） |
| パッケージ構成 | 単一パッケージ。`unturtle_cli` は `unturtle.cli` に統合（entry point: `unturtle = "unturtle.cli:app"`） |
| DiffusionGemma バックボーン | 移植完了後の最初の新規 feature issue（移植には含めない） |
| 移植方式 | モジュール順次移植 PR（1モジュール=1 issue=1 PR、テスト先行） |
| 完了定義 | 「テストを契約とした移植＋内部リファクタ」：旧テストが import パス修正のみでグリーン、内部実装は unsloth 継承に寄せ直してよい |

## 新リポジトリ構成

```text
unturtle/                      # repo root
├── unturtle/
│   ├── trainer.py             # UnturtleTrainer — UnslothTrainer 継承にリファクタ
│   ├── fast_diffusion_model.py  # loader（FastModel 委譲）+ diffusion patcher に分離
│   ├── save.py
│   ├── diffusion/             # objectives: DiffusionTrainer / collator / scheduler / GRPO
│   ├── kernels/               # Triton kernels / fast LoRA
│   ├── models/
│   │   ├── backbones/         # llada / dream / modernbert（エイリアスなし）
│   │   ├── conversion/a2d/tiny_a2d/
│   │   └── generation/        # cache / block-decode / sampler registry
│   ├── eval/                  # lm-eval-harness 連携のみ
│   ├── utils/                 # attention_dispatch / packing（vendored 由来）
│   └── cli/                   # 旧 unturtle_cli を統合
├── tests/                     # モジュールと同時に移植
├── benchmarks/
├── docs/                      # dllm-gap-map.md ほか
├── CLAUDE.md / AGENTS.md      # スリム化して移植
├── NOTICE                     # vendored ファイルの出典（unsloth リポジトリ＋コミットID、Apache-2.0）
└── pyproject.toml             # unsloth への pip 依存（最新 pin）
```

vendored（unsloth 由来）コードは legacy 時点で4箇所に限定:
`trainer.py` / `utils/attention_dispatch.py` / `utils/packing.py` / `kernels/__init__.py`。
残すものは NOTICE に出典コミット ID を記録する。

git 管理外資産（`dev/local/`・`dev/papers/`・`.references/`）は手動コピー。
`dev/repos/` は再クローン。

## 移植フェーズ

1モジュール = 1 issue = 1 PR。各 PR は旧リポジトリの対応テストを**先に**持ち込み、
import パス修正のみでグリーンにしてから本体を移す。リファクタ量が大きいフェーズ
（3・6）は「移植コミット → リファクタコミット」の2段構成で退路を残す。

| # | 内容 | リファクタ方針 |
|---|---|---|
| 1 | 骨格: pyproject / CI / NOTICE / README | unsloth pin を**最新（DiffusionGemma 対応版）に引き上げてから開始**。legacy クローン側の venv で unsloth だけ最新に上げて fast テストを一度回し、壊れる箇所を後続フェーズのリファクタ対象として把握する |
| 2 | kernels/ + utils/ | vendored の packing / attention_dispatch は unsloth_zoo に同等物があれば置換、なければ NOTICE 付きで維持 |
| 3 | trainer.py + diffusion/ | `UnturtleTrainer` を `UnslothTrainer` 継承に書き換え、複製ロジックを削除。`DiffusionTrainer` はその上に。dllm 参照実装由来の構造（collator/scheduler 境界）も再点検 |
| 4 | models/backbones/ | エイリアスなしで素直に移植 |
| 5 | models/conversion/（tiny_a2d） | `a2d-*` load-compat なしで移植。dllm-hub 依存の名残（model_type 分岐等）を除去 |
| 6 | models/generation/ + fast_diffusion_model.py + save.py | FastDiffusionModel を loader（HF 登録済みモデルは unsloth `FastModel` に委譲）と diffusion patcher に分離。native クラス（llada/dream）bypass と sampler registry は維持 |
| 7 | eval/ | lighteval を除き素直に移植。`import unturtle.eval` が `lm_eval` を要求しない不変条件を維持 |
| 8 | cli/ + benchmarks/ + docs | CLI 統合（`unturtle.cli`）、gap-map 更新（Nemotron-Labs-Diffusion tri-mode を DiffusionGemma と並ぶ新 backbone 候補として記載）、CLAUDE.md / AGENTS.md スリム化 |

## テスト戦略

- 各 PR の合格条件: 旧リポジトリの該当テストが import パス修正のみでグリーン。
  リファクタコミットは同一テストのグリーン維持が条件。
- CI（GitHub Actions）: CPU の fast テスト（`-m "not slow"` 相当、マーカー維持）＋
  `ruff check` / `ruff format --check`。
- GPU 依存テスト（Triton / Flash / 実 checkpoint 系 `slow`+`gpu`）は CI に入れず、
  **フェーズ3完了時・フェーズ6完了時・全移植完了時の計3回**、手元 GPU で手動実行して記録。
- benchmarks/ は移植の最後に旧リポジトリと同条件で1回実行し、性能回帰がないことを確認
  （unsloth pin 引き上げの影響検出を兼ねる）。

## 完了定義

1. 旧 repo の fast テスト相当（lighteval・studio 関連を除く）が新 repo で全グリーン
2. GPU 手動テスト＋ベンチマーク比較で回帰なし
3. `pip install -e ".[huggingface]"` → `unturtle train/generate/eval` の CLI スモークが通る

## 捨てるものリスト（移植しない確定分）

- `unturtle_studio/`（AGPL 同梱）
- `unturtle.eval.experimental.lighteval`
- 非推奨エイリアス `unturtle/models/{llada,dream,modernbert}.py` と `models/a2d/`
- `a2d-*` load-compat と関連テスト
- `install.sh` ラッパー（uv セットアップ手順に一本化）
- unsloth の git 履歴・upstream remote

## リスクと対策

| リスク | 対策 |
|---|---|
| unsloth pin 引き上げで trainer パッチ / FastModel API が変化（最大の不確定要素） | フェーズ1で旧コード＋最新 pin の fast テストを先に回し、破損箇所を事前把握 |
| `UnslothTrainer` 継承化で vendored 版の独自修正（packed-seq 等）が暗黙に失われる | 着手前に vendored 版と upstream 版の diff を取り、差分をテストで固定 |
| `FastModel` 委譲が native クラス bypass と干渉 | フェーズ6は移植コミットで現行構造のまま通し、リファクタコミットで委譲化 |

## 移植後の最初のロードマップ（参考、本移植のスコープ外）

1. DiffusionGemma バックボーン（transformers 実装の薄いラッパー、ModernBERT パターン）
2. Nemotron-Labs-Diffusion バックボーン評価（3B/8B/14B、AR+diffusion tri-mode）
3. unsloth CLI へのプラグイン機構の upstream 提案（entry-points 走査 → `unsloth diffusion …`）
4. gap-map P1: dLLM-Cache
