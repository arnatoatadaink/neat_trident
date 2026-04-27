# TRIDENT 使い方ガイド

TRIDENT は NEAT (NeuroEvolution of Augmenting Topologies) を用いて、MED Framework 向けの自律的スキル探索・収集を行うサブシステムです。
FAISSベクトル検索の代替・補完として NEAT で進化させた検索関数を提供します。

---

## 目次

1. [環境セットアップ](#1-環境セットアップ)
2. [動作確認](#2-動作確認)
3. [コアコンポーネントの使い方](#3-コアコンポーネントの使い方)
4. [スクリプト一覧](#4-スクリプト一覧)
5. [テスト](#5-テスト)
6. [MED との統合](#6-med-との統合)
7. [ハイパーパラメーター最適化 (Optuna)](#7-ハイパーパラメーター最適化-optuna)
8. [ベンチマーク](#8-ベンチマーク)
9. [トラブルシューティング](#9-トラブルシューティング)

---

## 1. 環境セットアップ

### 前提

- Python 3.13
- Poetry
- (オプション) CUDA 12 対応 GPU + WSL2

### インストール

```bash
# 依存パッケージインストール
poetry install

# TensorNEAT は PyPI 未公開のため GitHub から直接インストール
.venv/bin/pip install git+https://github.com/EMI-Group/tensorneat.git

# Optuna (ハイパーパラメーター最適化に使う場合)
.venv/bin/pip install optuna
```

### GPU 有効化 (WSL2 + CUDA 12)

JAX の GPU バックエンドは `LD_LIBRARY_PATH` に CUDA ライブラリを追加することで有効になります。

```bash
# ~/.bashrc に追記済みの場合は source するだけ
source ~/.bashrc

# 確認
poetry run python -c "import jax; print(jax.devices())"
# GPU があれば [CudaDevice(id=0)] のように表示される
```

---

## 2. 動作確認

```bash
# 環境チェック (JAX・FAISS・TensorNEAT の疎通確認)
poetry run python scripts/phase0_verify.py

# 全テスト実行 (193 テスト、約 8 分)
poetry run pytest tests/ -q

# MED 統合レイヤー確認 (スタブ使用、約 1 分)
poetry run python scripts/med_integration_verify.py

# 実 MED DomainIndex との結合確認 (MED_ROOT 要設定)
MED_ROOT=/path/to/MED poetry run python scripts/med_integration_verify.py
```

---

## 3. コアコンポーネントの使い方

### 3-1. NeatIndexer (A型 — ベクトル近傍探索)

NEAT でクエリ→近傍スコアベクトルへの変換関数を進化させます。

```python
import numpy as np
from src.interfaces.neat_indexer import NeatIndexer

# コーパス準備
corpus = np.random.randn(100, 16).astype(np.float32)

# 初期化・進化
ni = NeatIndexer(
    input_dim=16,
    pop_size=400,
    species_size=20,
    generation_limit=50,
    seed=42,
)
ni.fit(corpus)

# 推論
query = np.random.randn(16).astype(np.float32)
output = ni.transform(query)   # shape: (16,)
```

### 3-2. HybridIndexer (FAISS / NEAT / hybrid 切り替え)

```python
from src.interfaces.neat_indexer import HybridIndexer

hi = HybridIndexer(neat_indexer=ni, mode="hybrid")  # "faiss" | "neat" | "hybrid"
hi.add(["doc_0", "doc_1", ...], corpus)
results = hi.search(query, k=5)
# → [("doc_3", 0.91), ("doc_0", 0.87), ...]
```

### 3-3. NeatGate (B型 — 報酬ゲート)

クエリとスキル候補の関連度を NEAT でスコアリングします。

```python
from src.interfaces.neat_gate import NeatGate

gate = NeatGate(input_dim=16, pop_size=100, generation_limit=20)
gate.fit(corpus)
score = gate.score(query, candidate)  # float
```

### 3-4. NeatSlotFiller (C型 — スロット充填)

知識グラフへのエンティティ充填を NEAT で実行します。

```python
from src.interfaces.neat_slot_filler import NeatSlotFiller

sf = NeatSlotFiller(input_dim=16, pop_size=100, generation_limit=20)
sf.fit(corpus)
filled = sf.fill(query, candidates)  # list[tuple[str, float]]
```

### 3-5. ContextSensitiveSearch (AssociationFn)

文脈ベクトルを考慮した連想スコア付き検索です。

```python
from src.med_integration.context_search import AssociationFn, ContextSensitiveSearch

af = AssociationFn(weights=[0.4, 0.2, 0.2, 0.2])  # 重みは合計 1.0
css = ContextSensitiveSearch(association_fn=af)

corpus_texts = ["document text here", ...]
css.add(corpus_texts, corpus)

context = np.random.randn(16).astype(np.float32)
results = css.search(query, context=context, k=5)
# → SearchResult のリスト (base_score + assoc_score)
```

### 3-6. NEATAssociationFn (NEAT で進化させた AssociationFn)

```python
from src.med_integration.neat_assoc_evolver import AssociationFnEvolver

evolver = AssociationFnEvolver(dim=16, pop_size=50, generation_limit=30)

# フィードバックペアで学習
feedback = [
    (query_vec, candidate_vec, context_vec, 1.0),   # 正例
    (query_vec, other_vec,     context_vec, 0.0),   # 負例
]
neat_af = evolver.evolve(feedback)

# ContextSensitiveSearch に差し込む
css = ContextSensitiveSearch(association_fn=neat_af)
```

### 3-7. MAP-Elites アーカイブ

```python
from src.map_elites_archive import TRIDENTArchive
import numpy as np

archive = TRIDENTArchive(grid_sizes={"indexer": 8, "gate": 8, "slot_filler": 8})

descriptor = np.array([0.5, 0.7], dtype=np.float32)  # BCS (2次元)
archive.add_indexer(skill_obj=ni, fitness=0.85, descriptor=descriptor)

best = archive.best_indexer()
```

### 3-8. ES-HyperNEAT

大次元空間 (384次元など) 向けに基板エンコーディングで効率的にトポロジーを探索します。

```python
from src.es_hyperneat import ESHyperNEATIndexer

indexer = ESHyperNEATIndexer(input_dim=384, pop_size=50, generation_limit=20)
indexer.fit(corpus_384dim)
output = indexer.transform(query_384dim)
```

---

## 4. スクリプト一覧

| スクリプト | 用途 | 実行例 |
|-----------|------|--------|
| `phase0_verify.py` | 環境・依存パッケージ疎通確認 | `poetry run python scripts/phase0_verify.py` |
| `neat_benchmark.py` | CPU/GPU ベンチマーク (30分) | `poetry run python scripts/neat_benchmark.py --max-seconds 1800` |
| `neat_optuna_tune.py` | Optuna NEAT ハイパーパラメーター最適化 | `poetry run python scripts/neat_optuna_tune.py --n-trials 50` |
| `med_integration_verify.py` | MED 統合レイヤー 7 チェック | `poetry run python scripts/med_integration_verify.py` |
| `eval_384dim.py` | 384 次元実埋め込みで ContextSensitiveSearch 評価 | `poetry run python scripts/eval_384dim.py` |
| `faiss_hybrid_verify.py` | HybridIndexer FAISS/NEAT 切り替え確認 | `poetry run python scripts/faiss_hybrid_verify.py` |
| `novelty_search_verify.py` | Novelty Search 動作確認 | `poetry run python scripts/novelty_search_verify.py` |

---

## 5. テスト

```bash
# 全テスト (193 tests)
poetry run pytest tests/ -q

# 特定ファイルのみ
poetry run pytest tests/test_neat_indexer.py -v

# integration マーク付きテストも含める (sentence-transformers が必要)
poetry run pytest tests/ -m "integration" -v
```

### テストファイル一覧

| ファイル | 対象 | テスト数 |
|---------|------|---------|
| `test_neat_indexer.py` | NeatIndexer + HybridIndexer | 17 |
| `test_neat_gate.py` | NeatGate | 13 |
| `test_neat_slot_filler.py` | NeatSlotFiller | 17 |
| `test_novelty_search.py` | NoveltyArchive | 21 |
| `test_map_elites.py` | TRIDENTArchive | 21 |
| `test_es_hyperneat.py` | ESHyperNEATIndexer | 19 |
| `test_context_search.py` | AssociationFn / ContextSensitiveSearch | 28 |
| `test_hyperbolic_association.py` | HyperbolicAssociationFn | 16 |
| `test_384dim_interface.py` | 384 次元統合 | 11 |
| `test_neat_assoc_evolver.py` | AssociationFnEvolver | 19 |
| `test_domain_index_adapter.py` | DomainIndexAdapter | 11 |

---

## 6. MED との統合

### スタブを使った開発フロー

MED 本体がなくても開発・テストが可能です。

```python
from src.med_integration.stub_med import StubMEDIndexer, StubMEDSkillStore
from src.med_integration.trident_adapter import TRIDENTMEDAdapter
from src.interfaces.neat_indexer import NeatIndexer, HybridIndexer
import numpy as np

DIM = 16
corpus = np.random.randn(100, DIM).astype(np.float32)
doc_ids = [f"doc_{i}" for i in range(100)]

ni = NeatIndexer(input_dim=DIM, pop_size=100, generation_limit=20)
ni.fit(corpus)
hi = HybridIndexer(neat_indexer=ni, mode="neat")

adapter = TRIDENTMEDAdapter(
    hybrid_indexer=hi,
    med_indexer=StubMEDIndexer(dimension=DIM),
    med_skill_store=StubMEDSkillStore(),
    dimension=DIM,
)

adapter.sync_indexer(doc_ids=doc_ids, corpus=corpus)
results = adapter.search(np.random.randn(DIM).astype(np.float32), k=5)
# → [("doc_42", 0.91), ...]
```

### 実 MED DomainIndex との接続

```python
import sys, os

# sys.modules の src 名前衝突を回避してから MED をインポートする
saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "src" or k.startswith("src.")}
sys.path.insert(0, "/path/to/MED")
try:
    from src.memory.faiss_index import DomainIndex
    from src.common.config import FAISSIndexConfig
finally:
    for k in list(sys.modules):
        if k == "src" or k.startswith("src."):
            del sys.modules[k]
    sys.modules.update(saved)

from src.med_integration.domain_index_adapter import DomainIndexAdapter

cfg = FAISSIndexConfig(dim=16, initial_type="Flat", metric="inner_product")
di  = DomainIndex(cfg)
med_indexer = DomainIndexAdapter(di, dimension=16)

# あとは StubMEDIndexer と同じように使える
adapter = TRIDENTMEDAdapter(
    hybrid_indexer=hi,
    med_indexer=med_indexer,
    med_skill_store=StubMEDSkillStore(),
    dimension=16,
)
```

> **注意**: MED 側に `MEDSkillStoreProtocol` 相当のコンポーネントがないため、
> SkillStore は現状 `StubMEDSkillStore` を継続使用します。

---

## 7. ハイパーパラメーター最適化 (Optuna)

### 基本実行

```bash
# 30 試行 (デフォルト)
poetry run python scripts/neat_optuna_tune.py

# 50 試行、本番サイズ
poetry run python scripts/neat_optuna_tune.py --n-trials 50 --dim 16 --corpus-size 100

# 既存 study を再開 (同じ study-name を指定)
poetry run python scripts/neat_optuna_tune.py --study-name neat_v1 --n-trials 20
```

### オプション一覧

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--n-trials` | 30 | 試行数 |
| `--dim` | 16 | 埋め込み次元 |
| `--corpus-size` | 100 | コーパスサイズ |
| `--study-name` | `neat_tune` | Optuna study 名 |
| `--seed` | 0 | 乱数シード |
| `--n-jobs` | 1 | 並列試行数 |

### 探索空間

| パラメーター | 型 | 範囲 |
|------------|---|-----|
| `pop_size` | int (log) | 50 〜 1000 |
| `species_size` | int | 5 〜 100 |
| `generation_limit` | int | 10 〜 80 |
| `max_nodes` | categorical | 64 / 128 / 256 |
| `max_conns` | categorical | 128 / 512 / 1024 |

> `max_nodes`/`max_conns` を categorical にしているのは、値が変わると TensorNEAT が JIT 再コンパイル (~30s) を行うためです。

### 結果の可視化

結果は `logs/optuna_neat.db` (SQLite) に累積保存されます。

```bash
pip install optuna-dashboard
optuna-dashboard logs/optuna_neat.db
# ブラウザで http://127.0.0.1:8080 を開く
```

---

## 8. ベンチマーク

```bash
# 30 分フルベンチ (CPU/GPU 自動検出)
poetry run python scripts/neat_benchmark.py --max-seconds 1800

# 10 分、pop=400
poetry run python scripts/neat_benchmark.py --max-seconds 600 --pop-size 400

# pop=400, species_size=20 固定 (推奨設定の再現)
poetry run python scripts/neat_benchmark.py --max-seconds 600 --pop-size 400 --species-size 20
```

ログは `logs/` 以下に JSONL 形式で保存されます。過去の実験結果は `LOG.md` を参照してください。

### 既知のベストコンフィグ (RTX 4060, dim=32, corpus=100)

| 設定 | ms/gen | 収束時間 | best fitness |
|-----|--------|---------|-------------|
| pop=100 | 14 ms | 8.2s | — |
| pop=400, species=20 | 66 ms | — | -0.586 |
| pop=1000 | 152 ms | — | -0.590 |

---

## 9. トラブルシューティング

### `ModuleNotFoundError: No module named 'tensorneat'`

```bash
.venv/bin/pip install git+https://github.com/EMI-Group/tensorneat.git
```

### `ModuleNotFoundError: No module named 'pydantic'`

MED 側の依存です。

```bash
.venv/bin/pip install pydantic pydantic-settings
```

### JAX が GPU を認識しない

```bash
# LD_LIBRARY_PATH に nvidia ライブラリを追加する (WSL2)
export LD_LIBRARY_PATH=$(find .venv -name "libcuda.so.1" -o -name "libnvjitlink.so.12" | xargs dirname | sort -u | tr '\n' ':')$LD_LIBRARY_PATH
poetry run python -c "import jax; print(jax.devices())"
```

恒久設定は `~/.bashrc` に同様の行を追加してください。

### `CUDA_ERROR_INVALID_ARGUMENT: Version does not match the format X.Y.Z`

無害なログメッセージです (JAX の内部チェック)。計算結果には影響しません。

### JIT コンパイルが毎回走って遅い

`max_nodes`/`max_conns` が試行ごとに変わると再コンパイルが発生します。
Optuna スクリプトでは categorical にすることで防いでいます。
手動ループでも同じ値を使い回してください。

### T4 GPU が使えない

T4 はサーバー向けカードでドライバが未インストールです。使用不可。
