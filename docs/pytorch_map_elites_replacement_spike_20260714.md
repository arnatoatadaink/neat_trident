# 検証記録: MAP-Elites(QDax代替)のPyTorch実現可能性スパイク (pytorch-map-elites-replacement-spike)

日付: 2026-07-14
関連タスク: `pytorch-map-elites-replacement-spike`
関連文書: `docs/pytorch_full_port_decision_revisit_20260714.md`,
`docs/jax_rocm_spike_20260714.md`
成果物: `src/pytorch_proto/map_elites.py`, `tests/test_pytorch_map_elites.py`

## 結論: スパイクは通過 — QDax相当のMAP-Elites機能は小規模自前実装で代替可能
## ただし別軸で、このマシン上の`pytest tests/`実行そのものが現状ブロックされている(スパイクとは無関係の既知問題)

## 1. スパイク本題: QDax代替の実現可能性

`src/map_elites_archive.py`(`SkillRepertoire`/`TRIDENTArchive`/`EvolutionLoop`)が
実際に使っているQDaxの`MapElitesRepertoire`のAPI面を洗い出し、`GridArchive`
(`src/pytorch_proto/map_elites.py`)が同等の振る舞いを再現できるか照合した:

| 実際の使用箇所 | QDax API | GridArchive での対応 |
|---|---|---|
| `MapElitesRepertoire.init_default(genotype, centroids)` | 初期化 | `GridArchive.__post_init__`(centroids生成 + fitness/genotype初期化) |
| `self._repo.centroids` | セントロイド参照 | `GridArchive.centroids` |
| `self._repo.fitnesses` | fitness参照 | `GridArchive.fitnesses` |
| `self._repo.add(gen, desc, fit)`(常にbatch=1で呼び出し) | エリート更新 | `GridArchive.add(descriptor, fitness)` → `(adopted, cell_index)` |
| `SkillRepertoire._resolve_cell`(独自numpy実装、QDax内部は不使用) | 最近傍セントロイド解決 | `GridArchive.resolve_cell`(同一の二乗ユークリッド距離式) |
| `filled_cells` / `coverage` / `best_fitness` / `qd_score` | 派生統計 | 同名プロパティとして実装、式も一致 |

**セル解決式・エリート更新条件は既にQDax内部機能に依存せず自前実装だった**
(`_resolve_cell`はnumpyで独自計算)。QDaxが実際に提供している価値は
「genotype/descriptor/fitnessを保持するコンテナ」のみで、`GridArchive`は
これを過不足なく代替する。8/8テスト成功(`tests/test_pytorch_map_elites.py`)。

### 未検証・保留のギャップ(テストファイル内`Test Guarantee Gaps`と同一、再掲)

- **バッチ`.add()`**: QDaxのAPIはバッチ対応だが、`SkillRepertoire.add()`は常に
  batch=1で呼んでおり、`EvolutionLoop`も1世代1レコードのみ追加する呼び出し
  パターン。`GridArchive.add()`はシングルレコード限定だが、現行の呼び出し
  パターンと1:1で一致するため、スパイクの範囲では未実装がブロッカーにならない。
  本格移植時に真のバッチ追加が必要になった場合は追加実装が必要。
- **GPU配置**: CPUテンソルのみで検証。`device="cuda"`/ROCm相当のデバイス上での
  動作は未確認。
- QDaxとのライブ突き合わせ(同一プロセスでQDax版とGridArchive版を並走させて
  同一descriptor列に対する結果を直接比較)はしていない — 独立実装した
  numpy版の参照式との突き合わせに留まる(理由は次節)。
- **fitness同点時の`adopted`判定が異なる**: `SkillRepertoire.add`は更新後の
  `abs(current_fit - record.fitness) < 1e-6`で判定するため同点でも`adopted=True`
  になるが、`GridArchive.add`は`fitness > 既存値`の厳密不等号のため同点は
  `adopted=False`。格納されるfitness自体はどちらの実装でも変化しない
  (同点なら上書きしてもしなくても値は同じ)ため実害はないが、戻り値の
  `adopted`フラグの意味は完全には一致しない。float同点は測度ゼロの事象では
  あるが、`adopted`フラグをロジックに使う呼び出し側を移植する際は注意。

### 結論の位置づけ

**技術的には「通過」**: 代替ライブラリ選定は不要で、既存の`_resolve_cell`
アルゴリズム自体が元々QDax非依存だったため、コンテナ部分の再実装(数十行)
で置き換えられる。`pytorch_full_port_decision_revisit_20260714.md`のゲート
条件を満たす。

ただし「通過」は「本格移植を推奨する」ことを意味しない。同文書が示す通り、
ユーザーの本機(AMD機)導入の主動機はLLM実行であり、NEATはこのマシン上では
副次的利用。本格移植とColab常用(JAXコード現状維持)のどちらを選ぶかは
ユーザー判断であり、本スパイクはその判断材料(技術的に可能)を提供するのみ。

## 2. 副次的に発覚した問題: `pytest tests/`(フルスイート)がこのマシンで完走しない

本スパイクの検証実施中、`poetry run pytest tests/ -q`(CLAUDE.mdが完了条件と
定める全テスト実行)がこのマシン上で正常に完走しないことが判明した。
**これは本スパイクのコード(`src/pytorch_proto/`)が原因ではなく、既存の
JAX-ROCm環境問題が全テストスイート実行時に露呈したもの**(下記で切り分け済み)。

### 切り分け結果

1. **torch関連ファイルを`--ignore`で除外してもハングは再現した** →
   torchとjaxの同一プロセス内共存が原因という当初の仮説は誤りと判明。
2. **原因は`ROCR_VISIBLE_DEVICES`未設定時のJAX-ROCmデバイス列挙ハング**
   (`jax_rocm_spike_20260714.md`で既報告の「異種デュアルGPU可視化で
   `jax.devices()`が45分以上ハング」と同一の`make_c_api_client`スタック)。
   `ROCR_VISIBLE_DEVICES=0 ROCPROFILER_REGISTER_ENABLED=0`を設定すると
   ハングは解消し、6:38で完走する。
3. **ただし上記環境変数を設定しても、6 failed / 52 errors**。エラー内容は
   `jax_rocm_spike_20260714.md`で既報告の`HIP_ERROR_InvalidValue`
   (世代ループ2世代目のデバイス→ホスト転送で決定論的にクラッシュ)と同一。
   新規のバグではない。

### 検証済みの正常系

- `tests/test_pytorch_map_elites.py`単体: 8 passed
- `tests/test_pytorch_map_elites.py` + `tests/test_pytorch_proto.py`
  (torch系のみ): 11 passed
- ハング自体がtorch非依存であることは`--ignore`での再現実験で直接確認済み
  (原因は環境変数の欠落)。ハング解消後の「6 failed / 52 errors」という
  件数がtorch系ファイル追加の有無で変わらないかまでは未確認 — JAX系テストは
  `pytorch_proto`をimportしておらずHIP_ERROR_InvalidValueは本セッション以前
  から既報告のバグであるため件数が変わる経路は考えにくいが、これは推論であり
  実測(例: 変更をstashして同条件で再実行し件数比較)はしていない。

### 現状の運用上の含意(対応はユーザー判断待ち・本スパイクでは実施しない)

- `CLAUDE.md`/`jax-verification.md`が要求する「`poetry run pytest tests/ -q`
  を完了前に実行」は、**現状このマシンでは環境変数なしでは無限ハング、
  環境変数ありでも52件のエラーが出る状態**であり、額面通りには満たせない。
- 対応の選択肢(このスパイクでは未実施・提案のみ):
  1. `ROCR_VISIBLE_DEVICES=0 ROCPROFILER_REGISTER_ENABLED=0`を
     ローカル実行の前提条件として`jax-verification.md`に明記する
  2. JAX-GPU実機テストにマーカー(例: `gpu`)を切り、CPU-onlyで通る
     テストとは別運用にする
  3. `pytest-forked`等でtorch系とjax系を別プロセスに分離する
     (依存追加・JAX JITのプロセスごとの再コンパイルコストとのトレードオフ)
- いずれも本タスクのスコープ外の判断であり、ユーザーへの報告事項として
  留め、CLAUDE.md等の変更は行っていない。

## 次のアクション

1. `pytorch-map-elites-replacement-spike`は完了・PASS(with caveats)として
   TODO更新
2. 本格移植 vs Colab常用の最終判断はユーザーに委ねる
   (`pytorch_full_port_decision_revisit_20260714.md`のゲートが開いた、
   という事実のみ提供)
3. 「フルテストスイートがこのマシンで完走しない」問題は新規TODO候補として
   別途ユーザーに提案する(本文書のセクション2)
