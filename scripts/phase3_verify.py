"""
Phase 3 動作確認スクリプト
NeatSlotFiller (C型 I/F) の基本動作を小規模データで検証する。

文脈次元  = 8
スキーマ  = KG ("node", "relation", "weight")
サンプル数 = 40 (訓練) + 10 (テスト)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

print("=" * 60)
print("Phase 3 — NeatSlotFiller (C型 I/F) 動作確認")
print("=" * 60)

# ─── 合成データ作成 ───
# node     = tanh(ctx[0])         ← 文脈の第1成分
# relation = tanh(ctx[1] * 0.5)   ← 文脈の第2成分 (弱め)
# weight   = tanh(ctx[2] ** 2 - 0.5) ← 非線形マッピング
rng = np.random.default_rng(2)
CONTEXT_DIM = 8
SCHEMA      = ("node", "relation", "weight")
N_TRAIN     = 40
N_TEST      = 10

def make_dataset(n, rng):
    ctx = rng.standard_normal((n, CONTEXT_DIM)).astype(np.float32)
    tgt = np.zeros((n, len(SCHEMA)), dtype=np.float32)
    tgt[:, 0] = np.tanh(ctx[:, 0])
    tgt[:, 1] = np.tanh(ctx[:, 1] * 0.5)
    tgt[:, 2] = np.tanh(ctx[:, 2] ** 2 - 0.5)
    return ctx, tgt

train_ctx, train_tgt = make_dataset(N_TRAIN, rng)
test_ctx,  test_tgt  = make_dataset(N_TEST,  rng)

print(f"\nデータ:")
print(f"  context_dim : {CONTEXT_DIM}")
print(f"  schema      : {SCHEMA}")
print(f"  train/test  : {N_TRAIN}/{N_TEST}")
print(f"  target range: [{train_tgt.min():.3f}, {train_tgt.max():.3f}]")

# ─── インポート確認 ───
print("\n[1] NeatSlotFiller インポート確認 ... ", end="", flush=True)
try:
    from src.interfaces.neat_slot_filler import (
        NeatSlotFiller,
        NeatKGWriter,
        SlotFitProblem,
        compute_slot_bcs_descriptor,
        KG_SCHEMA, SQL_SCHEMA, GRPO_SCHEMA,
    )
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── スキーマ定数 ───
print("[2] スキーマ定数確認 ... ", end="", flush=True)
try:
    assert KG_SCHEMA   == ("node", "relation", "weight")
    assert SQL_SCHEMA  == ("table", "col", "condition")
    assert GRPO_SCHEMA == ("reward", "reason", "target")
    print(f"✅  KG={KG_SCHEMA}, SQL={SQL_SCHEMA}, GRPO={GRPO_SCHEMA}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── SlotFitProblem ───
print("[3] SlotFitProblem 構築 ... ", end="", flush=True)
try:
    prob = SlotFitProblem(contexts=train_ctx, targets=train_tgt)
    assert prob.input_shape  == (CONTEXT_DIM,)
    assert prob.output_shape == (len(SCHEMA),)
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── NeatSlotFiller 構築 ───
print("[4] NeatSlotFiller 構築 ... ", end="", flush=True)
try:
    filler = NeatSlotFiller(
        slot_names=SCHEMA,
        context_dim=CONTEXT_DIM,
        pop_size=20,
        species_size=3,
        generation_limit=5,
        seed=42,
    )
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── fit() ───
print("[5] NeatSlotFiller.fit() 進化 (5世代) ...")
try:
    filler.fit(contexts=train_ctx, targets=train_tgt)
    print("    ✅ fit 完了")
except Exception as e:
    print(f"    ❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── fill() 単一 ───
print("[6] fill() 単一入力 ... ", end="", flush=True)
try:
    slots = filler.fill(test_ctx[0])
    assert set(slots.keys()) == set(SCHEMA)
    assert all(isinstance(v, float) for v in slots.values())
    assert all(-1.0 <= v <= 1.0 for v in slots.values()), f"範囲外: {slots}"
    print(f"✅  {slots}")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── fill_batch() ───
print("[7] fill_batch() バッチ入力 ... ", end="", flush=True)
try:
    slots_list = filler.fill_batch(test_ctx)
    assert len(slots_list) == N_TEST
    assert all(set(s.keys()) == set(SCHEMA) for s in slots_list)
    print(f"✅  {N_TEST} サンプル, 例={slots_list[0]}")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── fill_rate() ───
print("[8] fill_rate() 確認 ... ", end="", flush=True)
try:
    fr = filler.fill_rate(test_ctx)
    assert 0.0 <= fr <= 1.0
    print(f"✅  fill_rate={fr:.3f}")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── bcs_descriptor() ───
print("[9] bcs_descriptor() 確認 ... ", end="", flush=True)
try:
    desc = filler.bcs_descriptor(test_ctx, targets=test_tgt)
    assert desc.shape == (2,)
    assert np.all((desc >= 0) & (desc <= 1)), f"範囲外: {desc}"
    print(f"✅  [fill_rate={desc[0]:.3f}, accuracy={desc[1]:.3f}]")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── NeatKGWriter ───
print("[10] NeatKGWriter 動作確認 ... ", end="", flush=True)
try:
    kg_filler = NeatSlotFiller(
        slot_names=KG_SCHEMA,
        context_dim=CONTEXT_DIM,
        pop_size=20,
        species_size=3,
        generation_limit=3,
        seed=0,
    )
    kg_filler.fit(contexts=train_ctx, targets=train_tgt)
    writer = NeatKGWriter(neat_filler=kg_filler)
    triple = writer.generate_triple(test_ctx[0])
    assert set(triple.keys()) == {"node", "relation", "weight"}
    assert 0.0 <= triple["weight"] <= 1.0, f"weight 範囲外: {triple['weight']}"
    print(f"✅  triple={triple}")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── サマリー ───
print("\n" + "=" * 60)
print("Phase 3 動作確認 サマリー")
print("=" * 60)
print("✅ NeatSlotFiller (C型 I/F) 基本動作確認完了")
print(f"   context_dim : {CONTEXT_DIM}")
print(f"   schema      : {SCHEMA}")
print(f"   fill_rate   : {fr:.3f}")
print(f"   BCS 記述子  : [充填率={desc[0]:.3f}, 正確性={desc[1]:.3f}]")
print()
print("次のステップ:")
print("  - Phase 4: QDax MAP-Elites 統合 (A/B/C型スキルアーカイブ)")
print("  - Novelty Search カスタム実装")
print("  - 本番次元・本番データでの統合テスト")
