"""
Phase 2 動作確認スクリプト
NeatGate (B型 I/F) の基本動作を小規模データで検証する。

文脈次元  = 8
スキル数  = 4
サンプル数 = 40 (訓練) + 10 (テスト)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

print("=" * 60)
print("Phase 2 — NeatGate (B型 I/F) 動作確認")
print("=" * 60)

# ─── 合成データ作成 ───
# 文脈ベクトルの前半が正 → スキル0/1 を有効化
# 文脈ベクトルの後半が正 → スキル2/3 を有効化
rng = np.random.default_rng(1)
CONTEXT_DIM = 8
NUM_SKILLS  = 4
N_TRAIN     = 40
N_TEST      = 10

def make_dataset(n, rng):
    contexts = rng.standard_normal((n, CONTEXT_DIM)).astype(np.float32)
    targets  = np.zeros((n, NUM_SKILLS), dtype=np.float32)
    targets[:, 0] = (contexts[:, 0] > 0).astype(np.float32)
    targets[:, 1] = (contexts[:, 1] > 0).astype(np.float32)
    targets[:, 2] = (contexts[:, 4] > 0).astype(np.float32)
    targets[:, 3] = (contexts[:, 5] > 0).astype(np.float32)
    return contexts, targets

train_ctx, train_tgt = make_dataset(N_TRAIN, rng)
test_ctx,  test_tgt  = make_dataset(N_TEST,  rng)

print(f"\nデータ:")
print(f"  context_dim : {CONTEXT_DIM}")
print(f"  num_skills  : {NUM_SKILLS}")
print(f"  train samples: {N_TRAIN}, test: {N_TEST}")
print(f"  発火率 (train): {train_tgt.mean():.3f}")

# ─── インポート確認 ───
print("\n[1] NeatGate インポート確認 ... ", end="", flush=True)
try:
    from src.interfaces.neat_gate import (
        NeatGate,
        BinaryGateProblem,
        NeatAugmentedReward,
        compute_gate_bcs_descriptor,
    )
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── BinaryGateProblem ───
print("[2] BinaryGateProblem 構築 ... ", end="", flush=True)
try:
    prob = BinaryGateProblem(contexts=train_ctx, targets=train_tgt)
    assert prob.input_shape  == (CONTEXT_DIM,)
    assert prob.output_shape == (NUM_SKILLS,)
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── NeatGate 構築 ───
print("[3] NeatGate 構築 ... ", end="", flush=True)
try:
    gate = NeatGate(
        context_dim=CONTEXT_DIM,
        num_skills=NUM_SKILLS,
        threshold=0.5,
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
print("[4] NeatGate.fit() 進化 (5世代) ...")
try:
    gate.fit(contexts=train_ctx, targets=train_tgt)
    print("    ✅ fit 完了")
except Exception as e:
    print(f"    ❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── activate() ─── 単一
print("[5] activate() 単一入力 ... ", end="", flush=True)
try:
    mask = gate.activate(test_ctx[0])
    assert mask.shape == (NUM_SKILLS,)
    assert mask.dtype == bool
    print(f"✅  mask={mask}  (K={NUM_SKILLS})")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── activate() ─── バッチ
print("[6] activate() バッチ入力 ... ", end="", flush=True)
try:
    masks = gate.activate(test_ctx)
    assert masks.shape == (N_TEST, NUM_SKILLS)
    print(f"✅  shape={masks.shape}, 発火率={masks.mean():.3f}")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── probs() ───
print("[7] probs() 確認 ... ", end="", flush=True)
try:
    p = gate.probs(test_ctx[0])
    assert p.shape == (NUM_SKILLS,)
    assert np.all((p >= 0) & (p <= 1))
    print(f"✅  probs={np.round(p, 3)}")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── accuracy() ───
print("[8] accuracy() 確認 ... ", end="", flush=True)
try:
    acc_train = gate.accuracy(train_ctx, train_tgt)
    acc_test  = gate.accuracy(test_ctx,  test_tgt)
    assert 0.0 <= acc_train <= 1.0
    print(f"✅  train_acc={acc_train:.3f}, test_acc={acc_test:.3f}")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── BCS descriptor ───
print("[9] bcs_descriptor() 確認 ... ", end="", flush=True)
try:
    desc = gate.bcs_descriptor(test_ctx)
    assert desc.shape == (2,)
    assert np.all((desc >= 0) & (desc <= 1)), f"範囲外: {desc}"
    print(f"✅  [firing_rate={desc[0]:.3f}, specificity={desc[1]:.3f}]")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── NeatAugmentedReward ───
print("[10] NeatAugmentedReward 確認 ... ", end="", flush=True)
try:
    dummy_base_reward = lambda ctx, act: float(np.dot(ctx[:4], act[:4]))
    augmented = NeatAugmentedReward(
        neat_gate=gate,
        base_reward=dummy_base_reward,
        gate_weight=0.3,
    )
    ctx_sample = test_ctx[0]
    act_sample = np.ones(CONTEXT_DIM, dtype=np.float32)
    reward = augmented(ctx_sample, act_sample)
    assert isinstance(reward, float)
    print(f"✅  reward={reward:.4f}")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── サマリー ───
print("\n" + "=" * 60)
print("Phase 2 動作確認 サマリー")
print("=" * 60)
print("✅ NeatGate (B型 I/F) 基本動作確認完了")
print(f"   context_dim : {CONTEXT_DIM}")
print(f"   num_skills  : {NUM_SKILLS}")
print(f"   test acc    : {acc_test:.3f}")
print(f"   BCS 記述子  : [発火率={desc[0]:.3f}, 特異性={desc[1]:.3f}]")
print()
print("次のステップ:")
print("  - Phase 3: C型 NeatSlotFiller 実装")
print("  - MAP-Elites アーカイブ (QDax) 統合")
print("  - GRPO 報酬関数との結合テスト")
