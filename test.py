from framework.VReID import VReID

reid = VReID()

opt_alpha = 0.22
best_temporal_model = "GRU"

print(f"[RESULT] Baseline ReID Performance - Visual Component Only")
cmc, m = reid.evaluate(0, best_temporal_model, dynamic=False, gt_vlds=False, temporal_window=0, temporal_sigma=0, logging=False, return_assignment=False)
print(f"mAP : {m:.3f}")
print(f"CMC-1 : {cmc[0]:.3f}")
print(f"CMC-5 : {cmc[4]:.3f}")

print(f"[RESULT] Proposed ReID Framework Performance with optimal alpha = {opt_alpha}")
cmc, m = reid.evaluate(opt_alpha, best_temporal_model, dynamic=True, gt_vlds=False, temporal_window=0, temporal_sigma=0, logging=False, return_assignment=False)
print(f"mAP : {m:.3f}")
print(f"CMC-1 : {cmc[0]:.3f}")
print(f"CMC-5 : {cmc[4]:.3f}")