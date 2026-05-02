from hackathon_judge_ft.data import load_frontier_df, validate, split
from hackathon_judge_ft.evaluate import parse_verdict

df = load_frontier_df("madhacks")
validate(df)
train_ds, test_ds, train_pairs, test_pairs = split(df)
print(f"train={len(train_ds)} test={len(test_ds)}")
print(f"sample roles: {[m['role'] for m in train_ds[0]['messages']]}")

assert parse_verdict("<think>...</think>\nVERDICT: A") == "A"
assert parse_verdict("VERDICT: B") == "B"
assert parse_verdict("nothing") is None
print("all assertions passed")
