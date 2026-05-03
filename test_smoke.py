from hackathon_judge_ft.data import load_frontier, validate, split
from hackathon_judge_ft.evaluate import parse_verdict

ds = load_frontier("madhacks")
validate(ds)
train_ds, test_ds, train_pairs, test_pairs = split(ds)
print(f"train={len(train_ds)} test={len(test_ds)}")
print(f"sample roles: {[m['role'] for m in train_ds[0]['messages']]}")

assert parse_verdict("<think>...</think>\nVERDICT: A") == "A"
assert parse_verdict("VERDICT: B") == "B"
assert parse_verdict("VERDICT: TIE") == "tie"
assert parse_verdict("VERDICT: A\nVERDICT: B") == "B"
assert parse_verdict("nothing") is None
print("all assertions passed")
