import torch

model_path = 'agents/EGPO_2000.pt'
loaded = torch.load(model_path)

print("Type:", type(loaded))
print("\nIs dict:", isinstance(loaded, dict))

if isinstance(loaded, dict):
    print("\nTop-level keys:", list(loaded.keys()))
    print("\nAll keys (first 20):")
    all_keys = list(loaded.keys())
    for i, key in enumerate(all_keys[:20]):
        print("  {}. {}".format(i+1, key))
    if len(all_keys) > 20:
        print("  ... and {} more keys".format(len(all_keys) - 20))
else:
    print("\nDirect state_dict keys (first 20):")
    all_keys = list(loaded.keys())
    for i, key in enumerate(all_keys[:20]):
        print("  {}. {}".format(i+1, key))
    if len(all_keys) > 20:
        print("  ... and {} more keys".format(len(all_keys) - 20))
        
print("\nTotal keys:", len(loaded.keys()))

# Check for critic keys
critic_keys = [k for k in loaded.keys() if 'critic' in k.lower()]
print("\nCritic keys found:", len(critic_keys))
if critic_keys:
    print("Critic keys:", critic_keys[:5])
