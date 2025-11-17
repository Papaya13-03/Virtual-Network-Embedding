import os
from src.utils.generate_virtual_network import generate_virtual_network_test
from src.utils.dataset_to_json import dataset_to_json
from src.utils.generate_dataset import generate_dataset
from src.utils.generate_substrate_network import generate_substrate_network

seed = 42
datasets_dir = "./datasets"
base_name = "small"

os.makedirs(datasets_dir, exist_ok=True)

existing_files = os.listdir(datasets_dir)
test_numbers = []
for f in existing_files:
    if f.startswith(base_name) and f.endswith(".json"):
        try:
            num = int(f[len(base_name)+1:-5])
            test_numbers.append(num)
        except ValueError:
            continue
next_test_number = max(test_numbers, default=0) + 1

dataset = generate_dataset(
    substrate_generator=generate_substrate_network,
    virtual_generator=generate_virtual_network_test,
    total_time_units=10000,
    avg_requests=500,
    avg_lifetime=1000,
    seed=seed
)

output_file = os.path.join(datasets_dir, f"{base_name}_{next_test_number}.json")
dataset_to_json(dataset, output_file)

print(f"Dataset generated and saved to '{output_file}'")
