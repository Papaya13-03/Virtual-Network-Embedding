
# -------------------------------
# Example of usage
# -------------------------------
from src.algorithms.MC_VNE.mc_vne import MC_VNE_TimeSeries
from src.utils.load_dataset_from_json import load_dataset_from_json


dataset = load_dataset_from_json("./datasets/large_1.json")
substrate = dataset["substrate_network"]
requests = sorted(dataset["virtual_requests"], key=lambda r: r["arrival_time"])

vne = MC_VNE_TimeSeries(substrate)

for req in requests:
    # Release expired resources
    vne.release_expired(req["arrival_time"])
    success = vne.embed_request(req["vnetwork"], req["arrival_time"], req["lifetime"])
    print(f"Request at {req['arrival_time']} embedding {'succeeded' if success else 'failed'}")
