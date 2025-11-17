from src.utils.load_dataset_from_json import load_dataset_from_json
from src.utils.visualize_dataset import visualize_dataset

dataset = load_dataset_from_json("./datasets/small_2.json")
visualize_dataset(dataset)