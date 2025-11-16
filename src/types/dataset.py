from typing import List, TypedDict
from src.types.request import VirtualRequest
from src.types.substrate import SubstrateNetwork


class Dataset(TypedDict):
    substrate_network: SubstrateNetwork
    virtual_requests: List[VirtualRequest]