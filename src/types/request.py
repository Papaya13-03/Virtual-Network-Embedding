from typing import TypedDict
from src.types.virtual import VirtualNetwork


class VirtualRequest(TypedDict):
    vnetwork: VirtualNetwork
    arrival_time: float
    lifetime: float