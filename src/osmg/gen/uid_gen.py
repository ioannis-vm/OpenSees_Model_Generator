"""
Model Generator for OpenSees ~ uid generator
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

from dataclasses import dataclass
from itertools import count


@dataclass
class UIDGenerator:
    """
    Generates unique identifiers, uids, for various things.
    """

    def new(self, thing: str):
        """
        Provide a uid for a new node
        """
        if hasattr(self, thing):
            res = next(getattr(self, thing))
        else:
            setattr(self, thing, count(0))
            res = next(getattr(self, thing))
        return res
