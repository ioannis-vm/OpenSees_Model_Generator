"""objects that create unique IDs."""

from dataclasses import dataclass
from itertools import count

# from osmg.model_objects.node import Node


@dataclass
class UIDGenerator:
    """Generates unique identifiers (uids) for various objects."""

    def new(self, thing: object) -> int:
        """
        Generate a new uid for an object based on its category.

        Arguments:
            thing: The object for which to generate a uid.

        Returns:
            A unique identifier for an object of the given type.

        Raises:
          ValueError: If an unknown object class is specified.
        """
        object_type = thing.__class__.__name__
        valid_types = {
            'Node': 'NODE',
            'ElasticSection': 'SECTION',
            'ComponentAssembly': 'COMPONENT',
            'BeamColumnAssembly': 'COMPONENT',
            'BarAssembly': 'COMPONENT',
            'GeomTransf': 'TRANSFORMATION',
            'ElasticBeamColumn': 'ELEMENT',
            'ZeroLength': 'ELEMENT',
            'Bar': 'ELEMENT',
            'TwoNodeLink': 'ELEMENT',
            'NodeRecorder': 'RECORDER',
            'DriftRecorder': 'RECORDER',
            'ElementRecorder': 'RECORDER',
            '_TestChild': 'TESTING',
            'Elastic': 'MATERIAL',
            'ElasticPPGap': 'MATERIAL',
            'Steel4': 'MATERIAL',
            'Fatigue': 'MATERIAL',
            'IMKBilin': 'MATERIAL',
            'Pinching4': 'MATERIAL',
            'LeadRubberX': 'ELEMENT',
            'TripleFrictionPendulum': 'ELEMENT',
            'Coulomb': 'FRICTIONMODEL',
        }

        if object_type not in valid_types:
            msg = f'Unknown object class: {object_type}'
            raise ValueError(msg)

        type_category = valid_types[object_type]
        if hasattr(self, type_category):
            res = next(getattr(self, type_category))
            assert isinstance(res, int)
        else:
            setattr(self, type_category, count(0))
            res = next(getattr(self, type_category))
            assert isinstance(res, int)
        return res
