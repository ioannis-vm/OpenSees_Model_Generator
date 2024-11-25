"""Recorder objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from osmg.core.uid_object import UIDObject


@dataclass
class Recorder(UIDObject):
    """Recorder base class."""

    # TODO(JVM): figure out binary format.

    def ops_args(self) -> list[object]:  # noqa: PLR6301
        """Obtain the OpenSees arguments."""
        msg = 'Child classes should implement this.'
        raise NotImplementedError(msg)


@dataclass
class NodeRecorder(Recorder):
    """
    OpenSees Node recorder.

    The Node recorder type records the response of a number of nodes
    at every converged step.
    https://opensees.berkeley.edu/wiki/index.php?title=Node_Recorder
    """

    recorder_type: Literal['Node', 'EnvelopeNode']
    nodes: tuple[int, ...]
    dofs: tuple[int, ...]
    response_type: Literal[
        'disp', 'vel', 'accel', 'incrDisp', 'eigen', 'reaction', 'rayleighForces'
    ]
    file_name: str
    number_of_significant_digits: int
    output_time: bool
    delta_t: float | None = field(default=None)
    time_series_tag: int | None = field(default=None)
    mode_number: int | None = field(default=None)

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        output: list[object] = [
            self.recorder_type,
            '-file',
            self.file_name,
            '-precision',
            self.number_of_significant_digits,
        ]
        if self.time_series_tag:
            output.extend(
                [
                    '-timeSeries',
                    self.time_series_tag,
                ]
            )
        if self.output_time:
            output.extend(['-time'])
        if self.delta_t:
            output.extend(['-dT', self.delta_t])
        output.extend(['-node', *self.nodes])
        output.extend(['-dof', *self.dofs])
        output.append(self.response_type)
        if self.response_type == 'eigen':
            output.append(self.mode_number)
        return output


@dataclass
class DriftRecorder(Recorder):
    """
    OpenSees Drift recorder.

    The Drift type records the displacement drift between two
    nodes. The drift is taken as the ratio between the prescribed
    relative displacement and the specified distance between the
    nodes.
    https://opensees.berkeley.edu/wiki/index.php?title=Node_Recorder
    """

    nodes_i: tuple[int, ...]
    nodes_j: tuple[int, ...]
    perpendicular_directions: tuple[int, ...]
    dofs: tuple[int, ...]
    file_name: str
    number_of_significant_digits: int
    output_time: bool
    # delta_t: float | None = field(default=None)

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        output: list[object] = [
            'Drift',
            '-file',
            self.file_name,
            '-precision',
            self.number_of_significant_digits,
        ]
        if self.output_time:
            output.extend(['-time'])
        # if self.delta_t:
        #     output.extend(['-dT', self.delta_t])
        output.extend(['-iNode', *self.nodes_i])
        output.extend(['-jNode', *self.nodes_j])
        output.extend(['-dof', *self.dofs])
        output.extend(['-perpDirn', *self.perpendicular_directions])
        # TODO(JVM): see if delta_t is supported.
        return output


@dataclass
class ElementRecorder(Recorder):
    """
    OpenSees Element recorder.

    The Element recorder type records the response of a number of
    elements at every converged step. The response recorded is
    element-dependent and also depends on the arguments which are
    passed to the setResponse() element method.
    https://opensees.berkeley.edu/wiki/index.php?title=Node_Recorder
    """

    recorder_type: Literal['Element', 'EnvelopeElement']
    elements: tuple[int, ...]
    element_arguments: tuple[str, ...]
    file_name: str
    number_of_significant_digits: int
    output_time: bool
    delta_t: float | None = field(default=None)

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        output: list[object] = [
            self.recorder_type,
            '-file',
            self.file_name,
            '-precision',
            self.number_of_significant_digits,
        ]
        if self.output_time:
            output.extend(['-time'])
        if self.delta_t:
            output.extend(['-dT', self.delta_t])
        output.extend(['-ele', *self.elements])
        output.extend([*self.element_arguments])
        return output
