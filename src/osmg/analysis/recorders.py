"""Recorder objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from osmg.core.uid_object import UIDObject


@dataclass
class Recorder(UIDObject):
    """Recorder base class."""

    file_name: str
    # TODO(JVM): figure out binary format.

    def __post_init__(self) -> None:
        """Post-initialization."""
        self._data = None

    def ops_args(self) -> list[object]:  # noqa: PLR6301
        """Obtain the OpenSees arguments."""
        msg = 'Child classes should implement this.'
        raise NotImplementedError(msg)

    def get_data(self) -> pd.DataFrame:  # noqa: PLR6301
        """Retrieve the data."""
        msg = 'Child classes should implement this.'
        raise NotImplementedError(msg)

    def set_data(self, data: pd.DataFrame) -> None:
        """Overwrite the data."""
        self._data = data


@dataclass
class NodeRecorder(Recorder):
    """
    OpenSees Node recorder.

    The Node recorder type records the response of a number of nodes
    at every converged step.

    Note: I haven't been able to get the `eigen` type working.

    https://opensees.berkeley.edu/wiki/index.php?title=Node_Recorder
    """

    recorder_type: Literal['Node', 'EnvelopeNode']
    nodes: tuple[int, ...]
    dofs: tuple[int, ...]
    response_type: Literal[
        'disp', 'vel', 'accel', 'incrDisp', 'reaction', 'rayleighForces'
    ]
    number_of_significant_digits: int
    output_time: bool | None
    delta_t: float | None = field(default=None)
    time_series_tag: int | None = field(default=None)

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
        return output

    def get_data(self) -> pd.DataFrame:
        """
        Retrieve the data.

        Returns:
          The data.
        """
        if self.recorder_type == 'EnvelopeNode':
            raise NotImplementedError

        if self._data is None:
            index_col = 0 if self.output_time else None
            data = pd.read_csv(
                self.file_name,
                sep=' ',
                index_col=index_col,
                header=None,
                engine='pyarrow',
            )
            data = data.astype(float)
            header_data = [(node, dof) for node in self.nodes for dof in self.dofs]
            data.columns = pd.MultiIndex.from_tuples(
                header_data, names=('node', 'dof')
            )
            data.index.name = 'time' if self.output_time else None
            self._data = data
        return self._data


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
    number_of_significant_digits: int | None
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
        ]
        if self.number_of_significant_digits:
            output.extend(
                [
                    '-precision',
                    self.number_of_significant_digits,
                ]
            )
        if self.output_time:
            output.extend(['-time'])
        if self.delta_t:
            output.extend(['-dT', self.delta_t])
        output.extend(['-ele', *self.elements])
        output.extend([*self.element_arguments])
        return output

    def get_data(self, *, update_index: bool = True) -> pd.DataFrame:
        """
        Retrieve the data.

        Returns:
          The data.
        """
        if self.recorder_type == 'EnvelopeElement':
            raise NotImplementedError

        if self._data is None:
            index_col = 0 if self.output_time else None
            data = pd.read_csv(
                self.file_name,
                sep=' ',
                index_col=index_col,
                header=None,
                engine='pyarrow',
            )
            data = data.astype(float)
            if update_index:
                # get number of dofs
                num_dof = int(data.shape[1] / len(self.elements) / 2.0)
                # construct header
                header_data = [
                    (element, station, dof)
                    for element in self.elements
                    for station in (0.00, 1.00)
                    for dof in range(1, num_dof + 1)
                ]
                data.columns = pd.MultiIndex.from_tuples(
                    header_data, names=('element', 'station', 'dof')
                )
                data.index.name = 'time' if self.output_time else None
            self._data = data
        return self._data
