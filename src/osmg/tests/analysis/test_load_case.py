"""Unit tests for load case and related functions."""

import numpy as np
import pandas as pd
import pytest

from osmg.analysis.load_case import (
    combine,
    combine_single,
    ensure_minmax_level_exists_or_add,
)


class TestAddMinMaxLevel:
    """Tests the `ensure_minmax_level_exists_or_add` function."""

    def test_ensure_minmax_level_exists_or_add_single_level(self) -> None:
        """Test adding 'min/max' level to a single-level MultiIndex DataFrame."""
        test_df = pd.DataFrame(
            [[1, 2], [3, 4]],
            columns=pd.MultiIndex.from_tuples(
                [('a', 'x'), ('a', 'y')], names=['level_1', 'level_2']
            ),
        )
        result = ensure_minmax_level_exists_or_add(test_df)

        expected_columns = pd.MultiIndex.from_tuples(
            [
                ('a', 'x', 'max'),
                ('a', 'x', 'min'),
                ('a', 'y', 'max'),
                ('a', 'y', 'min'),
            ],
            names=['level_1', 'level_2', 'min/max'],
        )
        assert result.columns.equals(expected_columns)
        assert np.allclose(result.iloc[:, :2].to_numpy(), test_df.to_numpy())

    def test_ensure_minmax_level_exists_or_add_existing_level(self) -> None:
        """Does not alter a DataFrame with an existing 'min/max' level."""
        has_minmax = pd.DataFrame(
            [[1, 2], [3, 4]],
            columns=pd.MultiIndex.from_tuples(
                [('a', 'x', 'max'), ('a', 'y', 'min')],
                names=['level_1', 'level_2', 'min/max'],
            ),
        )
        result = ensure_minmax_level_exists_or_add(has_minmax)
        pd.testing.assert_frame_equal(result, has_minmax)


class TestCombineSingle:
    """Tests the `combine_single` function."""

    def setup_method(self) -> None:
        """Set up common test data."""
        self.df1 = pd.DataFrame(
            [[1, 2], [3, 4]],
            columns=pd.MultiIndex.from_tuples(
                [('a', 'x'), ('a', 'y')], names=['level_1', 'level_2']
            ),
        )
        self.df2 = pd.DataFrame(
            [[5, 6], [7, 8]],
            columns=pd.MultiIndex.from_tuples(
                [('a', 'x'), ('a', 'y')], names=['level_1', 'level_2']
            ),
        )

    def test_combine_add(self) -> None:
        """Test the 'add' action in combine."""
        result = combine_single(self.df1, self.df2, 'add')
        expected = pd.DataFrame(
            [[6, 8], [10, 12]],
            columns=pd.MultiIndex.from_tuples(
                [('a', 'x'), ('a', 'y')],
                names=['level_1', 'level_2'],
            ),
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_combine_envelope(self) -> None:
        """Test the 'envelope' action in combine."""
        result = combine_single(self.df1, self.df2, 'envelope')
        expected = pd.DataFrame(
            [[5, 5, 2, 2], [7, 7, 4, 4]],
            columns=pd.MultiIndex.from_tuples(
                [
                    ('a', 'x', 'max'),
                    ('a', 'x', 'min'),
                    ('a', 'y', 'max'),
                    ('a', 'y', 'min'),
                ],
                names=['level_1', 'level_2', 'min/max'],
            ),
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_combine_invalid_action(self) -> None:
        """An invalid action raises a ValueError."""
        with pytest.raises(
            ValueError, match='Action must be one of `add` or `envelope`.'
        ):
            combine_single(self.df1, self.df2, 'invalid')  # type: ignore

    def test_combine_different_columns(self) -> None:
        """Combining DataFrames with different columns raises an error."""
        df3 = pd.DataFrame(
            [[9], [10]],  # Adjust to match the number of columns
            columns=pd.MultiIndex.from_tuples(
                [('b', 'z')], names=['level_1', 'level_2']
            ),
        )
        with pytest.raises(
            ValueError, match='Cannot align DataFrames with different columns'
        ):
            combine_single(self.df1, df3, 'add')


class TestCombine:
    """Tests the `combine` function."""

    def setup_method(self) -> None:
        """Set up common test data."""
        self.df1 = pd.DataFrame(
            [[1, 2], [3, 4]],
            columns=pd.MultiIndex.from_tuples(
                [('a', 'x'), ('a', 'y')], names=['level_1', 'level_2']
            ),
        )
        self.df2 = pd.DataFrame(
            [[5, 6], [7, 8]],
            columns=pd.MultiIndex.from_tuples(
                [('a', 'x'), ('a', 'y')], names=['level_1', 'level_2']
            ),
        )
        self.df3 = pd.DataFrame(
            [[9, 10], [11, 12]],
            columns=pd.MultiIndex.from_tuples(
                [('a', 'x'), ('a', 'y')], names=['level_1', 'level_2']
            ),
        )

    def test_combine_add(self) -> None:
        """Test combining  DataFrames with the 'add' action."""
        result = combine([self.df1, self.df2, self.df3], 'add')
        expected = pd.DataFrame(
            [[15, 18], [21, 24]],
            columns=pd.MultiIndex.from_tuples(
                [('a', 'x'), ('a', 'y')],
                names=['level_1', 'level_2'],
            ),
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_combine_envelope(self) -> None:
        """Test combining  DataFrames with the 'envelope' action."""
        result = combine([self.df1, self.df2, self.df3], 'envelope')
        expected = pd.DataFrame(
            [[9, 9, 5, 2], [11, 11, 7, 4]],
            columns=pd.MultiIndex.from_tuples(
                [
                    ('a', 'x', 'max'),
                    ('a', 'x', 'min'),
                    ('a', 'y', 'max'),
                    ('a', 'y', 'min'),
                ],
                names=['level_1', 'level_2', 'min/max'],
            ),
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_combine_insufficient_dataframes(self) -> None:
        """Test that combining fewer than two DataFrames raises a ValueError."""
        with pytest.raises(
            ValueError, match='At least two DataFrames are required to combine.'
        ):
            combine([self.df1], 'add')

    def test_combine_different_columns(self) -> None:
        """Test combining  DataFrames with mismatched columns raises an error."""
        df4 = pd.DataFrame(
            [[1], [2]],
            columns=pd.MultiIndex.from_tuples(
                [('b', 'z')], names=['level_1', 'level_2']
            ),
        )
        with pytest.raises(
            ValueError, match='Cannot align DataFrames with different columns'
        ):
            combine([self.df1, self.df2, df4], 'add')

    def test_combine_invalid_action(self) -> None:
        """Test that an invalid action raises a ValueError."""
        with pytest.raises(
            ValueError, match='Action must be one of `add` or `envelope`.'
        ):
            combine([self.df1, self.df2], 'invalid')  # type: ignore
