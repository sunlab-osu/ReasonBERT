# utility fuction from tapas to convert numeric 
# Below: all functions from number_utils.py as well as 2 functions (namely get_all_spans and normalize_for_match)
# from text_utils.py of the original implementation. URL's:
# - https://github.com/google-research/tapas/blob/master/tapas/utils/number_utils.py
# - https://github.com/google-research/tapas/blob/master/tapas/utils/text_utils.py
import collections
import datetime
import enum
import itertools
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Optional, Text, Tuple, Union

import numpy as np

# Constants for parsing date expressions.
# Masks that specify (by a bool) which of (year, month, day) will be populated.
_DateMask = collections.namedtuple("_DateMask", ["year", "month", "day"])

_YEAR = _DateMask(True, False, False)
_YEAR_MONTH = _DateMask(True, True, False)
_YEAR_MONTH_DAY = _DateMask(True, True, True)
_MONTH = _DateMask(False, True, False)
_MONTH_DAY = _DateMask(False, True, True)

# Pairs of patterns to pass to 'datetime.strptime' and masks specifying which
# fields will be set by the corresponding pattern.
_DATE_PATTERNS = (
    ("%B", _MONTH),
    ("%Y", _YEAR),
    ("%Ys", _YEAR),
    ("%b %Y", _YEAR_MONTH),
    ("%B %Y", _YEAR_MONTH),
    ("%B %d", _MONTH_DAY),
    ("%b %d", _MONTH_DAY),
    ("%d %b", _MONTH_DAY),
    ("%d %B", _MONTH_DAY),
    ("%B %d, %Y", _YEAR_MONTH_DAY),
    ("%d %B %Y", _YEAR_MONTH_DAY),
    ("%m-%d-%Y", _YEAR_MONTH_DAY),
    ("%Y-%m-%d", _YEAR_MONTH_DAY),
    ("%Y-%m", _YEAR_MONTH),
    ("%B %Y", _YEAR_MONTH),
    ("%d %b %Y", _YEAR_MONTH_DAY),
    ("%Y-%m-%d", _YEAR_MONTH_DAY),
    ("%b %d, %Y", _YEAR_MONTH_DAY),
    ("%d.%m.%Y", _YEAR_MONTH_DAY),
    ("%A, %b %d", _MONTH_DAY),
    ("%A, %B %d", _MONTH_DAY),
)

# This mapping is used to convert date patterns to regex patterns.
_FIELD_TO_REGEX = (
    ("%A", r"\w+"),  # Weekday as locale’s full name.
    ("%B", r"\w+"),  # Month as locale’s full name.
    ("%Y", r"\d{4}"),  # Year with century as a decimal number.
    ("%b", r"\w{3}"),  # Month as locale’s abbreviated name.
    ("%d", r"\d{1,2}"),  # Day of the month as a zero-padded decimal number.
    ("%m", r"\d{1,2}"),  # Month as a zero-padded decimal number.
)


def _process_date_pattern(dp):
    """Compute a regex for each date pattern to use as a prefilter."""
    pattern, mask = dp
    regex = pattern
    regex = regex.replace(".", re.escape("."))
    regex = regex.replace("-", re.escape("-"))
    regex = regex.replace(" ", r"\s+")
    for field, field_regex in _FIELD_TO_REGEX:
        regex = regex.replace(field, field_regex)
    # Make sure we didn't miss any of the fields.
    assert "%" not in regex, regex
    return pattern, mask, re.compile("^" + regex + "$")


def _process_date_patterns():
    return tuple(_process_date_pattern(dp) for dp in _DATE_PATTERNS)


_PROCESSED_DATE_PATTERNS = _process_date_patterns()

_MAX_DATE_NGRAM_SIZE = 5

# Following DynSp:
# https://github.com/Microsoft/DynSP/blob/master/util.py#L414.
_NUMBER_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
]

_ORDINAL_WORDS = [
    "zeroth",
    "first",
    "second",
    "third",
    "fourth",
    "fith",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
]

_ORDINAL_SUFFIXES = ["st", "nd", "rd", "th"]

# _NUMBER_PATTERN = re.compile(r"((^|\s)[+-])?((\.\d+)|(\d+(,\d\d\d)*(\.\d*)?))")
_NUMBER_PATTERN = re.compile(r"\b([+-])?((\.\d+)|(\d+(,\d\d\d)*(\.\d*)?))\b")

# Following DynSp:
# https://github.com/Microsoft/DynSP/blob/master/util.py#L293.
_MIN_YEAR = 1700
_MAX_YEAR = 2100

_INF = float("INF")


class Date:
    def __init__(self, year=None, month=None, day=None):
        self.year = year
        self.month = month
        self.day = day

class NumericValue:
    def __init__(self, float_value=None, date=None):
        self.float_value = float_value
        self.date = date


@dataclass
class NumericValueSpan:
    def __init__(self, begin_index=None, end_index=None, values=None):
        self.begin_index = begin_index
        self.end_index = end_index
        self.values = values


def _get_numeric_value_from_date(date, mask):
    """Converts date (datetime Python object) to a NumericValue object with a Date object value."""
    if date.year < _MIN_YEAR or date.year > _MAX_YEAR:
        raise ValueError("Invalid year: %d" % date.year)

    new_date = Date()
    if mask.year:
        new_date.year = date.year
    if mask.month:
        new_date.month = date.month
    if mask.day:
        new_date.day = date.day
    return NumericValue(date=new_date)


def _get_span_length_key(span):
    """Sorts span by decreasing length first and incresing first index second."""
    return span[1] - span[0], -span[0]


def _get_numeric_value_from_float(value):
    """Converts float (Python) to a NumericValue object with a float value."""
    return NumericValue(float_value=value)


# Doesn't parse ordinal expressions such as '18th of february 1655'.
def _parse_date(text):
    """Attempts to format a text as a standard date string (yyyy-mm-dd)."""
    text = re.sub(r"Sept\b", "Sep", text)
    for in_pattern, mask, regex in _PROCESSED_DATE_PATTERNS:
        if not regex.match(text):
            continue
        try:
            date = datetime.datetime.strptime(text, in_pattern).date()
        except ValueError:
            continue
        try:
            return _get_numeric_value_from_date(date, mask)
        except ValueError:
            continue
    return None


def _parse_number(text):
    """Parses simple cardinal and ordinals numbers."""
    for suffix in _ORDINAL_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    text = text.replace(",", "")
    try:
        value = float(text)
    except ValueError:
        return None
    if math.isnan(value):
        return None
    if value == _INF:
        return None
    return value


def get_all_spans(text, max_ngram_length):
    """
    Split a text into all possible ngrams up to 'max_ngram_length'. Split points are white space and punctuation.

    Args:
      text: Text to split.
      max_ngram_length: maximal ngram length.
    Yields:
      Spans, tuples of begin-end index.
    """
    start_indexes = []
    for index, char in enumerate(text):
        if not char.isalnum():
            continue
        if index == 0 or not text[index - 1].isalnum():
            start_indexes.append(index)
        if index + 1 == len(text) or not text[index + 1].isalnum():
            for start_index in start_indexes[-max_ngram_length:]:
                yield start_index, index + 1


def normalize_for_match(text):
    return " ".join(text.lower().split())


def format_text(text):
    """Lowercases and strips punctuation."""
    text = text.lower().strip()
    if text == "n/a" or text == "?" or text == "nan":
        text = EMPTY_TEXT

    text = re.sub(r"[^\w\d]+", " ", text).replace("_", " ")
    text = " ".join(text.split())
    text = text.strip()
    if text:
        return text
    return EMPTY_TEXT


def parse_text(text):
    """
    Extracts longest number and date spans.

    Args:
      text: text to annotate

    Returns:
      List of longest numeric value spans.
    """
    span_dict = collections.defaultdict(list)
    for match in _NUMBER_PATTERN.finditer(text):
        span_text = text[match.start() : match.end()]
        number = _parse_number(span_text)
        if number is not None:
            span_dict[match.span()].append(_get_numeric_value_from_float(number))

    for begin_index, end_index in get_all_spans(text, max_ngram_length=1):
        if (begin_index, end_index) in span_dict:
            continue
        span_text = text[begin_index:end_index]

        number = _parse_number(span_text)
        if number is not None:
            span_dict[begin_index, end_index].append(_get_numeric_value_from_float(number))
        for number, word in enumerate(_NUMBER_WORDS):
            if span_text == word:
                span_dict[begin_index, end_index].append(_get_numeric_value_from_float(float(number)))
                break
        for number, word in enumerate(_ORDINAL_WORDS):
            if span_text == word:
                span_dict[begin_index, end_index].append(_get_numeric_value_from_float(float(number)))
                break

    for begin_index, end_index in get_all_spans(text, max_ngram_length=_MAX_DATE_NGRAM_SIZE):
        span_text = text[begin_index:end_index]
        date = _parse_date(span_text)
        if date is not None:
            span_dict[begin_index, end_index].append(date)

    spans = sorted(span_dict.items(), key=lambda span_value: _get_span_length_key(span_value[0]), reverse=True)
    selected_spans = []
    for span, value in spans:
        for selected_span, _ in selected_spans:
            if selected_span[0] <= span[0] and span[1] <= selected_span[1]:
                break
        else:
            selected_spans.append((span, value))

    selected_spans.sort(key=lambda span_value: span_value[0][0])

    numeric_value_spans = []
    for span, values in selected_spans:
        numeric_value_spans.append(NumericValueSpan(begin_index=span[0], end_index=span[1], values=values))
    return numeric_value_spans


# Below: all functions from number_annotation_utils.py and 2 functions (namely filter_invalid_unicode
# and filter_invalid_unicode_from_table) from text_utils.py of the original implementation. URL's:
# - https://github.com/google-research/tapas/blob/master/tapas/utils/number_annotation_utils.py
# - https://github.com/google-research/tapas/blob/master/tapas/utils/text_utils.py


_PrimitiveNumericValue = Union[float, Tuple[Optional[float], Optional[float], Optional[float]]]
_SortKeyFn = Callable[[NumericValue], Tuple[float, Ellipsis]]

_DATE_TUPLE_SIZE = 3

EMPTY_TEXT = "EMPTY"

NUMBER_TYPE = "number"
DATE_TYPE = "date"


def _get_value_type(numeric_value):
    if numeric_value.float_value is not None:
        return NUMBER_TYPE
    elif numeric_value.date is not None:
        return DATE_TYPE
    raise ValueError("Unknown type: %s" % numeric_value)


def _get_value_as_primitive_value(numeric_value):
    """Maps a NumericValue proto to a float or tuple of float."""
    if numeric_value.float_value is not None:
        return numeric_value.float_value
    if numeric_value.date is not None:
        date = numeric_value.date
        value_tuple = [None, None, None]
        # All dates fields are cased to float to produce a simple primitive value.
        if date.year is not None:
            value_tuple[0] = float(date.year)
        if date.month is not None:
            value_tuple[1] = float(date.month)
        if date.day is not None:
            value_tuple[2] = float(date.day)
        return tuple(value_tuple)
    raise ValueError("Unknown type: %s" % numeric_value)


def _get_all_types(numeric_values):
    return {_get_value_type(value) for value in numeric_values}


def get_numeric_sort_key_fn(numeric_values):
    """
    Creates a function that can be used as a sort key or to compare the values. Maps to primitive types and finds the
    biggest common subset. Consider the values "05/05/2010" and "August 2007". With the corresponding primitive values
    (2010.,5.,5.) and (2007.,8., None). These values can be compared by year and date so we map to the sequence (2010.,
    5.), (2007., 8.). If we added a third value "2006" with primitive value (2006., None, None), we could only compare
    by the year so we would map to (2010.,), (2007.,) and (2006.,).

    Args:
     numeric_values: Values to compare

    Returns:
     A function that can be used as a sort key function (mapping numeric values to a comparable tuple)

    Raises:
      ValueError if values don't have a common type or are not comparable.
    """
    value_types = _get_all_types(numeric_values)
    if len(value_types) != 1:
        raise ValueError("No common value type in %s" % numeric_values)

    value_type = next(iter(value_types))
    if value_type == NUMBER_TYPE:
        # Primitive values are simple floats, nothing to do here.
        return _get_value_as_primitive_value

    # The type can only be Date at this point which means the primitive type
    # is a float triple.
    valid_indexes = set(range(_DATE_TUPLE_SIZE))

    for numeric_value in numeric_values:
        value = _get_value_as_primitive_value(numeric_value)
        assert isinstance(value, tuple)
        for tuple_index, inner_value in enumerate(value):
            if inner_value is None:
                valid_indexes.discard(tuple_index)

    if not valid_indexes:
        raise ValueError("No common value in %s" % numeric_values)

    def _sort_key_fn(numeric_value):
        value = _get_value_as_primitive_value(numeric_value)
        return tuple(value[index] for index in valid_indexes)

    return _sort_key_fn


def _consolidate_numeric_values(row_index_to_values, min_consolidation_fraction):
    """
    Finds the most common numeric values in a column and returns them

    Args:
        row_index_to_values:
            For each row index all the values in that cell.
        min_consolidation_fraction:
            Fraction of cells that need to have consolidated value.
        debug_info:
            Additional information only used for logging

    Returns:
        For each row index the first value that matches the most common value. Rows that don't have a matching value
        are dropped. Empty list if values can't be consolidated.
    """
    type_counts = collections.Counter()
    for numeric_value_spans in row_index_to_values.values():
        type_counts.update(_get_all_types(itertools.chain(*(span.values for span in numeric_value_spans))))
    if not type_counts:
        return {}
    max_count = max(type_counts.values())
    if max_count < len(row_index_to_values) * min_consolidation_fraction:
        # logging.log_every_n(logging.INFO, 'Can\'t consolidate types: %s %s %d', 100,
        #                     debug_info, row_index_to_values, max_count)
        return {}

    valid_types = set()
    for value_type, count in type_counts.items():
        if count == max_count:
            valid_types.add(value_type)
    if len(valid_types) > 1:
        assert DATE_TYPE in valid_types
        max_type = DATE_TYPE
    else:
        max_type = next(iter(valid_types))

    new_row_index_to_value = {}
    for index, value_spans in row_index_to_values.items():
        # Extract the first matching value.
        found = False
        for span in value_spans:
            for value in span.values:
                if _get_value_type(value) == max_type:
                    new_row_index_to_value[index] = [[span.begin_index, span.end_index], value]
                    found = True
                    break
            if found:
                break

    return new_row_index_to_value


def _get_numeric_values(text):
    """Parses text and returns numeric values."""
    numeric_spans = parse_text(text)
    return itertools.chain(*(span.values for span in numeric_spans))


def _get_column_values(table, col_index):
    """
    Parses text in column and returns a dict mapping row_index to values. This is the _get_column_values function from
    number_annotation_utils.py of the original implementation

    Args:
      table: Pandas dataframe
      col_index: integer, indicating the index of the column to get the numeric values of
    """
    index_to_values = {}
    for row_index, row in table.iterrows():
        text = normalize_for_match(row[col_index].text)
        index_to_values[row_index] = list(_get_numeric_values(text))
    return index_to_values


def get_numeric_relation(value, other_value, sort_key_fn):
    """Compares two values and returns their relation or None."""
    value = sort_key_fn(value)
    other_value = sort_key_fn(other_value)
    if value == other_value:
        return Relation.EQ
    if value < other_value:
        return Relation.LT
    if value > other_value:
        return Relation.GT
    return None


def add_numeric_values_to_question(question):
    """Adds numeric value spans to a question."""
    original_text = question
    question = normalize_for_match(question)
    numeric_spans = parse_text(question)
    return Question(original_text=original_text, text=question, numeric_spans=numeric_spans)


def filter_invalid_unicode(text):
    """Return an empty string and True if 'text' is in invalid unicode."""
    return ("", True) if isinstance(text, bytes) else (text, False)


def filter_invalid_unicode_from_table(table):
    """
    Removes invalid unicode from table. Checks whether a table cell text contains an invalid unicode encoding. If yes,
    reset the table cell text to an empty str and log a warning for each invalid cell

    Args:
        table: table to clean.
    """
    # to do: add table id support
    if not hasattr(table, "table_id"):
        table.table_id = 0

    for row_index, row in table.iterrows():
        for col_index, cell in enumerate(row):
            cell, is_invalid = filter_invalid_unicode(cell)
            if is_invalid:
                logging.warning(
                    "Scrub an invalid table body @ table_id: %s, row_index: %d, " "col_index: %d",
                    table.table_id,
                    row_index,
                    col_index,
                )
    for col_index, column in enumerate(table.columns):
        column, is_invalid = filter_invalid_unicode(column)
        if is_invalid:
            logging.warning("Scrub an invalid table header @ table_id: %s, col_index: %d", table.table_id, col_index)


def add_numeric_table_values(table, min_consolidation_fraction=0.7, debug_info=None):
    """
    Parses text in table column-wise and adds the consolidated values. Consolidation refers to finding values with a
    common types (date or number)

    Args:
        table:
            Table to annotate.
        min_consolidation_fraction:
            Fraction of cells in a column that need to have consolidated value.
        debug_info:
            Additional information used for logging.
    """
    table = table.copy()
    # First, filter table on invalid unicode
    filter_invalid_unicode_from_table(table)

    # Second, replace cell values by Cell objects
    for row_index, row in table.iterrows():
        for col_index, cell in enumerate(row):
            table.iloc[row_index, col_index] = Cell(text=cell)

    # Third, add numeric_value attributes to these Cell objects
    for col_index, column in enumerate(table.columns):
        column_values = _consolidate_numeric_values(
            _get_column_values(table, col_index),
            min_consolidation_fraction=min_consolidation_fraction,
            debug_info=(debug_info, column),
        )

        for row_index, numeric_value in column_values.items():
            table.iloc[row_index, col_index].numeric_value = numeric_value

    return table

def _get_numeric_column_ranks(column_ids, row_ids, table):
    """Returns column ranks for all numeric columns."""

    ranks = [0] * len(column_ids)
    inv_ranks = [0] * len(column_ids)

    # original code from tf_example_utils.py of the original implementation
    if table is not None:
        for col_index in range(len(table.columns)):
            table_numeric_values = self._get_column_values(table, col_index)

            if not table_numeric_values:
                continue

            try:
                key_fn = get_numeric_sort_key_fn(table_numeric_values.values())
            except ValueError:
                continue

            table_numeric_values = {row_index: key_fn(value) for row_index, value in table_numeric_values.items()}

            table_numeric_values_inv = collections.defaultdict(list)
            for row_index, value in table_numeric_values.items():
                table_numeric_values_inv[value].append(row_index)

            unique_values = sorted(table_numeric_values_inv.keys())

            for rank, value in enumerate(unique_values):
                for row_index in table_numeric_values_inv[value]:
                    for index in self._get_cell_token_indexes(column_ids, row_ids, col_index, row_index):
                        ranks[index] = rank + 1
                        inv_ranks[index] = len(unique_values) - rank

    return ranks, inv_ranks