"""
Shared helpers for RTB command-line tools.
"""

from __future__ import annotations

import argparse
import textwrap


def link(uri: str, label: str | None = None) -> str:
    """Return a terminal hyperlink escape sequence.

    :param uri: target URL
    :param label: display label, defaults to the URI itself
    :return: OSC-8 hyperlink string
    """
    # https://stackoverflow.com/questions/40419276/python-how-to-print-text-to-console-as-hyperlink
    if label is None:
        label = uri
    escape_mask = "\033]8;{};{}\033\\{}\033]8;;\033\\"
    return escape_mask.format("", uri, label)


class CustomHelpFormatter(argparse.HelpFormatter):
    """Argparse formatter that wraps at word boundaries on explicit newlines."""

    def _split_lines(self, text: str, width: int) -> list[str]:
        lines = text.splitlines()
        wrapped: list[str] = []
        for line in lines:
            wrapped.extend(
                textwrap.wrap(
                    line, width, break_long_words=False, break_on_hyphens=False
                )
            )
        return wrapped


class CustomDefaultsHelpFormatter(
    CustomHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """Custom formatter that also appends default values in help output."""


class LineWrapRawTextHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Argparse formatter that reflowes whitespace and wraps at 80 columns."""

    def _split_lines(self, text: str, width: int) -> list[str]:
        text = self._whitespace_matcher.sub(" ", text).strip()
        return textwrap.wrap(text, 80)


class LineWrapRawTextDefaultsHelpFormatter(
    LineWrapRawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """Line-wrapped formatter that also appends default values."""


RTB_URL = "https://github.com/petercorke/robotics-toolbox-python"
RTB_LINK = link(RTB_URL, "Robotics Toolbox for Python")
