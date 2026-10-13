"""Checkpoints requesting a non-sequential option layout are rejected, never run as sequential."""

import pytest

from laya_apple.errors import LayaAppleError
from laya_apple.model import _check_option_layout


@pytest.mark.parametrize("cfg", [{}, {"option_layout": None}, {"option_layout": "sequential"}])
def test_sequential_layout_is_accepted(cfg):
    _check_option_layout(cfg, {})


@pytest.mark.parametrize("layout", ["parallel", "", "Sequential", 1])
def test_other_layouts_fail_closed(layout):
    with pytest.raises(LayaAppleError, match="option_layout"):
        _check_option_layout({"option_layout": layout}, {})
    with pytest.raises(LayaAppleError, match="option_layout"):
        _check_option_layout({}, {"option_layout": layout})
