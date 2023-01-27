import os
import sys

sys.path.insert(0, os.path.abspath('..'))

from skateboard import SegmentSkateboard


def test_skateboard_has_wheelbase_attribute():

    skateboard = SegmentSkateboard()
    assert hasattr(skateboard, 'wheelbase')
    assert skateboard.wheelbase == 0.2


if __name__ == "__name__":
    test_skateboard_has_wheelbase_attribute()
