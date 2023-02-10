from skateboard import SegmentSkateboard


def test_skateboard_has_wheelbase_attribute():
    skateboard = SegmentSkateboard()
    assert hasattr(skateboard, 'wheelbase')
    assert skateboard.wheelbase == 0.2


def test_skateboard_is_segment_skateboard():
    skateboard = SegmentSkateboard(wheelbase=0.3)
    assert skateboard.wheelbase == 0.3
