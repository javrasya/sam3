# DISCERN FORK LOCAL ADDITION -- not part of upstream SAM3 (see Discern ADR 0002).
"""Object Match: which detection inside a Zoom Window is this Object.

The bug these cover: three dummies 40 to 85 pixels apart, framed by 240 pixel
windows, so every window held every Object. Selecting on score picked whichever
Object the model preferred, and two Objects merged onto one within two frames.
"""

from sam3.zoom_anchor import box_iou, choose_object_match

# obj 1, obj 2 and obj 3 as they sit inside one 240px window, in crop space.
OBJ1 = (200, 60, 240, 180)
OBJ2 = (115, 60, 155, 180)
OBJ3 = (75, 60, 115, 180)
ALL_THREE = [OBJ1, OBJ2, OBJ3]


class TestObjectMatch:
    def test_an_object_is_matched_to_itself_not_to_the_highest_scoring_neighbour(self):
        # obj 2's own previous box, barely moved since the previous frame.
        assert choose_object_match(ALL_THREE, (118, 62, 158, 182)) == 1

    def test_each_object_in_a_shared_window_matches_its_own(self):
        picked = [
            choose_object_match(ALL_THREE, own)
            for own in [(203, 62, 243, 182), (118, 62, 158, 182), (78, 62, 118, 182)]
        ]
        assert picked == [0, 1, 2]

    def test_an_object_that_is_not_in_its_window_is_not_matched(self):
        # Nothing overlaps where it was: absent beats a confident wrong answer.
        assert choose_object_match(ALL_THREE, (0, 0, 20, 20)) is None

    def test_an_object_with_no_previous_box_is_left_to_the_caller(self):
        assert choose_object_match(ALL_THREE, None) is None

    def test_no_detections_matches_nothing(self):
        assert choose_object_match([], OBJ1) is None

    def test_overlap_beats_a_larger_box_that_merely_contains_the_object(self):
        # A detection swallowing the whole window overlaps everything; the
        # Object's own tight box must still win on IoU.
        swallowing = (0, 0, 240, 240)
        assert choose_object_match([swallowing, OBJ2], (118, 62, 158, 182)) == 1


class TestBoxIou:
    def test_disjoint_boxes_do_not_overlap(self):
        assert box_iou(OBJ1, OBJ3) == 0.0

    def test_a_box_fully_overlaps_itself(self):
        assert box_iou(OBJ2, OBJ2) == 1.0

    def test_touching_edges_are_not_an_overlap(self):
        assert box_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0
