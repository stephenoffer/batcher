#!/usr/bin/env python3
"""Draw `frame_chain.svg` -- chaining two poses into one, then applying it per point.

Source of truth: `docs/user-guide/analyze/robotics.md`, sections "Moving a point between
frames" and "Chaining frames". Poses are named target first, so `ego_from_lidar` takes a
lidar-frame point into the ego frame; the calibration file gives the lidar mount and the
localizer gives the vehicle in the world. `se3_compose(world_from_ego, ego_from_lidar)`
cancels the adjacent frame name and yields `world_from_lidar`. The translations are the
page's own example: ego at (100, 50, 0), lidar mount at (1.2, 0, 1.8), composed translation
(101.2, 50.0, 1.8), all with identity rotations. `se3_transform` rotates and then translates,
and `se3_inverse_transform` goes the other way. The page's advice is to compose once per
frame and apply the single result per point. The arithmetic runs in the `bc-spatial` crate.

Drawn as a chain because the naming convention is spatial: a pose is an arrow from its
source frame to its target frame, and composing two arrows that meet at a frame is the whole
idea.
"""

from __future__ import annotations

from _authoring import arrow, band, card, code, curve, label, note, svg, write

W, H = 980, 500

CY = 72
CH = 70
MID = CY + CH / 2

POINT_CODE = [
    'world_from_lidar = ("w_tx", "w_ty", "w_tz", "w_qx", "w_qy", "w_qz", "w_qw")',
    'bt.se3_transform(world_from_lidar, ("x", "y", "z"), prefix="world_")',
]

body: list[str] = [
    band(20, 20, 940, 288, "THREE FRAMES, TWO LOGGED POSES, ONE COMPOSED POSE", "blue"),
    card(40, CY, 200, CH, "lidar frame", "where the returns are"),
    card(390, CY, 200, CH, "ego frame", "the vehicle"),
    card(740, CY, 200, CH, "world frame", "does not move"),
    arrow(240, MID, 388, MID),
    label(314, MID - 12, "ego_from_lidar", anchor="middle", size=12),
    note(314, MID + 22, "calibration file", anchor="middle"),
    note(314, MID + 40, "offset (1.2, 0, 1.8)", anchor="middle"),
    arrow(590, MID, 738, MID),
    label(664, MID - 12, "world_from_ego", anchor="middle", size=12),
    note(664, MID + 22, "localizer", anchor="middle"),
    note(664, MID + 40, "offset (100, 50, 0)", anchor="middle"),
    curve(140, CY + CH, 490, 318, 840, CY + CH + 4, "amber"),
    label(490, 248, "world_from_lidar", anchor="middle", size=13),
    note(
        490,
        268,
        "se3_compose(world_from_ego, ego_from_lidar): the adjacent ego cancels",
        anchor="middle",
    ),
    note(490, 288, "composed offset (101.2, 50.0, 1.8)", anchor="middle"),
    band(20, 324, 940, 162, "THEN, ONCE PER POINT", "grey"),
    code(38, 362, POINT_CODE, 904, 12),
    note(38, 448, "One transform per point with the composed pose: it rotates, then translates."),
    note(38, 468, "se3_inverse_transform takes a world point back into the lidar frame."),
]

write("frame_chain", svg(W, H, "".join(body)))
print("wrote frame_chain.svg")
