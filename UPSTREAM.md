# Upstream and contributions

This project builds on [DiyaJabin/vision-based-robotic-arm](https://github.com/DiyaJabin/vision-based-robotic-arm).
The initial source snapshot was read from the author's downloaded repository on 2026-11-13.
No upstream commit SHA was present in that ZIP checkout, so no exact commit is claimed.

The `core/`, `robot/`, `simulation/` and `perception/` foundations originate from that MIT-licensed project.
Keep `LICENSES/upstream-MIT.txt` when redistributing these portions. Shared motion code may have
subsequent fixes; the Git history and release notes identify modifications.

The `robosort/` package adds explicit RGB-D contracts, reproducible scene catalogues, predicted-mask
and box localization, synthetic instance-segmentation datasets, managed training artifacts,
truth-independent class decisions, per-object outcomes, and auditable paired experiments.

The overall integrated research application is distributed under AGPL-3.0-or-later because it uses
Ultralytics' open-source YOLO implementation. Upstream MIT code retains its original notice.
Third-party model weights and dependencies retain their own licenses. This project does not claim
to have invented YOLO or to validate physical finger grasping or sim-to-real transfer.
