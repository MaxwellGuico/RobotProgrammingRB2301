# Simulation decision viewer

The RViz publisher is `rb2301_fp/rviz_sim_view.py`. Its configuration is
`rviz/fp_simulation.rviz`; `fp_rviz_sim.sh` starts both. All visualization and
replay code lives outside `obstacle_course.py`. The viewer subscribes to existing
telemetry and never publishes robot commands.
The launcher removes conflicting editor Snap GTK settings and uses software
OpenGL rendering. Run it from a normal terminal so RViz can save its usual
`~/.rviz2` settings. Closing an RViz window started with `--rviz` at the end of a
simulation check is intentional test cleanup; a standalone replay remains open.

After building `rb2301_fp`, attach to an existing simulation whose controller
was started with `--simulation --terminal-debug`:

```bash
./fp_rviz_sim.sh
```

Use the same `ROS_DOMAIN_ID` as the simulator and controller. For an isolated
`fp_sim_check.py` run, the simplest option starts the live viewer inside the same
run environment and closes it with the test:

```bash
source install/setup.bash
python3 tools/fp_sim_check.py --name inspect_seed_11 --seed 11 --rviz
```

Run names must be unique. To attach separately in the same ROS/network
environment, use the domain printed by that runner:

```bash
./fp_rviz_sim.sh --domain 188
```

To inspect a saved run without Gazebo, replay its recorded diagnostics:

```bash
./fp_rviz_sim.sh --domain 230 --replay gap_commit_final_random_35547
```

The replay follows the recorded wall-time samples, holds the final frame when
finished, and never sends recorded commands to the robot. The argument can also
be a path to a `trajectory.jsonl`. Use a separate domain for replay when a live
viewer is already running. Add `--no-gui` to publish markers without opening RViz.
For current-format runs, replay reads the lookahead from the adjacent controller
snapshot without executing it. Thus a saved 0.40 m run retains its original
geometry even when the live controller uses 0.20 m. If the saved setting is
unavailable, the preview is explicitly labelled as using the current config.

RViz opens in a top-down `odom` view. Expand **Robot decisions → Namespaces** to
toggle individual layers. Use the mouse wheel to zoom and the Move Camera tool
to pan. The publisher transforms its own markers into `odom`; it does not depend
on laser-frame TF or change the simulator's TF tree.

| Layer | Meaning |
| --- | --- |
| Blue circle and box | Controller's clearance circle and motion-check footprint |
| Grey points | LiDAR returns after the controller's mount transform |
| Faint red rays | Tested directions, drawn every five degrees |
| Green sectors | Current clear openings at the selected planning horizon, with at least three adjacent one-degree samples |
| Cyan arc | Reference for the maximum 0.20 m movement commitment |
| Cyan arrow | Gap that the current scan would select if replanning now |
| Yellow arrow and sphere | Actual committed waypoint from controller telemetry |
| Magenta curve and boxes | Next 0.5 seconds of motion predicted from the published command |
| Grey track markers | Observed panel tracks, with their learned periods when available |
| Visited-space points | Older path samples used to prefer new clear openings |
| Orange track and plane | Actual selected swinging obstacle during positioning, observing, or crossing |
| Orange point | Panel staging point |

Green shows geometric eligibility, not a guarantee that the full turn is safe.
The controller separately checks its turning footprint and may slow down or
stop. Cyan is a fresh scan preview; yellow is the retained selection and can
differ during a commitment. A previous panel is not highlighted as selected
during the next panel's scout phase. Hardware telemetry is not interpreted as
simulation coordinates. Missing live topics are labelled stale after one second.

## Recorded validation before reverse removal

Automatic reversing has since been removed at the user's request. The current
controller stops on a stall and rejects negative forward velocity. The recorded
three-seed passes below used the earlier reverse-recovery behaviour and are not
validation of the new no-reverse controller.

Three random seeds were drawn before validation and retained through iteration:
35547, 3259, and 4632. The final controller passed all three with identical core
and navigation-helper hashes. Each run reached the third section, crossed both
moving panels, reached `done`, and held a stopped command for two seconds.
Independent checks confirmed the robot was beyond both actual panel planes and
approximately 0.36 m from the end wall.

| Seed | Wall time | Result |
| --- | ---: | --- |
| 35547 | 148.90 s | Complete |
| 3259 | 313.51 s | Complete |
| 4632 | 188.68 s | Complete |

Seed 3259 still took detours before finding the final opening. The 4632 batch
attempt was interrupted by its process environment; rerunning the same seeded
world independently passed. The interruption remains recorded in the manifest.

Navigation lives in `gap_navigation.py`. The main `obstacle_course.py` is 379
lines, reduced from 504. All 79 unit tests pass. The committed movement is at most
0.20 m; planning can evaluate farther ahead. Tight openings shorten the movement
to the available horizon, and footprint safety remains active every control tick.

Reports, plots, source snapshots, and the random-seed manifest are under
`log/fp_sim_checks/`: `gap_commitment_validation.json`,
`gap_commitment_validation.png`, and `gap_commitment_validation_manifest.json`.

## Earlier fixed-horizon trials

The fixed 0.40 m and 0.20 m trials below are earlier baselines. The current
controller uses the separate `gap_navigation.py` helper, evaluates longer
openings, and follows a committed segment for at most 0.20 m. The RViz status
shows the movement commitment and next-gap planning horizon separately. The
viewer shares the controller's `choose_gap()` function rather than duplicating
the selection rule, including path-history preference from telemetry. Replays
read both the core and helper snapshots.

### Fixed 0.40 m results

Seeds 11, 22, and 33 were tested sequentially with the same controller source,
fixed 0.40 m lookahead and commitment, 0.02 m arrival tolerance, two-degree
steering deadband, filtered yaw damping, and limited turn acceleration.
Each run reached the 240 s limit while still in `approach`; none reached the
moving panels. These runs expose repeated route circling despite the steering
command smoothing. This setting is not validated for course completion.

Logs and trajectory recordings are under
`log/fp_sim_checks/lookahead_040_smooth_seed_{11,22,33}/`.
The combined results and path plot are
`log/fp_sim_checks/lookahead_040_smooth_report.json` and `.png`.

### Fixed 0.20 m fallback results

The fallback retains the steering changes and reduces forward speed as the
waypoint approaches: the speed cap is 1.5 times the remaining waypoint distance.
Seeds 11, 22, and 33 were retested sequentially with identical controller sources.
All three reached the 240 s limit in `approach`; repeated route circling and
recovery remain. Neither tested fixed-lookahead setting is validated for course
completion. The final control/viewer unit suite passes 72 tests.

The final fallback recordings are
`log/fp_sim_checks/lookahead_020_approach_seed_{11,22,33}/`;
the report is `log/fp_sim_checks/lookahead_020_approach_report.json` and `.png`.
Replay the fallback with:

```bash
./fp_rviz_sim.sh --domain 230 --replay lookahead_020_approach_seed_11
```
