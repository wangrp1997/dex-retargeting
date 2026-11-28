# Thumb opposition guidance: investigation, 2026-11-28

The operator clarified that opposition here means proximity between complete
distal links, with thumb-pad inward facing as a secondary preference. Pinching
would additionally require aligned pad contact regions; that is outside this
task. Slower opposition solving is acceptable. This memo records research and
offline diagnostics. The live default remains Botyard v4.2; the experimental
controller is disabled by default and can be selected with
`botyard_opposition:=inward`. Following the operator's real-hand test, that mode
is released as [Botyard v4.3](../botyard_v4_3/README.md); the offline observations
and limitations below remain the basis for it.

## What the existing metric misses

Botyard v4.2 closes the distance between complete distal STL surfaces. This can
reward lateral or dorsal proximity. Its secondary facing preference has weight
0.02 and points toward a field of four primary-pose finger pads, rather than
explicitly toward the currently opposing finger's volar surface.

On the cached right-hand example, the index opposition window (10.3–11.1 s) has
median full-surface gap 1.181 mm, but the median angle between the CAD thumb pad
normal and the vector toward the index pad centre is 86.12 degrees. A small gap
therefore does not certify frontal pad contact. This angle is an approximate
geometric diagnostic using fixed CAD pad centres/normals, not tactile evidence.

## Initial-value diagnostic

With the existing objective and protection bounds, each of three source frames
was solved from ten initial thumb base poses: the primary pose and a 3-by-3 grid
of THJ4/THJ3 at -45, 0 and +45 degrees. The primary, source chain and other initial
joints were identical across trials. A synthetic 1 s step exposed the static
search range; these are not live-rate motion or latency tests.

| Example frame | Pair | Full-surface gap range, mm | Thumb-facing angle range, degrees |
| --- | --- | ---: | ---: |
| 318 | Index | 1.124–4.261 | 85.95–89.05 |
| 369 | Middle | 1.737–1.772 | 85.14–85.28 |
| 417 | Ring | 3.379–3.513 | 108.81–109.42 |

All returned candidates passed the existing source and physical protections.
Different initial values can produce different results, but none of these trials
produced frontal pad opposition. This does not establish a global optimum or
general unreachability. It does show why changing the initializer alone is
insufficient evidence that the facing problem is solved.

Separate static probes added a weighted thumb-normal-to-finger residual. They
improved facing while sometimes increasing separation substantially. Merely
raising the orientation weight is not a validated solution either. Thumb pose
allowances, pad geometry and the combined position/orientation task need to be
checked together.

## Public implementations checked

* [ContactOpt, CVPR 2021](https://github.com/facebookresearch/ContactOpt): optimizes
  hand pose against desired surface contact. Its
  [contact model](https://github.com/facebookresearch/ContactOpt/blob/main/contactopt/diffcontact.py)
  places capsules along surface normals and includes an optional normal-dot
  contact formulation (`sdf_to_contact`, method 4). The default contact method is
  0; method 4 should not be described as the paper's default. The
  [runner](https://github.com/facebookresearch/ContactOpt/blob/main/contactopt/run_contactopt.py)
  implements random restarts and retains the lowest-loss result; the demo and
  image-estimate presets request eight restarts. This is MANO/object fitting,
  not an existing Botyard self-opposition controller. Its deformable-tissue
  penetration allowance must not replace the robot's physical clearance.
* [BioIK / bio_ik](https://github.com/TAMS-Group/bio_ik): generalized optimization
  IK with weighted position, orientation, look-at, direction and displacement
  goals. The [goal definitions](https://github.com/TAMS-Group/bio_ik/blob/master/include/bio_ik/goal_types.h)
  demonstrate the kind of explicit directional target needed here. The default
  repository mode is `bio2_memetic`. This is a useful formulation/search
  reference, not a drop-in replacement for our constrained SLSQP pipeline.
* [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet): its
  [energy](https://github.com/PKU-EPIC/DexGraspNet/blob/main/grasp_generation/utils/energy.py)
  combines distance, contact-normal force closure, penetration and joint terms;
  its optimizer combines gradient steps, contact resampling and annealing.
  Force closure against an object is a different objective from aligning the
  thumb pad with another finger, so that term should not be copied verbatim.

## Tested adaptation after the opposition/pinching clarification

The earlier volar-region objective was more restrictive than requested. It
improved some angles while increasing the example index gap from 1.18 to 5.98 mm;
it is retained only as an explicitly selected offline counterexample (`--mode
volar`). The corrected experiment uses **complete distal links** for its
optimization distances. Volar-region and opposing-pad metrics are supplementary
diagnostics, not position/alignment targets in this corrected controller.

The offline script is `../opposition_guidance_experiment.py`. It has two levels:

1. Solve the existing v4.2 full-link closure task from the actual previous
   command. Retain its source-shape and physical bounds, including free J4s.
2. Improve a soft thumb inward-facing objective, with four additional constraints
   `alpha_i * (gap_i - first_gap_i) <= 0.1 mm`, where
   `alpha_i = weight_i / (sum(weights) + 1e-12)`. The geometry is the complete
   distal STL in both levels. The allowance is confidence weighted, not a
   universal unweighted 0.1 mm cap on every finger. All trials retain the same
   previous command and frame dt; restarts cannot accumulate velocity allowance.

The palm-frame normal reference uses the CAD neutral thumb base orientation and
the first result's **current thumb flexion**. It does not request two pad centres
or opposing pad normals to align. Reference angles are used to calculate a
direction, not fixed or commanded joints. The reference is frozen within each
frame, so moving a finger cannot directly reduce its angular objective.

Up to three seeds are tried, with feasible evaluated results retained on native
optimizer failures. The joint anchor has constant weight 0.03, independent of
proximity: weakening both the facing task and its anchor together would allow
tiny nonzero weights to select a visibly different pose on an open hand. An
objective already below the existing 1e-6 stopping precision can return the
first result directly because all terms are nonnegative. There is no gesture
classifier or index-crossing threshold.

The distance bounds reference the **same-frame** first solve warmed from the
experimental trajectory. They do not guarantee a 0.1 mm difference from an
independent historical v4.2 trajectory, whose preceding commands differ. Full
causal replay is therefore still necessary, rather than checking one frame.

## Final offline comparison

The final experiment used inward guidance weight 0.2 and up to three starts.
All official example frames (621), custom frames (395 total, 330 tracked) and
sweep frames (149) were replayed. Full-link collision clearance, joint limits,
velocity and source protections passed throughout; the maximum normalized
distance-priority residual was below its 1e-7 acceptance tolerance.

The following are medians over the previously used source time windows. The
angle is against the **inward reference defined above**, not MANO orientation or
the direction to another finger's pad centre. It must not be compared directly
with the 86 degree pad-centre-facing diagnostic earlier in this memo.

| Window | Full-link gap v4.2 / experiment, mm | Inward reference angle v4.2 / experiment, degrees |
| --- | ---: | ---: |
| Example index, 10.3–11.1 s | 1.181 / 1.620 | 46.88 / 20.38 |
| Example middle, 11.85–12.54 s | 1.774 / 1.696 | 16.76 / 5.21 |
| Example ring, 13.67–14.21 s | 3.015 / 3.074 | 59.58 / 56.40 |
| Example little, 15.12–15.65 s | 18.493 / 18.574 | 62.94 / 61.55 |
| Custom index, 7.9–9.1 s | 1.616 / 1.715 | 34.33 / 30.56 |
| Custom index, 53.3–60.9 s | 1.482 / 1.578 | 28.04 / 25.14 |

Four-finger mean direction errors were 10.447 / 10.270 degrees on example and
12.588 / 12.587 on custom. The two example fist windows retained their source
protections; their direction errors were 23.83 / 22.94 and 35.51 / 30.90 degrees.
Sweep commanded joints differed by at most 0.00373 degrees. Strict raw-source
open-hand screening found only one example frame and none on custom/sweep, so
that screening alone cannot establish open-hand quality. Both-hand tests cover
negligible proximity weights, objective derivatives, fallback and the shared
velocity allowance across starts. The complete suite passed 155 tests; two
parameter cases intentionally skip a corrected-mode-only assertion on the
earlier volar counterexample.

Custom frames 208 and 240 encountered numerical optimizer exceptions; the
retained commands remained feasible. The experiment improves inward bias most
clearly for example index/middle opposition, but does not solve little-finger
reach or establish uniformly better performance for every gesture. MANO
proximity and geometric gaps do not establish sensed hardware contact.

The live default remains **Botyard v4.2**. The reviewed inward controller can now
be selected explicitly with `botyard_opposition:=inward`; both serial and spawned
live paths use the same factory and the exact reviewed parameters. Removing that
option or selecting `v4_2` restores the original controller. The hardware relay
records the mode in diagnostic metadata and retains its existing control settings
and input/feedback timeouts. Large outputs stay in
the ignored `logs/experiments/20261128-opposition-guidance-final/` directory:
`comparison.json`, per-frame CSVs, commands, source-code snapshots, manifest and
`example/key_comparison.mp4` / `custom/key_comparison.mp4`. Earlier v1/v2/v3
directories denote experiment rounds, not released Botyard algorithm versions.
