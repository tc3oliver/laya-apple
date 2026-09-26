# Phase 0: is the window-onset E-core residency hetero-specific? (#96 / #99 raw data only)

Post-hoc closure analysis of existing raw data; it changes no earlier verdict. Definitions, reading rules and the G1/G2/G3 classification are in `scripts/onset_phase0.py`, fixed before its output was read.

Overall: **G3**. A: **G3**; PB-ASYNC: **G3**.

- A: A-r1 c0 solo_short neither E- nor P-onset; A-r1 c0 solo_long unreadable (excluded); A-r1 c1 solo_long unreadable (excluded); A-r1 c1 solo_short E-onset, switch 0.5; A-r2 c0 solo_short E-onset, switch 0.5; A-r2 c0 solo_long unreadable (excluded); A-r2 c1 solo_long unreadable (excluded); A-r2 c1 solo_short E-onset, switch 0.5; A-r1 c0 hetero not E-onset; A-r1 c1 hetero not E-onset; A-r2 c0 hetero not E-onset; A-r2 c1 hetero returns to P within 1 s
- PB-ASYNC: PB-ASYNC-r1 c0 solo_short neither E- nor P-onset; PB-ASYNC-r1 c0 solo_long unreadable (excluded); PB-ASYNC-r1 c1 solo_long unreadable (excluded); PB-ASYNC-r1 c1 solo_short E-onset, switch 0.5; PB-ASYNC-r2 c0 solo_short E-onset, switch 0.5; PB-ASYNC-r2 c0 solo_long unreadable (excluded); PB-ASYNC-r2 c1 solo_long unreadable (excluded); PB-ASYNC-r2 c1 solo_short E-onset, switch 0.5; B-r1 c0 solo_short E-onset, switch 0.5; B-r1 c0 solo_long unreadable (excluded); B-r1 c1 solo_long unreadable (excluded); B-r1 c1 solo_short neither E- nor P-onset; O-r1 c0 solo_short E-onset, switch 0.5; O-r1 c0 solo_long unreadable (excluded); O-r1 c1 solo_long unreadable (excluded); O-r1 c1 solo_short E-onset, switch 0.5; Q-r1 c0 solo_short neither E- nor P-onset; Q-r1 c0 solo_long unreadable (excluded); Q-r1 c1 solo_long unreadable (excluded); Q-r1 c1 solo_short neither E- nor P-onset; PB-ASYNC-r2 c0 hetero not E-onset

## Production A: hetero windows at onset

A had no host-slow transient in #96. Parent-active E share by bucket (worker in parentheses), and the E->P switch time.

| run | cycle | E-onset | 0-0.5 | 0.5-1 | 1-2 | 2-4 | 4-8 | 10-20 | switch parent / worker |
|---|---|---|---|---|---|---|---|---|---|
| A-r1 | 0 | no | 0.44 (0.54) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | P from onset / 0.5 |
| A-r1 | 1 | no | 0.20 (0.28) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | P from onset / P from onset |
| A-r2 | 0 | no | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | P from onset / P from onset |
| A-r2 | 1 | yes | 0.58 (0.61) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.5 / 0.5 |

## Every sampled window: reading summary (parent-active)

| run | cell | cycle | condition | 0-0.5 E (ms) | 0.5-1 E (ms) | readable | onset | switch parent / worker |
|---|---|---|---|---|---|---|---|---|
| A-r1 | A | 0 | solo_short | 0.35 (59.8) | 0.00 (37.3) | yes | mixed | P from onset / – |
| A-r1 | A | 0 | solo_long | 0.54 (12.8) | 0.00 (6.0) | no | – | 0.5 / P from onset |
| A-r1 | A | 0 | hetero | 0.44 (91.7) | 0.00 (42.4) | yes | mixed | P from onset / 0.5 |
| A-r1 | A | 1 | hetero | 0.20 (78.5) | 0.00 (42.7) | yes | P | P from onset / P from onset |
| A-r1 | A | 1 | solo_long | 0.35 (9.2) | 0.00 (6.9) | no | – | P from onset / P from onset |
| A-r1 | A | 1 | solo_short | 0.68 (94.2) | 0.00 (37.2) | yes | E | 0.5 / – |
| A-r2 | A | 0 | solo_short | 0.52 (66.5) | 0.00 (37.7) | yes | E | 0.5 / – |
| A-r2 | A | 0 | solo_long | 0.51 (14.0) | 0.00 (6.0) | no | – | 0.5 / P from onset |
| A-r2 | A | 0 | hetero | 0.00 (54.4) | 0.00 (43.9) | yes | P | P from onset / P from onset |
| A-r2 | A | 1 | hetero | 0.58 (84.2) | 0.00 (42.0) | yes | E | 0.5 / 0.5 |
| A-r2 | A | 1 | solo_long | 0.18 (8.8) | 0.00 (5.9) | no | – | P from onset / P from onset |
| A-r2 | A | 1 | solo_short | 0.67 (90.8) | 0.00 (38.6) | yes | E | 0.5 / – |
| PB-ASYNC-r1 | PB-ASYNC | 0 | solo_short | 0.50 (48.1) | 0.00 (25.8) | yes | mixed | P from onset / – |
| PB-ASYNC-r1 | PB-ASYNC | 0 | solo_long | 0.38 (10.2) | 0.00 (5.8) | no | – | P from onset / P from onset |
| PB-ASYNC-r1 | PB-ASYNC | 0 | hetero | 0.95 (124.8) | 1.00 (150.1) | yes | E | 3.0 / 2.5 |
| PB-ASYNC-r1 | PB-ASYNC | 1 | hetero | 0.90 (111.2) | 1.00 (145.9) | yes | E | 4.0 / 4.0 |
| PB-ASYNC-r1 | PB-ASYNC | 1 | solo_long | 0.44 (12.1) | 0.00 (5.9) | no | – | P from onset / P from onset |
| PB-ASYNC-r1 | PB-ASYNC | 1 | solo_short | 0.52 (44.2) | 0.00 (26.4) | yes | E | 0.5 / – |
| PB-ASYNC-r2 | PB-ASYNC | 0 | solo_short | 0.52 (49.3) | 0.00 (26.1) | yes | E | 0.5 / – |
| PB-ASYNC-r2 | PB-ASYNC | 0 | solo_long | 0.52 (14.0) | 0.00 (5.8) | no | – | 0.5 / P from onset |
| PB-ASYNC-r2 | PB-ASYNC | 0 | hetero | 0.38 (61.8) | 0.00 (32.7) | yes | mixed | P from onset / P from onset |
| PB-ASYNC-r2 | PB-ASYNC | 1 | hetero | 0.94 (119.0) | 1.00 (147.1) | yes | E | 15.0 / 15.0 |
| PB-ASYNC-r2 | PB-ASYNC | 1 | solo_long | 0.60 (13.1) | 0.00 (5.7) | no | – | 0.5 / 0.5 |
| PB-ASYNC-r2 | PB-ASYNC | 1 | solo_short | 0.55 (46.7) | 0.00 (26.0) | yes | E | 0.5 / – |
| B-r1 | PB-ASYNC | 0 | solo_short | 0.53 (50.4) | 0.00 (25.8) | yes | E | 0.5 / – |
| B-r1 | PB-ASYNC | 0 | solo_long | 0.53 (14.3) | 0.00 (5.8) | no | – | 0.5 / P from onset |
| B-r1 | PB-ASYNC | 0 | hetero | 0.95 (120.1) | 1.00 (155.5) | yes | E | 19.5 / 19.5 |
| B-r1 | PB-ASYNC | 1 | hetero | 0.99 (134.4) | 1.00 (143.6) | yes | E | 17.0 / 17.0 |
| B-r1 | PB-ASYNC | 1 | solo_long | 0.18 (8.9) | 0.00 (5.8) | no | – | P from onset / P from onset |
| B-r1 | PB-ASYNC | 1 | solo_short | 0.45 (42.0) | 0.00 (25.6) | yes | mixed | P from onset / – |
| O-r1 | PB-ASYNC | 0 | solo_short | 0.51 (48.3) | 0.00 (26.2) | yes | E | 0.5 / – |
| O-r1 | PB-ASYNC | 0 | solo_long | 0.50 (14.5) | 0.00 (6.0) | no | – | 0.5 / P from onset |
| O-r1 | PB-ASYNC | 0 | hetero | 0.93 (127.5) | 1.00 (150.0) | yes | E | 4.0 / 4.0 |
| O-r1 | PB-ASYNC | 1 | hetero | 0.93 (130.4) | 1.00 (158.6) | yes | E | 11.5 / 11.5 |
| O-r1 | PB-ASYNC | 1 | solo_long | 0.55 (14.3) | 0.00 (6.4) | no | – | 0.5 / P from onset |
| O-r1 | PB-ASYNC | 1 | solo_short | 0.52 (44.7) | 0.00 (26.2) | yes | E | 0.5 / – |
| Q-r1 | PB-ASYNC | 0 | solo_short | 0.32 (40.3) | 0.00 (26.2) | yes | mixed | P from onset / – |
| Q-r1 | PB-ASYNC | 0 | solo_long | 0.53 (13.5) | 0.00 (6.2) | no | – | 0.5 / P from onset |
| Q-r1 | PB-ASYNC | 0 | hetero | 0.93 (124.0) | 1.00 (140.8) | yes | E | 2.0 / 2.0 |
| Q-r1 | PB-ASYNC | 1 | hetero | 0.94 (121.1) | 1.00 (144.6) | yes | E | never / never |
| Q-r1 | PB-ASYNC | 1 | solo_long | 0.01 (10.1) | 0.00 (5.8) | no | – | P from onset / P from onset |
| Q-r1 | PB-ASYNC | 1 | solo_short | 0.49 (42.1) | 0.00 (25.9) | yes | mixed | P from onset / – |

Not sampled (separate GPU-only instance): A-r1: c0 gpu_only, c1 gpu_only; A-r2: c0 gpu_only, c1 gpu_only; PB-ASYNC-r1: c0 gpu_only, c1 gpu_only; PB-ASYNC-r2: c0 gpu_only, c1 gpu_only; B-r1: c0 gpu_only, c1 gpu_only; O-r1: c0 gpu_only, c1 gpu_only; Q-r1: c0 gpu_only, c1 gpu_only.

## Active threads per window (>= 20 ms CPU in [t0, t0 + 4 s))

| run | cycle | condition | active threads (group:name:tid, onset CPU) |
|---|---|---|---|
| A-r1 | 0 | solo_short | chain:client-short:26483037(60 ms), chain:laya-ane-dispatch:26482955(264 ms) |
| A-r1 | 0 | solo_long | parent-other:client-long:26483559(54 ms), worker:gpu-worker:26482920(202 ms), worker:gpu-worker:26482958(101 ms), worker:gpu-worker:26483571(69 ms) |
| A-r1 | 0 | hetero | chain:client-short:26483759(63 ms), chain:laya-ane-dispatch:26482955(276 ms), parent-other:client-long:26483760(50 ms), worker:gpu-worker:26482920(201 ms), worker:gpu-worker:26482964(77 ms), worker:gpu-worker:26483605(67 ms) |
| A-r1 | 1 | hetero | chain:client-short:26484369(60 ms), chain:laya-ane-dispatch:26482955(267 ms), parent-other:client-long:26484370(46 ms), worker:gpu-worker:26482920(184 ms), worker:gpu-worker:26482964(94 ms), worker:gpu-worker:26483605(43 ms) |
| A-r1 | 1 | solo_long | parent-other:client-long:26484568(52 ms), worker:gpu-worker:26482920(208 ms), worker:gpu-worker:26483605(102 ms), worker:gpu-worker:26483932(66 ms) |
| A-r1 | 1 | solo_short | chain:client-short:26484816(61 ms), chain:laya-ane-dispatch:26482955(292 ms) |
| A-r2 | 0 | solo_short | chain:client-short:26489823(61 ms), chain:laya-ane-dispatch:26489715(264 ms) |
| A-r2 | 0 | solo_long | parent-other:client-long:26490115(55 ms), worker:gpu-worker:26489667(207 ms), worker:gpu-worker:26489717(108 ms), worker:gpu-worker:26490119(67 ms) |
| A-r2 | 0 | hetero | chain:client-short:26491254(58 ms), chain:laya-ane-dispatch:26489715(253 ms), parent-other:client-long:26491255(43 ms), worker:gpu-worker:26489667(179 ms), worker:gpu-worker:26489717(63 ms), worker:gpu-worker:26489718(59 ms) |
| A-r2 | 1 | hetero | chain:client-short:26492025(62 ms), chain:laya-ane-dispatch:26489715(270 ms), parent-other:client-long:26492026(48 ms), worker:gpu-worker:26489667(202 ms), worker:gpu-worker:26489717(61 ms), worker:gpu-worker:26489718(64 ms) |
| A-r2 | 1 | solo_long | parent-other:client-long:26492251(50 ms), worker:gpu-worker:26489667(197 ms), worker:gpu-worker:26489718(81 ms), worker:gpu-worker:26491366(68 ms) |
| A-r2 | 1 | solo_short | chain:client-short:26492486(60 ms), chain:laya-ane-dispatch:26489715(291 ms) |
| PB-ASYNC-r1 | 0 | solo_short | chain:client-short:26485462(63 ms), chain:coreml-callback:26485217(33 ms), chain:laya-ane-dispatch:26485223(135 ms) |
| PB-ASYNC-r1 | 0 | solo_long | parent-other:client-long:26485741(52 ms), worker:gpu-worker:26485187(211 ms), worker:gpu-worker:26485232(106 ms), worker:gpu-worker:26485752(68 ms) |
| PB-ASYNC-r1 | 0 | hetero | chain:client-short:26485939(186 ms), chain:coreml-callback:26485217(72 ms), chain:laya-ane-dispatch:26485223(400 ms), parent-other:client-long:26485940(179 ms), parent-other:laya-gpu-dispatch:26485240(36 ms), worker:gpu-worker:26485187(626 ms), worker:gpu-worker:26485226(147 ms), worker:gpu-worker:26485824(112 ms) |
| PB-ASYNC-r1 | 1 | hetero | chain:client-short:26486820(236 ms), chain:coreml-callback:26485217(89 ms), chain:laya-ane-dispatch:26485223(500 ms), parent-other:client-long:26486821(231 ms), parent-other:laya-gpu-dispatch:26485240(45 ms), worker:gpu-worker:26485187(777 ms), worker:gpu-worker:26485752(175 ms), worker:gpu-worker:26485823(120 ms) |
| PB-ASYNC-r1 | 1 | solo_long | parent-other:client-long:26487010(52 ms), worker:gpu-worker:26485187(200 ms), worker:gpu-worker:26485752(105 ms), worker:gpu-worker:26486883(68 ms) |
| PB-ASYNC-r1 | 1 | solo_short | chain:client-short:26487246(61 ms), chain:coreml-callback:26485217(32 ms), chain:laya-ane-dispatch:26485223(132 ms) |
| PB-ASYNC-r2 | 0 | solo_short | chain:client-short:26487809(63 ms), chain:coreml-callback:26487669(32 ms), chain:laya-ane-dispatch:26487680(136 ms) |
| PB-ASYNC-r2 | 0 | solo_long | parent-other:client-long:26487986(54 ms), worker:gpu-worker:26487639(208 ms), worker:gpu-worker:26487682(106 ms), worker:gpu-worker:26487990(65 ms) |
| PB-ASYNC-r2 | 0 | hetero | chain:client-short:26488190(64 ms), chain:coreml-callback:26487669(30 ms), chain:laya-ane-dispatch:26487680(143 ms), parent-other:client-long:26488191(52 ms), worker:gpu-worker:26487639(215 ms), worker:gpu-worker:26487682(71 ms), worker:gpu-worker:26488133(68 ms) |
| PB-ASYNC-r2 | 1 | hetero | chain:client-short:26488979(246 ms), chain:coreml-callback:26487669(93 ms), chain:laya-ane-dispatch:26487680(522 ms), parent-other:client-long:26488980(240 ms), parent-other:laya-gpu-dispatch:26487703(47 ms), worker:gpu-worker:26487639(817 ms), worker:gpu-worker:26487990(178 ms), worker:gpu-worker:26489002(122 ms) |
| PB-ASYNC-r2 | 1 | solo_long | parent-other:client-long:26489178(53 ms), worker:gpu-worker:26487639(205 ms), worker:gpu-worker:26488133(101 ms), worker:gpu-worker:26489186(66 ms) |
| PB-ASYNC-r2 | 1 | solo_short | chain:client-short:26489373(62 ms), chain:coreml-callback:26487669(31 ms), chain:laya-ane-dispatch:26487680(134 ms) |
| B-r1 | 0 | solo_short | chain:client-short:26548933(64 ms), chain:coreml-callback:26548819(32 ms), chain:laya-ane-dispatch:26548830(136 ms) |
| B-r1 | 0 | solo_long | parent-other:client-long:26549132(54 ms), worker:gpu-worker:26548522(205 ms), worker:gpu-worker:26548844(101 ms), worker:gpu-worker:26549142(69 ms) |
| B-r1 | 0 | hetero | chain:client-short:26549318(242 ms), chain:coreml-callback:26548819(95 ms), chain:laya-ane-dispatch:26548830(515 ms), parent-other:client-long:26549319(235 ms), parent-other:laya-gpu-dispatch:26548851(48 ms), worker:gpu-worker:26548522(796 ms), worker:gpu-worker:26548833(165 ms), worker:gpu-worker:26549324(142 ms) |
| B-r1 | 1 | hetero | chain:client-short:26549896(239 ms), chain:coreml-callback:26548819(89 ms), chain:laya-ane-dispatch:26548830(505 ms), parent-other:client-long:26549897(233 ms), parent-other:laya-gpu-dispatch:26548851(45 ms), worker:gpu-worker:26548522(808 ms), worker:gpu-worker:26548844(133 ms), worker:gpu-worker:26549324(153 ms) |
| B-r1 | 1 | solo_long | parent-other:client-long:26550214(49 ms), worker:gpu-worker:26548522(198 ms), worker:gpu-worker:26548844(101 ms), worker:gpu-worker:26549433(70 ms) |
| B-r1 | 1 | solo_short | chain:client-short:26550539(59 ms), chain:coreml-callback:26548819(32 ms), chain:laya-ane-dispatch:26548830(134 ms) |
| O-r1 | 0 | solo_short | chain:client-short:26551316(65 ms), chain:coreml-callback:26551161(33 ms), chain:laya-ane-dispatch:26551217(134 ms) |
| O-r1 | 0 | solo_long | parent-other:client-long:26551464(55 ms), worker:gpu-worker:26551136(210 ms), worker:gpu-worker:26551226(105 ms), worker:gpu-worker:26551468(67 ms) |
| O-r1 | 0 | hetero | chain:client-short:26551802(257 ms), chain:coreml-callback:26551161(96 ms), chain:laya-ane-dispatch:26551217(525 ms), parent-other:client-long:26551803(238 ms), parent-other:laya-gpu-dispatch:26551232(48 ms), worker:gpu-worker:26551136(831 ms), worker:gpu-worker:26551219(101 ms), worker:gpu-worker:26551468(163 ms), worker:gpu-worker:26551756(39 ms) |
| O-r1 | 1 | hetero | chain:client-short:26552436(262 ms), chain:coreml-callback:26551161(97 ms), chain:laya-ane-dispatch:26551217(532 ms), parent-other:client-long:26552437(235 ms), parent-other:laya-gpu-dispatch:26551232(47 ms), worker:gpu-worker:26551136(824 ms), worker:gpu-worker:26551219(157 ms), worker:gpu-worker:26551998(129 ms) |
| O-r1 | 1 | solo_long | parent-other:client-long:26552658(55 ms), worker:gpu-worker:26551136(206 ms), worker:gpu-worker:26551219(104 ms), worker:gpu-worker:26552663(68 ms) |
| O-r1 | 1 | solo_short | chain:client-short:26552877(63 ms), chain:coreml-callback:26551161(32 ms), chain:laya-ane-dispatch:26551217(132 ms) |
| Q-r1 | 0 | solo_short | chain:client-short:26553384(58 ms), chain:coreml-callback:26553290(32 ms), chain:laya-ane-dispatch:26553302(133 ms) |
| Q-r1 | 0 | solo_long | parent-other:client-long:26553606(54 ms), worker:gpu-worker:26553261(206 ms), worker:gpu-worker:26553305(107 ms), worker:gpu-worker:26553607(70 ms) |
| Q-r1 | 0 | hetero | chain:client-short:26553829(142 ms), chain:coreml-callback:26553290(58 ms), chain:laya-ane-dispatch:26553302(305 ms), parent-other:client-long:26553830(132 ms), parent-other:laya-gpu-dispatch:26553317(26 ms), worker:gpu-worker:26553261(477 ms), worker:gpu-worker:26553304(131 ms), worker:gpu-worker:26553835(96 ms) |
| Q-r1 | 1 | hetero | chain:client-short:26554593(250 ms), chain:coreml-callback:26553290(91 ms), chain:laya-ane-dispatch:26553302(522 ms), parent-other:client-long:26554594(247 ms), parent-other:laya-gpu-dispatch:26553317(48 ms), worker:gpu-worker:26553261(835 ms), worker:gpu-worker:26553985(168 ms), worker:gpu-worker:26554598(138 ms) |
| Q-r1 | 1 | solo_long | parent-other:client-long:26554908(50 ms), worker:gpu-worker:26553261(193 ms), worker:gpu-worker:26553835(85 ms), worker:gpu-worker:26554598(68 ms) |
| Q-r1 | 1 | solo_short | chain:client-short:26555114(59 ms), chain:coreml-callback:26553290(33 ms), chain:laya-ane-dispatch:26553302(131 ms) |

## Per bucket: E share, CPU ms, relative cycle rate P / E (cycles per CPU ns; relative, not a frequency)

| run | cycle | condition | aggregate | 0-0.5 | 0.5-1 | 1-2 | 2-4 | 4-8 | 10-20 |
|---|---|---|---|---|---|---|---|---|---|
| A-r1 | 0 | solo_short | parent-active | 0.35 · 60 · 3.38/1.14 | 0.00 · 37 · 3.98/– | 0.00 · 75 · 3.98/– | 0.00 · 151 · 3.99/– | 0.00 · 308 · 3.93/2.57 | 0.00 · 732 · 3.99/– |
| A-r1 | 0 | solo_short | chain | 0.35 · 60 · 3.38/1.14 | 0.00 · 37 · 3.98/– | 0.00 · 75 · 3.98/– | 0.00 · 151 · 3.99/– | 0.00 · 308 · 3.93/2.57 | 0.00 · 732 · 3.99/– |
| A-r1 | 0 | solo_long | parent-active | 0.54 · 13 · 3.50/1.56 | 0.00 · 6 · 4.43/– | 0.00 · 12 · 4.45/– | 0.00 · 23 · 4.44/– | 0.00 · 46 · 4.44/– | 0.00 · 117 · 4.43/– |
| A-r1 | 0 | solo_long | parent-other | 0.54 · 13 · 3.50/1.56 | 0.00 · 6 · 4.43/– | 0.00 · 12 · 4.45/– | 0.00 · 23 · 4.44/– | 0.00 · 46 · 4.44/– | 0.00 · 117 · 4.43/– |
| A-r1 | 0 | solo_long | worker | 0.44 · 89 · 2.59/2.21 | 0.00 · 41 · 3.12/– | 0.00 · 80 · 3.09/– | 0.00 · 162 · 3.09/– | 0.00 · 270 · 3.29/– | 0.00 · 645 · 3.37/– |
| A-r1 | 0 | hetero | parent-active | 0.44 · 92 · 2.67/1.69 | 0.00 · 42 · 4.14/– | 0.00 · 86 · 4.13/– | 0.00 · 169 · 4.14/– | 0.00 · 342 · 4.13/– | 0.00 · 853 · 4.13/2.52 |
| A-r1 | 0 | hetero | chain | 0.40 · 78 · 2.62/1.75 | 0.00 · 37 · 4.10/– | 0.00 · 76 · 4.10/– | 0.00 · 148 · 4.10/– | 0.00 · 301 · 4.09/– | 0.00 · 749 · 4.10/2.53 |
| A-r1 | 0 | hetero | parent-other | 0.69 · 14 · 3.28/1.46 | 0.00 · 5 · 4.43/– | 0.00 · 10 · 4.40/– | 0.00 · 21 · 4.43/– | 0.00 · 41 · 4.41/– | 0.00 · 104 · 4.41/2.43 |
| A-r1 | 0 | hetero | worker | 0.54 · 95 · 2.40/1.85 | 0.00 · 36 · 3.09/– | 0.00 · 74 · 3.17/– | 0.00 · 139 · 3.19/– | 0.00 · 236 · 3.39/– | 0.00 · 705 · 3.19/2.08 |
| A-r1 | 1 | hetero | parent-active | 0.20 · 78 · 2.83/1.87 | 0.00 · 43 · 4.14/– | 0.00 · 84 · 4.13/– | 0.00 · 168 · 4.14/– | 0.00 · 338 · 4.14/– | 0.00 · 843 · 4.13/– |
| A-r1 | 1 | hetero | chain | 0.16 · 68 · 2.78/1.93 | 0.00 · 37 · 4.10/– | 0.00 · 74 · 4.09/– | 0.00 · 148 · 4.10/– | 0.00 · 297 · 4.10/– | 0.00 · 741 · 4.09/– |
| A-r1 | 1 | hetero | parent-other | 0.50 · 10 · 3.34/1.74 | 0.00 · 5 · 4.43/– | 0.00 · 11 · 4.40/– | 0.00 · 20 · 4.42/– | 0.00 · 41 · 4.42/– | 0.00 · 103 · 4.41/– |
| A-r1 | 1 | hetero | worker | 0.28 · 75 · 2.40/2.21 | 0.00 · 30 · 3.29/– | 0.00 · 72 · 3.15/– | 0.00 · 144 · 3.11/– | 0.00 · 267 · 3.23/– | 0.00 · 580 · 3.39/– |
| A-r1 | 1 | solo_long | parent-active | 0.35 · 9 · 3.61/2.27 | 0.00 · 7 · 4.04/– | 0.00 · 12 · 4.45/– | 0.00 · 23 · 4.44/– | 0.00 · 48 · 4.43/– | 0.00 · 118 · 4.44/– |
| A-r1 | 1 | solo_long | parent-other | 0.35 · 9 · 3.61/2.27 | 0.00 · 7 · 4.04/– | 0.00 · 12 · 4.45/– | 0.00 · 23 · 4.44/– | 0.00 · 48 · 4.43/– | 0.00 · 118 · 4.44/– |
| A-r1 | 1 | solo_long | worker | 0.40 · 96 · 2.38/1.90 | 0.00 · 36 · 3.67/– | 0.00 · 82 · 3.12/– | 0.00 · 161 · 3.08/– | 0.00 · 328 · 3.09/– | 0.00 · 810 · 3.09/– |
| A-r1 | 1 | solo_short | parent-active | 0.68 · 94 · 3.35/1.38 | 0.00 · 37 · 4.00/– | 0.00 · 74 · 4.00/– | 0.00 · 147 · 3.98/– | 0.00 · 295 · 4.00/– | 0.00 · 737 · 3.99/– |
| A-r1 | 1 | solo_short | chain | 0.68 · 94 · 3.35/1.38 | 0.00 · 37 · 4.00/– | 0.00 · 74 · 4.00/– | 0.00 · 147 · 3.98/– | 0.00 · 295 · 4.00/– | 0.00 · 737 · 3.99/– |
| A-r2 | 0 | solo_short | parent-active | 0.52 · 66 · 3.38/1.10 | 0.00 · 38 · 3.97/– | 0.00 · 74 · 4.00/– | 0.00 · 147 · 3.99/– | 0.00 · 292 · 4.00/– | 0.00 · 736 · 3.99/– |
| A-r2 | 0 | solo_short | chain | 0.52 · 66 · 3.38/1.10 | 0.00 · 38 · 3.97/– | 0.00 · 74 · 4.00/– | 0.00 · 147 · 3.99/– | 0.00 · 292 · 4.00/– | 0.00 · 736 · 3.99/– |
| A-r2 | 0 | solo_long | parent-active | 0.51 · 14 · 3.22/1.33 | 0.00 · 6 · 4.45/– | 0.00 · 11 · 4.44/– | 0.00 · 23 · 4.44/– | 0.00 · 47 · 4.42/– | 0.00 · 129 · 4.25/– |
| A-r2 | 0 | solo_long | parent-other | 0.51 · 14 · 3.22/1.33 | 0.00 · 6 · 4.45/– | 0.00 · 11 · 4.44/– | 0.00 · 23 · 4.44/– | 0.00 · 47 · 4.42/– | 0.00 · 129 · 4.25/– |
| A-r2 | 0 | solo_long | worker | 0.45 · 99 · 2.44/1.76 | 0.00 · 41 · 3.14/– | 0.00 · 80 · 3.06/– | 0.00 · 162 · 3.09/– | 0.00 · 273 · 3.29/– | 0.00 · 669 · 3.44/– |
| A-r2 | 0 | hetero | parent-active | 0.00 · 54 · 3.03/– | 0.00 · 44 · 4.12/2.29 | 0.00 · 86 · 4.14/– | 0.00 · 169 · 4.13/– | 0.00 · 339 · 4.12/– | 0.00 · 859 · 4.10/2.58 |
| A-r2 | 0 | hetero | chain | 0.00 · 48 · 3.03/– | 0.00 · 39 · 4.08/2.35 | 0.00 · 75 · 4.10/– | 0.00 · 149 · 4.09/– | 0.00 · 298 · 4.08/– | 0.00 · 757 · 4.06/2.58 |
| A-r2 | 0 | hetero | parent-other | 0.00 · 7 · 3.09/– | 0.00 · 5 · 4.38/2.16 | 0.00 · 11 · 4.43/– | 0.00 · 20 · 4.43/– | 0.00 · 41 · 4.40/– | 0.00 · 103 · 4.38/– |
| A-r2 | 0 | hetero | worker | 0.00 · 42 · 2.60/– | 0.00 · 37 · 3.15/– | 0.00 · 75 · 3.16/– | 0.00 · 147 · 3.12/– | 0.00 · 293 · 3.12/– | 0.00 · 738 · 3.14/2.55 |
| A-r2 | 1 | hetero | parent-active | 0.58 · 84 · 2.92/1.78 | 0.00 · 42 · 4.13/– | 0.00 · 86 · 4.11/– | 0.00 · 168 · 4.11/– | 0.00 · 344 · 4.10/– | 0.00 · 852 · 4.08/– |
| A-r2 | 1 | hetero | chain | 0.57 · 72 · 2.93/1.79 | 0.00 · 37 · 4.09/– | 0.00 · 75 · 4.07/– | 0.00 · 148 · 4.07/– | 0.00 · 302 · 4.06/– | 0.00 · 751 · 4.04/– |
| A-r2 | 1 | hetero | parent-other | 0.60 · 12 · 2.86/1.73 | 0.00 · 5 · 4.44/– | 0.00 · 11 · 4.41/– | 0.00 · 20 · 4.43/– | 0.00 · 41 · 4.39/– | 0.00 · 102 · 4.38/– |
| A-r2 | 1 | hetero | worker | 0.61 · 71 · 2.56/1.72 | 0.00 · 36 · 3.10/– | 0.00 · 74 · 3.11/– | 0.00 · 145 · 3.09/– | 0.00 · 297 · 3.10/– | 0.00 · 739 · 3.09/– |
| A-r2 | 1 | solo_long | parent-active | 0.18 · 9 · 3.23/2.28 | 0.00 · 6 · 4.45/– | 0.00 · 12 · 4.44/– | 0.00 · 23 · 4.42/– | 0.00 · 47 · 4.43/– | 0.00 · 114 · 4.44/– |
| A-r2 | 1 | solo_long | parent-other | 0.18 · 9 · 3.23/2.28 | 0.00 · 6 · 4.45/– | 0.00 · 12 · 4.44/– | 0.00 · 23 · 4.42/– | 0.00 · 47 · 4.43/– | 0.00 · 114 · 4.44/– |
| A-r2 | 1 | solo_long | worker | 0.14 · 61 · 2.47/2.20 | 0.00 · 40 · 3.08/– | 0.00 · 82 · 3.09/– | 0.00 · 164 · 3.06/– | 0.00 · 273 · 3.28/– | 0.00 · 637 · 3.33/– |
| A-r2 | 1 | solo_short | parent-active | 0.67 · 91 · 3.34/1.47 | 0.00 · 39 · 3.96/– | 0.00 · 73 · 4.00/– | 0.00 · 149 · 3.99/– | 0.00 · 293 · 4.00/– | 0.00 · 732 · 3.99/– |
| A-r2 | 1 | solo_short | chain | 0.67 · 91 · 3.34/1.47 | 0.00 · 39 · 3.96/– | 0.00 · 73 · 4.00/– | 0.00 · 149 · 3.99/– | 0.00 · 293 · 4.00/– | 0.00 · 732 · 3.99/– |
| PB-ASYNC-r1 | 0 | solo_short | parent-active | 0.50 · 48 · 3.14/1.10 | 0.00 · 26 · 3.67/– | 0.00 · 52 · 3.67/– | 0.00 · 105 · 3.66/– | 0.00 · 203 · 3.66/– | 0.00 · 513 · 3.66/2.54 |
| PB-ASYNC-r1 | 0 | solo_short | chain | 0.50 · 48 · 3.14/1.10 | 0.00 · 26 · 3.67/– | 0.00 · 52 · 3.67/– | 0.00 · 105 · 3.66/– | 0.00 · 203 · 3.66/– | 0.00 · 513 · 3.66/2.54 |
| PB-ASYNC-r1 | 0 | solo_long | parent-active | 0.38 · 10 · 3.44/1.79 | 0.00 · 6 · 4.45/– | 0.00 · 12 · 4.38/– | 0.00 · 24 · 4.42/– | 0.00 · 47 · 4.43/– | 0.00 · 116 · 4.44/– |
| PB-ASYNC-r1 | 0 | solo_long | parent-other | 0.38 · 10 · 3.44/1.79 | 0.00 · 6 · 4.45/– | 0.00 · 12 · 4.38/– | 0.00 · 24 · 4.42/– | 0.00 · 47 · 4.43/– | 0.00 · 116 · 4.44/– |
| PB-ASYNC-r1 | 0 | solo_long | worker | 0.48 · 99 · 2.41/1.83 | 0.00 · 42 · 3.15/– | 0.00 · 83 · 3.13/– | 0.00 · 162 · 3.12/– | 0.00 · 269 · 3.29/– | 0.00 · 629 · 3.36/– |
| PB-ASYNC-r1 | 0 | hetero | parent-active | 0.95 · 125 · 2.00/1.48 | 1.00 · 150 · –/1.27 | 1.00 · 306 · –/1.25 | 0.66 · 292 · 3.72/1.18 | 0.00 · 262 · 3.89/– | 0.00 · 660 · 3.90/– |
| PB-ASYNC-r1 | 0 | hetero | chain | 0.94 · 97 · 1.89/1.49 | 1.00 · 110 · –/1.34 | 1.00 · 227 · –/1.31 | 0.65 · 224 · 3.61/1.22 | 0.00 · 208 · 3.78/– | 0.00 · 524 · 3.79/– |
| PB-ASYNC-r1 | 0 | hetero | parent-other | 0.96 · 27 · 2.62/1.41 | 1.00 · 40 · –/1.08 | 1.00 · 79 · –/1.09 | 0.70 · 68 · 4.17/1.05 | 0.00 · 54 · 4.34/– | 0.00 · 136 · 4.33/– |
| PB-ASYNC-r1 | 0 | hetero | worker | 0.87 · 163 · 1.95/1.53 | 1.00 · 140 · –/1.41 | 1.00 · 274 · –/1.39 | 0.62 · 307 · 3.13/1.29 | 0.00 · 275 · 3.35/– | 0.00 · 651 · 3.42/– |
| PB-ASYNC-r1 | 1 | hetero | parent-active | 0.90 · 111 · 1.87/1.52 | 1.00 · 146 · –/1.39 | 1.00 · 309 · –/1.24 | 0.95 · 536 · 2.47/1.35 | 0.00 · 268 · 3.91/– | 0.00 · 659 · 3.90/0.99 |
| PB-ASYNC-r1 | 1 | hetero | chain | 0.90 · 82 · 1.90/1.59 | 1.00 · 108 · –/1.47 | 1.00 · 234 · –/1.28 | 0.94 · 401 · 2.39/1.41 | 0.00 · 212 · 3.80/– | 0.00 · 523 · 3.79/0.99 |
| PB-ASYNC-r1 | 1 | hetero | parent-other | 0.90 · 29 · 1.76/1.31 | 1.00 · 38 · –/1.17 | 1.00 · 75 · –/1.10 | 0.96 · 134 · 2.85/1.20 | 0.00 · 55 · 4.33/– | 0.00 · 136 · 4.34/– |
| PB-ASYNC-r1 | 1 | hetero | worker | 0.77 · 145 · 1.87/1.65 | 1.00 · 134 · –/1.53 | 1.00 · 287 · –/1.33 | 0.95 · 507 · 2.41/1.51 | 0.00 · 320 · 3.19/– | 0.00 · 653 · 3.39/– |
| PB-ASYNC-r1 | 1 | solo_long | parent-active | 0.44 · 12 · 3.15/1.85 | 0.00 · 6 · 4.45/– | 0.00 · 11 · 4.44/– | 0.00 · 23 · 4.44/– | 0.00 · 48 · 4.35/– | 0.00 · 121 · 4.36/2.59 |
| PB-ASYNC-r1 | 1 | solo_long | parent-other | 0.44 · 12 · 3.15/1.85 | 0.00 · 6 · 4.45/– | 0.00 · 11 · 4.44/– | 0.00 · 23 · 4.44/– | 0.00 · 48 · 4.35/– | 0.00 · 121 · 4.36/2.59 |
| PB-ASYNC-r1 | 1 | solo_long | worker | 0.45 · 94 · 2.38/1.89 | 0.00 · 40 · 3.07/– | 0.00 · 79 · 3.07/– | 0.00 · 160 · 3.05/– | 0.00 · 269 · 3.30/– | 0.00 · 652 · 3.39/2.55 |
| PB-ASYNC-r1 | 1 | solo_short | parent-active | 0.52 · 44 · 3.10/1.30 | 0.00 · 26 · 3.66/– | 0.00 · 51 · 3.67/– | 0.00 · 103 · 3.68/– | 0.00 · 208 · 3.66/– | 0.00 · 515 · 3.66/– |
| PB-ASYNC-r1 | 1 | solo_short | chain | 0.52 · 44 · 3.10/1.30 | 0.00 · 26 · 3.66/– | 0.00 · 51 · 3.67/– | 0.00 · 103 · 3.68/– | 0.00 · 208 · 3.66/– | 0.00 · 515 · 3.66/– |
| PB-ASYNC-r2 | 0 | solo_short | parent-active | 0.52 · 49 · 3.18/1.14 | 0.00 · 26 · 3.67/– | 0.00 · 53 · 3.66/– | 0.00 · 103 · 3.68/– | 0.00 · 208 · 3.67/– | 0.00 · 516 · 3.67/– |
| PB-ASYNC-r2 | 0 | solo_short | chain | 0.52 · 49 · 3.18/1.14 | 0.00 · 26 · 3.67/– | 0.00 · 53 · 3.66/– | 0.00 · 103 · 3.68/– | 0.00 · 208 · 3.67/– | 0.00 · 516 · 3.67/– |
| PB-ASYNC-r2 | 0 | solo_long | parent-active | 0.52 · 14 · 3.12/1.31 | 0.00 · 6 · 4.42/– | 0.00 · 11 · 4.43/– | 0.00 · 23 · 4.44/– | 0.00 · 46 · 4.43/– | 0.00 · 113 · 4.44/– |
| PB-ASYNC-r2 | 0 | solo_long | parent-other | 0.52 · 14 · 3.12/1.31 | 0.00 · 6 · 4.42/– | 0.00 · 11 · 4.43/– | 0.00 · 23 · 4.44/– | 0.00 · 46 · 4.43/– | 0.00 · 113 · 4.44/– |
| PB-ASYNC-r2 | 0 | solo_long | worker | 0.38 · 95 · 2.33/1.86 | 0.00 · 41 · 3.10/– | 0.00 · 81 · 3.09/– | 0.00 · 163 · 3.09/– | 0.00 · 324 · 3.09/– | 0.00 · 733 · 3.18/– |
| PB-ASYNC-r2 | 0 | hetero | parent-active | 0.38 · 62 · 2.77/2.02 | 0.00 · 33 · 3.94/– | 0.00 · 65 · 3.91/– | 0.00 · 130 · 3.89/– | 0.00 · 256 · 3.89/– | 0.00 · 640 · 3.90/– |
| PB-ASYNC-r2 | 0 | hetero | chain | 0.36 · 50 · 2.71/2.06 | 0.00 · 27 · 3.82/– | 0.00 · 53 · 3.80/– | 0.00 · 106 · 3.78/– | 0.00 · 209 · 3.78/– | 0.00 · 525 · 3.80/– |
| PB-ASYNC-r2 | 0 | hetero | parent-other | 0.46 · 12 · 3.08/1.89 | 0.00 · 6 · 4.45/– | 0.00 · 12 · 4.43/– | 0.00 · 23 · 4.40/– | 0.00 · 46 · 4.42/– | 0.00 · 115 · 4.39/– |
| PB-ASYNC-r2 | 0 | hetero | worker | 0.35 · 64 · 2.55/2.08 | 0.00 · 41 · 3.18/– | 0.00 · 82 · 3.19/– | 0.00 · 165 · 3.19/– | 0.00 · 274 · 3.35/– | 0.00 · 652 · 3.43/– |
| PB-ASYNC-r2 | 1 | hetero | parent-active | 0.94 · 119 · 2.08/1.46 | 1.00 · 147 · –/1.38 | 1.00 · 305 · –/1.28 | 0.98 · 576 · 1.68/1.34 | 1.00 · 1225 · –/1.25 | 0.79 · 1771 · 3.71/1.31 |
| PB-ASYNC-r2 | 1 | hetero | chain | 0.92 · 86 · 2.00/1.53 | 1.00 · 110 · –/1.45 | 1.00 · 230 · –/1.33 | 0.97 · 435 · 1.68/1.40 | 1.00 · 916 · –/1.30 | 0.78 · 1342 · 3.61/1.36 |
| PB-ASYNC-r2 | 1 | hetero | parent-other | 0.97 · 33 · 2.62/1.26 | 1.00 · 37 · –/1.18 | 1.00 · 75 · –/1.15 | 0.98 · 141 · 1.64/1.17 | 1.00 · 309 · –/1.11 | 0.82 · 429 · 4.09/1.16 |
| PB-ASYNC-r2 | 1 | hetero | worker | 0.82 · 150 · 1.92/1.60 | 1.00 · 135 · –/1.48 | 1.00 · 288 · –/1.36 | 0.98 · 545 · 1.64/1.44 | 1.00 · 1031 · –/1.38 | 0.77 · 1519 · 3.31/1.43 |
| PB-ASYNC-r2 | 1 | solo_long | parent-active | 0.60 · 13 · 3.74/1.97 | 0.00 · 6 · 4.45/– | 0.00 · 11 · 4.44/– | 0.00 · 23 · 4.41/– | 0.00 · 45 · 4.40/– | 0.00 · 116 · 4.32/– |
| PB-ASYNC-r2 | 1 | solo_long | parent-other | 0.60 · 13 · 3.74/1.97 | 0.00 · 6 · 4.45/– | 0.00 · 11 · 4.44/– | 0.00 · 23 · 4.41/– | 0.00 · 45 · 4.40/– | 0.00 · 116 · 4.32/– |
| PB-ASYNC-r2 | 1 | solo_long | worker | 0.56 · 90 · 2.79/2.08 | 0.00 · 40 · 3.05/– | 0.00 · 81 · 3.05/– | 0.00 · 162 · 3.07/– | 0.00 · 274 · 3.22/– | 0.00 · 633 · 3.34/– |
| PB-ASYNC-r2 | 1 | solo_short | parent-active | 0.55 · 47 · 3.08/1.28 | 0.00 · 26 · 3.67/– | 0.00 · 52 · 3.68/1.00 | 0.00 · 103 · 3.67/– | 0.00 · 206 · 3.68/– | 0.00 · 512 · 3.66/– |
| PB-ASYNC-r2 | 1 | solo_short | chain | 0.55 · 47 · 3.08/1.28 | 0.00 · 26 · 3.67/– | 0.00 · 52 · 3.68/1.00 | 0.00 · 103 · 3.67/– | 0.00 · 206 · 3.68/– | 0.00 · 512 · 3.66/– |
| B-r1 | 0 | solo_short | parent-active | 0.53 · 50 · 3.15/1.10 | 0.00 · 26 · 3.68/– | 0.00 · 53 · 3.68/– | 0.00 · 103 · 3.67/– | 0.00 · 208 · 3.68/– | 0.00 · 514 · 3.68/– |
| B-r1 | 0 | solo_short | chain | 0.53 · 50 · 3.15/1.10 | 0.00 · 26 · 3.68/– | 0.00 · 53 · 3.68/– | 0.00 · 103 · 3.67/– | 0.00 · 208 · 3.68/– | 0.00 · 514 · 3.68/– |
| B-r1 | 0 | solo_long | parent-active | 0.53 · 14 · 3.14/1.26 | 0.00 · 6 · 4.45/– | 0.00 · 11 · 4.43/– | 0.00 · 23 · 4.44/– | 0.00 · 45 · 4.44/– | 0.00 · 115 · 4.43/– |
| B-r1 | 0 | solo_long | parent-other | 0.53 · 14 · 3.14/1.26 | 0.00 · 6 · 4.45/– | 0.00 · 11 · 4.43/– | 0.00 · 23 · 4.44/– | 0.00 · 45 · 4.44/– | 0.00 · 115 · 4.43/– |
| B-r1 | 0 | solo_long | worker | 0.45 · 94 · 2.43/1.97 | 0.00 · 40 · 3.11/– | 0.00 · 80 · 3.10/– | 0.00 · 161 · 3.09/– | 0.00 · 271 · 3.26/– | 0.00 · 633 · 3.37/– |
| B-r1 | 0 | hetero | parent-active | 0.95 · 120 · 1.99/1.58 | 1.00 · 156 · –/1.26 | 0.96 · 268 · 1.70/1.49 | 1.00 · 591 · –/1.38 | 0.99 · 1193 · 1.60/1.33 | 0.98 · 2830 · 2.60/1.33 |
| B-r1 | 0 | hetero | chain | 0.93 · 87 · 1.82/1.66 | 1.00 · 115 · –/1.32 | 0.95 · 203 · 1.68/1.54 | 1.00 · 446 · –/1.43 | 0.99 · 900 · 1.62/1.37 | 0.98 · 2129 · 2.54/1.38 |
| B-r1 | 0 | hetero | parent-other | 0.98 · 33 · 3.30/1.36 | 1.00 · 41 · –/1.09 | 0.98 · 65 · 1.79/1.34 | 1.00 · 145 · –/1.24 | 0.99 · 293 · 1.54/1.21 | 0.98 · 702 · 2.85/1.16 |
| B-r1 | 0 | hetero | worker | 0.86 · 143 · 1.98/1.72 | 1.00 · 146 · –/1.39 | 0.96 · 262 · 1.62/1.59 | 1.00 · 552 · –/1.50 | 0.99 · 1068 · 1.66/1.44 | 0.98 · 2347 · 2.54/1.43 |
| B-r1 | 1 | hetero | parent-active | 0.99 · 134 · 3.87/1.40 | 1.00 · 144 · –/1.37 | 1.00 · 289 · –/1.36 | 0.98 · 545 · 2.58/1.27 | 1.00 · 1181 · –/1.31 | 0.91 · 2319 · 3.62/1.28 |
| B-r1 | 1 | hetero | chain | 0.98 · 99 · 3.87/1.45 | 1.00 · 106 · –/1.44 | 1.00 · 216 · –/1.43 | 0.98 · 413 · 2.58/1.31 | 1.00 · 883 · –/1.36 | 0.90 · 1745 · 3.53/1.33 |
| B-r1 | 1 | hetero | parent-other | 1.00 · 36 · –/1.27 | 1.00 · 37 · –/1.17 | 1.00 · 73 · –/1.18 | 0.98 · 132 · 2.58/1.16 | 1.00 · 298 · –/1.14 | 0.92 · 574 · 3.99/1.13 |
| B-r1 | 1 | hetero | worker | 0.93 · 157 · 3.89/1.57 | 1.00 · 134 · –/1.50 | 1.00 · 273 · –/1.43 | 0.98 · 531 · 2.56/1.37 | 1.00 · 1127 · –/1.40 | 0.90 · 2136 · 3.27/1.39 |
| B-r1 | 1 | solo_long | parent-active | 0.18 · 9 · 3.24/2.34 | 0.00 · 6 · 4.44/– | 0.00 · 11 · 4.44/– | 0.00 · 23 · 4.44/– | 0.00 · 46 · 4.42/– | 0.00 · 114 · 4.44/– |
| B-r1 | 1 | solo_long | parent-other | 0.18 · 9 · 3.24/2.34 | 0.00 · 6 · 4.44/– | 0.00 · 11 · 4.44/– | 0.00 · 23 · 4.44/– | 0.00 · 46 · 4.42/– | 0.00 · 114 · 4.44/– |
| B-r1 | 1 | solo_long | worker | 0.32 · 89 · 2.36/1.97 | 0.00 · 40 · 3.10/– | 0.00 · 80 · 3.09/– | 0.00 · 160 · 3.08/– | 0.00 · 270 · 3.29/– | 0.00 · 634 · 3.36/– |
| B-r1 | 1 | solo_short | parent-active | 0.45 · 42 · 3.08/1.65 | 0.00 · 26 · 3.67/– | 0.00 · 53 · 3.66/– | 0.00 · 104 · 3.66/– | 0.00 · 205 · 3.67/– | 0.00 · 522 · 3.67/1.17 |
| B-r1 | 1 | solo_short | chain | 0.45 · 42 · 3.08/1.65 | 0.00 · 26 · 3.67/– | 0.00 · 53 · 3.66/– | 0.00 · 104 · 3.66/– | 0.00 · 205 · 3.67/– | 0.00 · 522 · 3.67/1.17 |
| O-r1 | 0 | solo_short | parent-active | 0.51 · 48 · 3.22/1.19 | 0.00 · 26 · 3.68/– | 0.00 · 53 · 3.67/– | 0.00 · 104 · 3.67/– | 0.00 · 205 · 3.68/– | 0.00 · 513 · 3.67/– |
| O-r1 | 0 | solo_short | chain | 0.51 · 48 · 3.22/1.19 | 0.00 · 26 · 3.68/– | 0.00 · 53 · 3.67/– | 0.00 · 104 · 3.67/– | 0.00 · 205 · 3.68/– | 0.00 · 513 · 3.67/– |
| O-r1 | 0 | solo_long | parent-active | 0.50 · 15 · 3.16/1.28 | 0.00 · 6 · 4.43/– | 0.00 · 11 · 4.44/– | 0.00 · 23 · 4.44/– | 0.00 · 50 · 4.28/– | 0.00 · 114 · 4.43/– |
| O-r1 | 0 | solo_long | parent-other | 0.50 · 15 · 3.16/1.28 | 0.00 · 6 · 4.43/– | 0.00 · 11 · 4.44/– | 0.00 · 23 · 4.44/– | 0.00 · 50 · 4.28/– | 0.00 · 114 · 4.43/– |
| O-r1 | 0 | solo_long | worker | 0.47 · 102 · 2.49/1.77 | 0.00 · 41 · 3.11/– | 0.00 · 79 · 3.08/– | 0.00 · 161 · 3.09/– | 0.00 · 278 · 3.33/– | 0.00 · 632 · 3.35/– |
| O-r1 | 0 | hetero | parent-active | 0.93 · 128 · 1.78/1.40 | 1.00 · 150 · –/1.34 | 1.00 · 296 · –/1.33 | 1.00 · 591 · –/1.34 | 0.10 · 301 · 3.83/1.19 | 0.00 · 666 · 3.90/– |
| O-r1 | 0 | hetero | chain | 0.92 · 99 · 1.73/1.40 | 1.00 · 112 · –/1.40 | 1.00 · 225 · –/1.37 | 1.00 · 443 · –/1.39 | 0.09 · 237 · 3.72/1.25 | 0.00 · 528 · 3.79/– |
| O-r1 | 0 | hetero | parent-other | 0.96 · 29 · 2.09/1.38 | 1.00 · 38 · –/1.15 | 1.00 · 71 · –/1.21 | 1.00 · 148 · –/1.17 | 0.12 · 63 · 4.26/1.02 | 0.00 · 138 · 4.34/– |
| O-r1 | 0 | hetero | worker | 0.86 · 164 · 1.83/1.49 | 1.00 · 141 · –/1.41 | 1.00 · 280 · –/1.45 | 1.00 · 550 · –/1.45 | 0.08 · 356 · 3.15/1.37 | 0.00 · 819 · 3.18/– |
| O-r1 | 1 | hetero | parent-active | 0.93 · 130 · 1.78/1.52 | 1.00 · 159 · –/1.24 | 1.00 · 291 · –/1.37 | 1.00 · 594 · –/1.32 | 0.99 · 1195 · 2.06/1.28 | 0.46 · 1045 · 3.88/1.28 |
| O-r1 | 1 | hetero | chain | 0.92 · 100 · 1.66/1.55 | 1.00 · 119 · –/1.27 | 1.00 · 222 · –/1.41 | 1.00 · 450 · –/1.36 | 0.99 · 900 · 2.01/1.33 | 0.45 · 811 · 3.76/1.32 |
| O-r1 | 1 | hetero | parent-other | 0.97 · 30 · 2.80/1.43 | 1.00 · 39 · –/1.15 | 1.00 · 68 · –/1.22 | 1.00 · 145 · –/1.17 | 0.99 · 295 · 2.26/1.13 | 0.50 · 233 · 4.31/1.15 |
| O-r1 | 1 | hetero | worker | 0.86 · 137 · 2.09/1.59 | 1.00 · 143 · –/1.31 | 1.00 · 276 · –/1.44 | 1.00 · 554 · –/1.43 | 0.99 · 988 · 2.25/1.41 | 0.43 · 952 · 3.42/1.38 |
| O-r1 | 1 | solo_long | parent-active | 0.55 · 14 · 3.09/1.26 | 0.00 · 6 · 4.36/– | 0.00 · 12 · 4.44/– | 0.00 · 23 · 4.44/– | 0.00 · 46 · 4.42/– | 0.00 · 115 · 4.44/– |
| O-r1 | 1 | solo_long | parent-other | 0.55 · 14 · 3.09/1.26 | 0.00 · 6 · 4.36/– | 0.00 · 12 · 4.44/– | 0.00 · 23 · 4.44/– | 0.00 · 46 · 4.42/– | 0.00 · 115 · 4.44/– |
| O-r1 | 1 | solo_long | worker | 0.47 · 95 · 2.34/1.75 | 0.00 · 41 · 3.18/– | 0.00 · 80 · 3.09/– | 0.00 · 161 · 3.08/– | 0.00 · 271 · 3.28/– | 0.00 · 630 · 3.36/– |
| O-r1 | 1 | solo_short | parent-active | 0.52 · 45 · 3.12/1.29 | 0.00 · 26 · 3.70/– | 0.00 · 52 · 3.67/– | 0.00 · 104 · 3.67/– | 0.00 · 210 · 3.66/– | 0.00 · 520 · 3.68/0.98 |
| O-r1 | 1 | solo_short | chain | 0.52 · 45 · 3.12/1.29 | 0.00 · 26 · 3.70/– | 0.00 · 52 · 3.67/– | 0.00 · 104 · 3.67/– | 0.00 · 210 · 3.66/– | 0.00 · 520 · 3.68/0.98 |
| Q-r1 | 0 | solo_short | parent-active | 0.32 · 40 · 3.12/1.15 | 0.00 · 26 · 3.67/– | 0.00 · 52 · 3.76/– | 0.00 · 105 · 3.68/– | 0.00 · 203 · 3.65/– | 0.00 · 513 · 3.65/– |
| Q-r1 | 0 | solo_short | chain | 0.32 · 40 · 3.12/1.15 | 0.00 · 26 · 3.67/– | 0.00 · 52 · 3.76/– | 0.00 · 105 · 3.68/– | 0.00 · 203 · 3.65/– | 0.00 · 513 · 3.65/– |
| Q-r1 | 0 | solo_long | parent-active | 0.53 · 13 · 3.07/1.32 | 0.00 · 6 · 4.35/– | 0.00 · 12 · 4.43/– | 0.00 · 23 · 4.44/– | 0.00 · 45 · 4.44/– | 0.00 · 115 · 4.40/– |
| Q-r1 | 0 | solo_long | parent-other | 0.53 · 13 · 3.07/1.32 | 0.00 · 6 · 4.35/– | 0.00 · 12 · 4.43/– | 0.00 · 23 · 4.44/– | 0.00 · 45 · 4.44/– | 0.00 · 115 · 4.40/– |
| Q-r1 | 0 | solo_long | worker | 0.49 · 98 · 2.38/1.87 | 0.00 · 42 · 3.12/– | 0.00 · 81 · 3.08/– | 0.00 · 161 · 3.07/– | 0.00 · 269 · 3.28/– | 0.00 · 625 · 3.38/– |
| Q-r1 | 0 | hetero | parent-active | 0.93 · 124 · 1.87/1.49 | 1.00 · 141 · –/1.38 | 0.92 · 266 · 1.78/1.43 | 0.00 · 133 · 3.89/– | 0.00 · 265 · 3.89/– | 0.00 · 658 · 3.90/– |
| Q-r1 | 0 | hetero | chain | 0.92 · 92 · 1.79/1.53 | 1.00 · 105 · –/1.44 | 0.91 · 202 · 1.76/1.48 | 0.00 · 105 · 3.78/– | 0.00 · 210 · 3.78/– | 0.00 · 523 · 3.79/– |
| Q-r1 | 0 | hetero | parent-other | 0.97 · 32 · 2.35/1.39 | 1.00 · 35 · –/1.21 | 0.94 · 64 · 1.88/1.27 | 0.00 · 27 · 4.30/– | 0.00 · 55 · 4.33/– | 0.00 · 135 · 4.33/– |
| Q-r1 | 0 | hetero | worker | 0.86 · 151 · 1.85/1.65 | 1.00 · 131 · –/1.47 | 0.93 · 258 · 1.80/1.56 | 0.00 · 164 · 3.17/– | 0.00 · 277 · 3.37/– | 0.00 · 652 · 3.41/– |
| Q-r1 | 1 | hetero | parent-active | 0.94 · 121 · 2.04/1.38 | 1.00 · 145 · –/1.39 | 1.00 · 296 · –/1.31 | 1.00 · 597 · –/1.27 | 0.99 · 1174 · 1.80/1.31 | 0.99 · 2917 · 1.93/1.34 |
| Q-r1 | 1 | hetero | chain | 0.92 · 87 · 1.92/1.45 | 1.00 · 107 · –/1.47 | 1.00 · 221 · –/1.36 | 1.00 · 449 · –/1.31 | 0.99 · 880 · 1.77/1.37 | 0.99 · 2198 · 1.90/1.39 |
| Q-r1 | 1 | hetero | parent-other | 0.98 · 34 · 2.99/1.21 | 1.00 · 38 · –/1.14 | 1.00 · 76 · –/1.14 | 1.00 · 148 · –/1.13 | 0.99 · 294 · 1.93/1.13 | 1.00 · 719 · 2.13/1.19 |
| Q-r1 | 1 | hetero | worker | 0.85 · 150 · 2.03/1.59 | 1.00 · 138 · –/1.51 | 1.00 · 281 · –/1.40 | 1.00 · 571 · –/1.38 | 0.99 · 1020 · 1.82/1.42 | 1.00 · 2431 · 2.09/1.45 |
| Q-r1 | 1 | solo_long | parent-active | 0.01 · 10 · 2.82/1.77 | 0.00 · 6 · 4.45/– | 0.00 · 11 · 4.44/– | 0.00 · 23 · 4.44/– | 0.00 · 46 · 4.44/– | 0.00 · 115 · 4.42/– |
| Q-r1 | 1 | solo_long | parent-other | 0.01 · 10 · 2.82/1.77 | 0.00 · 6 · 4.45/– | 0.00 · 11 · 4.44/– | 0.00 · 23 · 4.44/– | 0.00 · 46 · 4.44/– | 0.00 · 115 · 4.42/– |
| Q-r1 | 1 | solo_long | worker | 0.00 · 66 · 2.50/– | 0.00 · 40 · 3.06/– | 0.00 · 80 · 3.06/– | 0.00 · 161 · 3.05/– | 0.00 · 324 · 3.06/– | 0.00 · 810 · 3.05/– |
| Q-r1 | 1 | solo_short | parent-active | 0.49 · 42 · 2.97/1.69 | 0.00 · 26 · 3.68/– | 0.00 · 52 · 3.68/– | 0.00 · 104 · 3.66/– | 0.00 · 209 · 3.67/– | 0.00 · 516 · 3.67/– |
| Q-r1 | 1 | solo_short | chain | 0.49 · 42 · 2.97/1.69 | 0.00 · 26 · 3.68/– | 0.00 · 52 · 3.68/– | 0.00 · 104 · 3.66/– | 0.00 · 209 · 3.67/– | 0.00 · 516 · 3.67/– |

## Limits

- Only the sampled threads: the listed parent threads (MainThread, client-short, client-long, laya-ane-dispatch, laya-gpu-dispatch, coreml-callback where it exists) and the auto instance's GPU worker. Other parent threads and the GPU-only instance are not in the data.
- Not system-wide: other processes and core occupancy are not observed.
- No cause is observable: these counters show where threads ran, not why they were placed there.
- Post-hoc on #96 and #99 raw data; it changes no earlier verdict.
