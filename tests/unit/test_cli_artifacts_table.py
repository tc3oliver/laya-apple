"""`laya-apple artifacts list --format table`: a scannable view of `artifact_capabilities()`,
and the JSON output (the default) staying byte-identical."""

from __future__ import annotations

import json

import pytest

from laya_apple import cli

EMPTY = (
    "no artifacts registered; fetch prebuilt ones with `laya-apple artifacts fetch MODEL` "
    "or build them with `laya-apple artifacts build MODEL`\n"
)


def record(model="laya-typed-decisions", bucket=128, status="validated", passed=True, revision="r1", **target):
    """A record shaped like `artifact_capabilities()`'s, with only the fields the table reads."""
    return {
        "path": f"/cache/{model}/{bucket}",
        "status": status,
        "model": model,
        "revision": revision,
        "bucket": bucket,
        "compute_target": {
            "compute_units": "CPU_AND_NE",
            "ops": {"ane": 412, "gpu": 0, "cpu": 0},
            "transitions": 0,
        }
        | target,
        "parity": {"passed": passed},
    }


@pytest.fixture
def capabilities(monkeypatch):
    """Make `artifact_capabilities()` return what the test sets with `capabilities(...)`."""

    def use(records):
        monkeypatch.setattr("laya_apple.artifacts.artifact_capabilities", lambda: records)

    return use


def run(capsys, *argv):
    a = cli.build_parser().parse_args(["artifacts", "list", *argv])
    assert cli.cmd_artifacts(a) == 0
    return capsys.readouterr().out


def test_table_with_an_empty_cache_says_so_and_how_to_fill_it(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path))
    assert run(capsys, "--format", "table") == EMPTY
    assert run(capsys, "--capabilities", "--format", "table") == EMPTY


def test_table_with_one_artifact(capabilities, capsys):
    capabilities([record()])
    assert run(capsys, "--format", "table") == (
        "MODEL                 BUCKET  STATUS     PARITY PASSED  COMPUTE PLAN\n"
        "laya-typed-decisions     128  validated  yes            CPU_AND_NE (ane=412 gpu=0 cpu=0 transitions=0)\n"
    )


def test_table_with_many_artifacts_aligns_every_column(capabilities, capsys):
    capabilities(
        [
            record("laya", 64),
            record("laya-typed-decisions", 1024, passed=False),
            record("laya-multilingual", 128, status="building", passed=None),
        ]
    )
    assert run(capsys, "--format", "table") == (
        "MODEL                 BUCKET  STATUS     PARITY PASSED  COMPUTE PLAN\n"
        "laya                      64  validated  yes            CPU_AND_NE (ane=412 gpu=0 cpu=0 transitions=0)\n"
        "laya-multilingual        128  building   -              CPU_AND_NE (ane=412 gpu=0 cpu=0 transitions=0)\n"
        "laya-typed-decisions    1024  validated  no             CPU_AND_NE (ane=412 gpu=0 cpu=0 transitions=0)\n"
    )


def rows(table):
    """(model, bucket) of each data row of the table text."""
    return [(line.split()[0], int(line.split()[1])) for line in table.splitlines()[1:]]


def test_table_rows_are_sorted_by_model_revision_and_numeric_bucket(capabilities, capsys):
    registry_order = [
        record("laya-typed-decisions", 512),
        record("laya", 512, revision="b"),
        record("laya-typed-decisions", 64),
        record("laya", 64, revision="b"),
        record("laya", 1024, revision="a"),
        record("laya-typed-decisions", 128),
        record("laya", 128, revision="b"),
        record("laya", 64, revision="a"),
    ]
    capabilities(registry_order)
    # 64 < 128 < 512 < 1024 as numbers; a string sort would put 1024 first and 64 last
    assert rows(run(capsys, "--format", "table")) == [
        ("laya", 64),  # revision a
        ("laya", 1024),  # revision a
        ("laya", 64),  # revision b
        ("laya", 128),
        ("laya", 512),
        ("laya-typed-decisions", 64),
        ("laya-typed-decisions", 128),
        ("laya-typed-decisions", 512),
    ]
    # the JSON output is not sorted: it keeps the registry order
    assert [r["path"] for r in json.loads(run(capsys, "--capabilities"))] == [r["path"] for r in registry_order]


def test_table_sorts_buckets_of_one_model_numerically(capabilities, capsys):
    capabilities([record("laya", 512), record("laya", 64), record("laya", 128)])
    assert [bucket for _, bucket in rows(run(capsys, "--format", "table"))] == [64, 128, 512]


def test_table_sorts_what_a_manifest_leaves_out_last(capabilities, capsys):
    capabilities([record(model=None, bucket=None), record("laya", "128"), record("laya", None), record("laya", 64)])
    lines = run(capsys, "--format", "table").splitlines()[1:]
    assert [line.split()[:2] for line in lines] == [["laya", "64"], ["laya", "128"], ["laya", "-"], ["-", "-"]]


def test_table_is_the_same_with_and_without_capabilities(capabilities, capsys):
    capabilities([record("laya", 64), record("laya-typed-decisions", 1024)])
    assert run(capsys, "--format", "table") == run(capsys, "--capabilities", "--format", "table")


def test_table_shows_a_dash_for_everything_a_manifest_leaves_out(capabilities, capsys):
    unreadable = {"model": None, "bucket": None, "status": "unreadable", "compute_target": None, "parity": {}}
    capabilities([unreadable, record(compute_units=None, ops=None, transitions=None)])
    assert run(capsys, "--format", "table") == (
        "MODEL                 BUCKET  STATUS      PARITY PASSED  COMPUTE PLAN\n"
        "laya-typed-decisions     128  validated   yes            -\n"
        "-                          -  unreadable  -              -\n"
    )


def test_table_replaces_control_characters_with_a_question_mark(capabilities, capsys):
    capabilities(
        [record("la\nya", status="val\x1b[31midated", compute_units="CPU\tNE\x07", ops=None, transitions=None)]
    )
    out = run(capsys, "--format", "table")
    assert out == (
        "MODEL  BUCKET  STATUS          PARITY PASSED  COMPUTE PLAN\n"
        "la?ya     128  val?[31midated  yes            CPU?NE?\n"
    )
    assert not {"\x1b", "\x07", "\t"} & set(out)
    assert len(out.splitlines()) == 2


def test_table_pads_by_display_width_not_by_code_points(capabilities, capsys):
    # two fullwidth characters take 4 columns; "e" + a combining accent takes 1
    capabilities([record("laya", 64), record("雷亞-模型", 128, status="ok"), record("café", 256)])
    out = run(capsys, "--format", "table")
    assert out == (
        "MODEL      BUCKET  STATUS     PARITY PASSED  COMPUTE PLAN\n"
        "café          256  validated  yes            CPU_AND_NE (ane=412 gpu=0 cpu=0 transitions=0)\n"
        "laya           64  validated  yes            CPU_AND_NE (ane=412 gpu=0 cpu=0 transitions=0)\n"
        "雷亞-模型     128  ok         yes            CPU_AND_NE (ane=412 gpu=0 cpu=0 transitions=0)\n"
    )


@pytest.mark.parametrize(
    ("target", "plan"),
    [
        ({"compute_units": "CPU_AND_NE", "ops": None, "transitions": None}, "CPU_AND_NE"),
        ({"compute_units": None, "ops": {"ane": 3}, "transitions": 1}, "(ane=3 transitions=1)"),
        ({"compute_units": "ALL", "ops": {"ane": 3, "cpu": 2}, "transitions": None}, "ALL (ane=3 cpu=2)"),
        ({"compute_units": "ALL", "ops": "unexpected", "transitions": 2}, "ALL (transitions=2)"),
    ],
)
def test_table_compute_plan_uses_only_what_the_manifest_records(capabilities, capsys, target, plan):
    capabilities([record(**target)])
    (_, row) = run(capsys, "--format", "table").splitlines()
    assert row.endswith(f"yes            {plan}")


def test_table_survives_a_hand_edited_manifest(capabilities, capsys):
    weird = record() | {"parity": {"passed": ["not", "a", "bool"]}, "status": 7}
    capabilities([weird])
    assert run(capsys, "--format", "table").splitlines()[1].split()[:4] == ["laya-typed-decisions", "128", "7", "-"]


def test_the_default_output_stays_the_same_json(capabilities, capsys):
    records = [record("laya", 64), record("laya-typed-decisions", 1024, passed=False)]
    capabilities(records)
    expected = json.dumps(records, indent=1, ensure_ascii=False, default=str) + "\n"
    assert run(capsys, "--capabilities") == expected
    assert run(capsys, "--capabilities", "--format", "json") == expected


def test_the_default_plain_list_stays_the_same_json(monkeypatch, capsys):
    listed = [
        {
            "path": "/cache/laya-typed-decisions/128",
            "manifest": {
                "status": "validated",
                "source": {"model": "laya-typed-decisions"},
                "artifact": {"length": 128},
                "parity": {"passed": True},
                "integrity": {"artifact_sha256": "a" * 64},
            },
        }
    ]
    monkeypatch.setattr("laya_apple.artifacts.list_artifacts", lambda: listed)
    expected = (
        "[\n"
        " {\n"
        '  "path": "/cache/laya-typed-decisions/128",\n'
        '  "status": "validated",\n'
        '  "source": {\n'
        '   "model": "laya-typed-decisions"\n'
        "  },\n"
        '  "artifact": {\n'
        '   "length": 128\n'
        "  },\n"
        '  "parity_passed": true,\n'
        f'  "sha256": "{"a" * 64}"\n'
        " }\n"
        "]\n"
    )
    assert run(capsys) == expected
    assert run(capsys, "--format", "json") == expected


def test_an_unknown_format_is_an_argument_error(capsys):
    with pytest.raises(SystemExit) as e:
        cli.build_parser().parse_args(["artifacts", "list", "--format", "yaml"])
    assert e.value.code == 2
    assert "invalid choice: 'yaml'" in capsys.readouterr().err
