"""Plan test 7: the course data on disk is the version we pinned (risk R15).

If staff push an update, ./setup_data.sh never pulls it silently; this test fails instead
if the clone and course_commit.txt ever disagree, or if files in the clone were edited.
"""

import subprocess

from conftest import COURSE_ROOT

import validity


def test_course_clone_is_at_the_pinned_commit(needs_data):
    head = subprocess.run(
        ["git", "-C", COURSE_ROOT, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert head == validity.course_pin(), (
        "run ./setup_data.sh to restore the pinned version"
    )


def test_course_clone_has_no_local_edits(needs_data):
    edited = subprocess.run(
        ["git", "-C", COURSE_ROOT, "diff", "--quiet", "HEAD"], check=False
    ).returncode  # 1 means some tracked file differs from the pinned commit
    assert edited == 0, (
        "files in data/cp2-repo were edited; discard the edits (staged ones too) with "
        "`git -C data/cp2-repo reset --hard`"
    )
