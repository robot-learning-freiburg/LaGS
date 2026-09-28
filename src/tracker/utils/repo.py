# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import git


@dataclass
class State:
    sha: str
    diff_staged: str
    diff_unstaged: str
    diff_untracked: str

    def as_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)


def git_repo() -> git.Repo:
    """
    Get the git repo object associated with this project.
    """
    return git.Repo(Path(__file__).parent, search_parent_directories=True)


def git_sha(repo: git.Repo = None) -> str:
    """
    Get the SHA of the current HEAD.
    """
    repo = repo or git_repo()

    return repo.head.commit.hexsha


def git_diff_unstaged(repo: git.Repo = None) -> git.DiffIndex:
    """
    Get a diff (text) of all currently unstaged changes.
    """
    repo = repo or git_repo()

    return repo.git.diff(strip_newline_in_stdout=False)


def git_diff_staged(repo: git.Repo = None) -> git.DiffIndex:
    """
    Get a diff (text) of all currently staged changes.
    """
    repo = repo or git_repo()

    return repo.git.diff("--staged", strip_newline_in_stdout=False)


def git_untracked(repo: git.Repo = None) -> List[str]:
    """
    Get a list of all currently untracked files, excluding files specified in
    .gitignore.
    """
    repo = repo or git_repo()

    return repo.untracked_files


def git_diff_untracked(repo: git.Repo = None) -> str:
    """
    Get a diff (text) adding/creating all currently untracked files, excluding
    files specified in .gitignore.
    """
    repo = repo or git_repo()

    diff = ""
    for file in git_untracked(repo):
        diff += repo.git.diff(
            "/dev/null", file, with_exceptions=False, strip_newline_in_stdout=False
        )

    return diff


def git_state(repo: git.Repo = None) -> State:
    """
    Get the current state of the repo, including SHA of the HEAD and diff texts
    for staged, unstaged, and untracked files.
    """
    repo = repo or git_repo()

    sha = git_sha(repo)
    diff_staged = git_diff_staged(repo)
    diff_unstaged = git_diff_unstaged(repo)
    diff_untracked = git_diff_untracked(repo)

    return State(
        sha=sha,
        diff_staged=diff_staged,
        diff_unstaged=diff_unstaged,
        diff_untracked=diff_untracked,
    )
