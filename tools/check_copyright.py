# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Check/fix copyright year in file headers."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import os
import subprocess
import sys


class CopyrightChecker:
    """Check copyright year in file headers."""

    _COPYRIGHT_STRING = "# (C) Copyright IBM "
    _YEAR_LENGTH = 4
    _YEAR_PREFIX = "20"

    def __init__(self, root_dir: str, check: bool) -> None:
        """Initialize the checker for files under ``root_dir``."""
        self._root_dir = root_dir
        self._check = check
        self._current_year = datetime.datetime.now().year
        # Files modified in the working tree are treated as "last changed this year",
        # so local -check results can differ from a clean-checkout CI run.
        self._changed_files = self._get_changed_files()

    @staticmethod
    def _get_year_from_date(date: str) -> int | None:
        if not date or len(date) < CopyrightChecker._YEAR_LENGTH:
            return None

        return int(date[: CopyrightChecker._YEAR_LENGTH])

    def _cmd_execute(self, args: list[str]) -> tuple[str, str | None]:
        env = {}
        for key in ["SYSTEMROOT", "PATH"]:
            value = os.environ.get(key)
            if value is not None:
                env[key] = value
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        with subprocess.Popen(
            args,
            cwd=self._root_dir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as popen:
            out, err = popen.communicate()
            popen.wait()
            out_str = out.decode("utf-8").strip()
            decoded_err = err.decode("utf-8").strip()
            err_str: str | None = decoded_err if decoded_err else None
            return out_str, err_str

    def _get_changed_files(self) -> list[str]:
        out_str, err_str = self._cmd_execute(["git", "diff", "--name-only", "HEAD"])
        if err_str:
            raise RuntimeError(err_str)

        return out_str.splitlines()

    def _get_file_last_year(self, relative_path: str) -> int | None:
        out_str, err_str = self._cmd_execute(["git", "log", "-1", "--format=%cI", relative_path])
        if err_str:
            raise RuntimeError(f"'{relative_path}' last year: {err_str}")

        return CopyrightChecker._get_year_from_date(out_str)

    @staticmethod
    def _parse_header_years(line: str) -> tuple[int | None, int | None]:
        """Parse the start and last year out of a copyright header line."""
        years = []
        for word in line.strip().split():
            for year in word.strip().split(","):
                if year.startswith(CopyrightChecker._YEAR_PREFIX) and (
                    len(year) >= CopyrightChecker._YEAR_LENGTH
                ):
                    with contextlib.suppress(ValueError):
                        years.append(int(year[: CopyrightChecker._YEAR_LENGTH]))

        if len(years) > 1:
            return years[0], years[1]
        if len(years) == 1:
            return years[0], years[0]
        return None, None

    @staticmethod
    def _expected_header(start_year: int | None, last_year: int) -> str:
        """Build the expected copyright header line ending in ``last_year``."""
        header = CopyrightChecker._COPYRIGHT_STRING
        if start_year and start_year != last_year:
            header += f"{start_year}, "
        header += f"{last_year}.\n"
        return header

    @staticmethod
    def _rewrite_line(file_path: str, file_lines: list[str], idx: int, new_line: str) -> None:
        """Replace line ``idx`` in ``file_path`` with ``new_line`` and save the file."""
        file_lines[idx] = new_line
        with open(file_path, "w", encoding="utf8") as file:
            file.writelines(file_lines)

    def check_copyright(self, file_path: str) -> tuple[bool, bool]:
        """Check (and optionally fix) the copyright year for a single file."""
        file_with_invalid_year = False
        file_has_header = False
        relative_path = os.path.relpath(file_path, self._root_dir)
        try:
            new_line = ""
            idx_new_line = -1
            with open(file_path, encoding="utf8") as file:
                file_lines = file.readlines()
            for idx, line in enumerate(file_lines):
                if not line.startswith(CopyrightChecker._COPYRIGHT_STRING):
                    continue

                file_has_header = True
                header_start_year, header_last_year = CopyrightChecker._parse_header_years(line)

                # last_year is the single source of truth: it drives both the
                # "is this wrong" check below and the replacement header text,
                # so -check and the fixer can never disagree on the target year.
                last_year: int | None
                if relative_path in self._changed_files:
                    self._changed_files.remove(relative_path)
                    last_year = self._current_year
                else:
                    last_year = self._get_file_last_year(relative_path)

                if last_year and header_last_year != last_year:
                    new_line = CopyrightChecker._expected_header(header_start_year, last_year)
                    if self._check:
                        print(
                            f"Wrong Copyright Year: '{relative_path}': "
                            f"Current: '{line[:-1]}' Correct: '{new_line[:-1]}'"
                        )
                    file_with_invalid_year = True
                    idx_new_line = idx

                break

            if not self._check and idx_new_line >= 0:
                CopyrightChecker._rewrite_line(file_path, file_lines, idx_new_line, new_line)
                file_with_invalid_year = False
                print(f"Fixed copyright year for {relative_path}.")

        except UnicodeDecodeError:
            return file_with_invalid_year, file_has_header

        return file_with_invalid_year, file_has_header

    def check(self) -> tuple[int, int]:
        """Check/fix copyright years for all git-tracked files under ``root_dir``."""
        files_with_invalid_year = 0
        files_with_header = 0
        # Use git ls-files rather than walking the directory tree, so that nested
        # git repositories (e.g. vendored subdirectories) and untracked/build
        # artifacts are never scanned.
        out_str, err_str = self._cmd_execute(["git", "ls-files"])
        if err_str:
            raise RuntimeError(err_str)

        for relative_path in out_str.splitlines():
            fullpath = os.path.join(self._root_dir, relative_path)
            if not os.path.isfile(fullpath):
                continue

            file_with_invalid_year, file_has_header = self.check_copyright(fullpath)
            if file_with_invalid_year:
                files_with_invalid_year += 1
            if file_has_header:
                files_with_header += 1

        return files_with_invalid_year, files_with_header


def check_path(path: str) -> str:
    """Validate the path argument."""
    if not path or os.path.isdir(path):
        return path

    raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Check Copyright Tool")
    PARSER.add_argument("-path", type=check_path, metavar="path", help="Root path of project.")
    PARSER.add_argument(
        "-check",
        required=False,
        action="store_true",
        help="Just check copyright, without fixing it.",
    )

    ARGS = PARSER.parse_args()
    if not ARGS.path:
        ARGS.path = os.getcwd()

    ARGS.path = os.path.abspath(os.path.realpath(os.path.expanduser(ARGS.path)))
    INVALID_YEAR, HAS_HEADER = CopyrightChecker(ARGS.path, ARGS.check).check()
    print(f"{INVALID_YEAR} of {HAS_HEADER} files with copyright header have wrong years.")

    sys.exit(0 if INVALID_YEAR == 0 else 1)
