#!/bin/sh
set -eu

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

release_target() {
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64 | Darwin-aarch64) printf 'aarch64-apple-darwin\n' ;;
    Linux-x86_64) printf 'x86_64-unknown-linux-gnu\n' ;;
    *) fail "ming has release builds for macOS on Apple silicon and Linux on x86-64; build it from source elsewhere" ;;
  esac
}

verify_archive() {
  actual=$(shasum -a 256 "$1")
  expected=$(cut -d ' ' -f 1 "$1.sha256")
  [ "$expected" = "${actual%% *}" ] || fail "checksum mismatch; download the release again"
}

release_version() {
  version=${1:-}
  if [ -z "$version" ]; then
    latest=$(curl -fsSL -o /dev/null -w '%{url_effective}' "$releases/latest") ||
      fail "cannot find the latest release; check $releases or specify a version"
    case "$latest" in
      "$releases"/tag/v*) version=${latest##*/} ;;
      *) fail "no release found; check $releases" ;;
    esac
  fi
  version=${version#v}
  case "$version" in
    '' | *[!0-9A-Za-z.+-]*) fail "invalid version; use a release version such as v0.1.0" ;;
  esac
  printf '%s\n' "$version"
}

main() {
  [ "$#" -le 1 ] || fail "usage: sh install.sh [VERSION]"
  releases="https://github.com/fusor-rs/ming-wasm/releases"
  target=$(release_target)
  version=$(release_version "${1:-}")
  name="ming-$version-$target"
  archive="$name.tar.gz"
  for tool in curl tar shasum; do
    command -v "$tool" >/dev/null 2>&1 || fail "install $tool before running this installer"
  done
  url="${MING_DOWNLOAD_BASE:-$releases/download}/v$version/$archive"
  root=${MING_INSTALL:-$HOME/.ming}
  umask 022
  mkdir -p "$root/bin"
  temporary=$(mktemp -d "$root/.install.XXXXXX")
  trap 'rm -r "$temporary"' 0
  trap 'exit 1' HUP INT TERM

  printf 'Downloading ming v%s for %s\n' "$version" "$target"
  curl -fsSL "$url" -o "$temporary/$archive" || fail "cannot download $url"
  curl -fsSL "$url.sha256" -o "$temporary/$archive.sha256" ||
    fail "cannot download $url.sha256"
  verify_archive "$temporary/$archive"
  tar -xzf "$temporary/$archive" -C "$temporary"
  unpacked="$temporary/$name"
  [ -f "$unpacked/bin/ming" ] && [ ! -L "$unpacked/bin/ming" ] ||
    fail "archive has no regular bin/ming"
  installed_version=$("$unpacked/bin/ming" --version) ||
    fail "downloaded ming could not run"
  [ "$installed_version" = "ming $version" ] ||
    fail "downloaded ming has the wrong version"
  mv -f "$unpacked/bin/ming" "$root/bin/ming"
  printf 'Installed ming v%s to %s\n' "$version" "$root"
  case ":$PATH:" in
    *":$root/bin:"*) ;;
    *) printf 'Add %s to PATH in your shell profile.\n' "$root/bin" ;;
  esac
}

main "$@"
