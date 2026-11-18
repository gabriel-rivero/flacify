#!/bin/sh
# Build Flacify.app around the release binary.
#
# One copy of the bundle layout, used by the release workflow and by anyone testing
# locally, so the plist that ships is the plist that was tested. The microphone usage
# string is not decoration: macOS kills or silently starves a process that opens an input
# device without it, and a launch from a terminal inherits the terminal's permission
# instead of the app's.
#
#     sh packaging/macos-app.sh [version] [binary]
#
# Version defaults to the workspace Cargo.toml, binary to target/release/flacify.
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

version=${1:-$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)}
binary=${2:-target/release/flacify}
test -n "$version" || { echo "no version in Cargo.toml" >&2; exit 1; }
test -x "$binary" || { echo "$binary is not built" >&2; exit 1; }

app=target/release/Flacify.app
rm -rf "$app"
mkdir -p "$app/Contents/MacOS"
cp "$binary" "$app/Contents/MacOS/flacify"
chmod +x "$app/Contents/MacOS/flacify"

cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>flacify</string>
  <key>CFBundleDisplayName</key><string>flacify</string>
  <key>CFBundleIdentifier</key><string>com.gabrielrivero.flacify</string>
  <key>CFBundleVersion</key><string>$version</string>
  <key>CFBundleShortVersionString</key><string>$version</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>flacify</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>flacify captures the audio you route to it, enhances it, and sends it to the output device you chose.</string>
</dict>
</plist>
PLIST

grep -q "<string>$version</string>" "$app/Contents/Info.plist" || {
    echo "the plist did not take $version" >&2
    exit 1
}
echo "$app ($version)"
