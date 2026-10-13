#!/bin/bash
# Fetch open fonts for realistic Chromium rendering during the crawl (OFL / SIL licensed).
set -u
cd "$(dirname "$0")"
mv -f noto.zip NotoSans-Regular.ttf 2>/dev/null || true
B=https://github.com/notofonts/notofonts.github.io/raw/main/fonts
for p in NotoSans/hinted/ttf/NotoSans-Bold.ttf NotoSans/hinted/ttf/NotoSans-Italic.ttf NotoSans/hinted/ttf/NotoSans-SemiBold.ttf \
         NotoSans/hinted/ttf/NotoSans-Medium.ttf NotoSerif/hinted/ttf/NotoSerif-Regular.ttf NotoSerif/hinted/ttf/NotoSerif-Bold.ttf \
         NotoSansMono/hinted/ttf/NotoSansMono-Regular.ttf NotoSansSymbols2/hinted/ttf/NotoSansSymbols2-Regular.ttf \
         NotoSansSymbols/hinted/ttf/NotoSansSymbols-Regular.ttf; do
  curl -sfL -m 60 -O "$B/$p" || echo "fail $p"
done
curl -sfL -m 120 -O https://github.com/googlefonts/noto-emoji/raw/main/fonts/NotoColorEmoji.ttf || echo "emoji fail"
cat > fonts.conf <<EOF
<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <dir>/usr/share/fonts</dir>
  <dir>$PWD</dir>
  <cachedir>$PWD/cache</cachedir>
  <alias><family>sans-serif</family><prefer><family>Liberation Sans</family><family>Noto Sans</family></prefer></alias>
  <alias><family>serif</family><prefer><family>Liberation Serif</family><family>Noto Serif</family></prefer></alias>
  <alias><family>monospace</family><prefer><family>Liberation Mono</family></prefer></alias>
  <alias><family>Helvetica</family><prefer><family>Liberation Sans</family></prefer></alias>
  <alias><family>Arial</family><prefer><family>Liberation Sans</family></prefer></alias>
  <alias><family>Times New Roman</family><prefer><family>Liberation Serif</family></prefer></alias>
</fontconfig>
EOF
FONTCONFIG_FILE=$PWD/fonts.conf fc-list | wc -l
