# Lorito

*Spanish for "little parrot". It types what you say, offline.*

Floating dictation pill. Click it (or run `lorito toggle`) to start/stop
recording; speech is transcribed locally with NVIDIA Parakeet TDT v3 (25 European
languages, punctuation included) and typed into the focused window
(Shift+Insert via the RemoteDesktop portal: apps paste the clipboard, terminals the
primary selection; both are restored afterwards).

States: `🎤` idle · live level bars while recording · grey wave while loading/transcribing · flat orange dots on error (hover for details).
Hover the pill to see the last transcript. Right-click quits.

## Setup (everything stays in this folder)

```bash
d=models/parakeet-tdt-0.6b-v3-int8 && mkdir -p $d && for f in encoder-model.int8.onnx \
  decoder_joint-model.int8.onnx nemo128.onnx vocab.txt; do curl -fL -o $d/$f \
  https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/$f; done
mv .cargo/config.toml /tmp/cfg && cargo vendor vendor && mv /tmp/cfg .cargo/config.toml
cargo build --release   # first build downloads ONNX Runtime into .tools/ort
```

App menu entry: `sed "s|@DIR@|$PWD|" lorito.desktop > ~/.local/share/applications/lorito.desktop`

Runtime needs `pw-record` (PipeWire) and wl-clipboard (`wl-copy`/`wl-paste`). Other model dir: `LORITO_MODEL=/path`.

## Auto-paste permission

Keystrokes go through the RemoteDesktop portal: on first start KDE asks
"Allow remote control?" (keyboard only). The grant is saved in `.portal-token`;
revoke it in System Settings → Application Permissions. If refused, text is only copied.

Known gap: Dolphin's embedded terminal (F4) treats Shift+Insert as ambiguous and
pastes nothing. Fix if needed: detect the focused widget via AT-SPI.

## Pill must not take focus (KDE, one-time)

Wayland has no way for a normal window to refuse focus, so a KWin rule does it
(otherwise clicking the pill makes the paste land in the pill):
System Settings → Window Management → Window Rules → Add New → Window class
`lorito` (exact) → Accept focus: Force/No, Keep above: Force/Yes, Skip taskbar: Force/Yes.

## Global shortcut (KDE)

System Settings → Shortcuts → Add New → Command:
`/path/to/Lorito/target/release/lorito toggle`.
Without the KWin rule above, use the shortcut rather than clicking: a click focuses the pill, so the paste would land there.
