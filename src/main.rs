//! Lorito: floating dictation button. Click (or `lorito toggle`) to
//! start/stop recording; audio is transcribed locally with Parakeet TDT v3 (ONNX)
//! and the text is pasted into the focused window via the RemoteDesktop portal.

use ashpd::desktop::remote_desktop::{DeviceType, KeyState, RemoteDesktop, SelectDevicesOptions};
use ashpd::desktop::PersistMode;
use eframe::egui;
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::mpsc::{channel, Receiver, Sender};
use transcribe_rs::onnx::parakeet::{ParakeetModel, ParakeetParams};
use transcribe_rs::onnx::Quantization;

fn runtime_path(name: &str) -> PathBuf {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(name)
}

enum State {
    Loading,
    Idle,
    Recording(Child),
    Transcribing,
    Error(String),
}

struct App {
    state: State,
    last_text: String,
    jobs: Sender<PathBuf>,
    results: Receiver<Result<String, String>>,
    toggles: Receiver<()>,
    paste: Sender<String>,
    wav: PathBuf,
    levels: [f32; BARS],
    last_sample: f64,
}

const BARS: usize = 9;
const DEFAULT_MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/parakeet-tdt-0.6b-v3-int8");

/// Mic level 0..=1 from the last ~50 ms that pw-record has written so far.
// ponytail: tails the growing WAV instead of opening a second audio stream.
fn mic_level(wav: &Path) -> f32 {
    use std::io::{Seek, SeekFrom};
    let mut buf = [0u8; 1600];
    let read = std::fs::File::open(wav).and_then(|mut f| {
        if f.metadata()?.len() < 44 + buf.len() as u64 {
            return Ok(false);
        }
        f.seek(SeekFrom::End(-(buf.len() as i64)))?;
        f.read_exact(&mut buf).map(|_| true)
    });
    if read.unwrap_or(false) { rms_level(&buf) } else { 0.0 }
}

/// RMS of s16le samples, curved so normal speech fills most of the bar.
fn rms_level(bytes: &[u8]) -> f32 {
    let samples = bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0);
    let (sum, n) = samples.fold((0.0, 0), |(s, n), x| (s + x * x, n + 1));
    if n == 0 {
        return 0.0;
    }
    ((sum / n as f32).sqrt() * 6.0).sqrt().min(1.0)
}

impl App {
    fn toggle(&mut self) {
        self.state = match std::mem::replace(&mut self.state, State::Idle) {
            State::Idle | State::Error(_) => {
                // ponytail: PipeWire's recorder instead of an audio crate (no ALSA headers needed).
                match Command::new("pw-record")
                    .args(["--rate", "16000", "--channels", "1", "--format", "s16"])
                    .arg(&self.wav)
                    .spawn()
                {
                    Ok(child) => {
                        self.levels = [0.0; BARS];
                        State::Recording(child)
                    }
                    Err(e) => State::Error(format!("pw-record: {e}")),
                }
            }
            State::Recording(mut child) => {
                // SIGINT lets pw-record finalize the WAV header.
                unsafe { libc::kill(child.id() as i32, libc::SIGINT) };
                let _ = child.wait();
                // The transcriber thread exits if the model failed to load.
                match self.jobs.send(self.wav.clone()) {
                    Ok(()) => State::Transcribing,
                    Err(_) => State::Error("transcriber not running (model failed to load)".into()),
                }
            }
            busy => busy, // Loading / Transcribing: ignore clicks
        };
    }
}

impl eframe::App for App {
    fn clear_color(&self, _: &egui::Visuals) -> [f32; 4] {
        [0.0; 4]
    }

    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        while self.toggles.try_recv().is_ok() {
            self.toggle();
        }
        if let Ok(res) = self.results.try_recv() {
            self.state = match res {
                Ok(text) => {
                    if !text.is_empty() {
                        let _ = self.paste.send(text.clone());
                    }
                    self.last_text = text;
                    State::Idle
                }
                Err(e) => State::Error(e),
            };
        }

        // Idle shows a mic (None); otherwise bar heights in 0..=1: live mic level
        // while recording, a travelling wave while transcribing, flat dots on error.
        let t = ctx.input(|i| i.time);
        let bars: Option<([f32; BARS], _)> = match &self.state {
            State::Recording(..) => {
                ctx.request_repaint_after(std::time::Duration::from_millis(50));
                // Sample on the 50 ms tick only, so hover repaints don't speed up the scroll.
                if t - self.last_sample >= 0.05 {
                    self.last_sample = t;
                    self.levels.rotate_left(1);
                    self.levels[BARS - 1] = mic_level(&self.wav);
                }
                Some((self.levels, egui::Color32::WHITE))
            }
            State::Transcribing | State::Loading => {
                ctx.request_repaint_after(std::time::Duration::from_millis(33));
                let t = t as f32;
                let wave = std::array::from_fn(|i| 0.3 + 0.3 * (t * 8.0 - i as f32 * 0.8).sin());
                Some((wave, egui::Color32::from_gray(150)))
            }
            State::Error(_) => Some(([0.0; BARS], egui::Color32::from_rgb(230, 140, 0))),
            State::Idle => None,
        };

        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ctx, |ui| {
            let rect = ui.max_rect();
            let resp = ui.interact(rect, ui.id().with("pill"), egui::Sense::click_and_drag());
            if resp.drag_started() {
                ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
            }
            if resp.clicked() {
                self.toggle();
            }
            // ponytail: no tray yet, right-click quits; move Quit to a tray menu once there are settings.
            if resp.secondary_clicked() {
                // Dropping a Child doesn't kill it: stop pw-record or the mic stays open.
                if let State::Recording(child) = &mut self.state {
                    let _ = child.kill();
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            let p = ui.painter();
            let radius = rect.height() / 2.0;
            p.rect_filled(rect, radius, egui::Color32::from_rgb(18, 18, 22));
            p.rect_stroke(rect.shrink(0.5), radius, (1.0, egui::Color32::from_gray(60)), egui::StrokeKind::Inside);
            let (bar_w, gap) = (3.0, 3.0);
            let left = rect.center().x - (BARS as f32 * (bar_w + gap) - gap) / 2.0;
            let max_h = rect.height() - 16.0;
            match bars {
                None => {
                    p.text(rect.center(), egui::Align2::CENTER_CENTER, "🎤", egui::FontId::proportional(18.0), egui::Color32::WHITE);
                }
                Some((bars, bar_color)) => {
                    for (i, h) in bars.iter().enumerate() {
                        let height = bar_w + h * (max_h - bar_w);
                        let x = left + i as f32 * (bar_w + gap);
                        let bar = egui::Rect::from_center_size(
                            egui::pos2(x + bar_w / 2.0, rect.center().y),
                            egui::vec2(bar_w, height),
                        );
                        p.rect_filled(bar, bar_w / 2.0, bar_color);
                    }
                }
            }
            let tip = match &self.state {
                State::Error(e) => e.as_str(),
                _ => self.last_text.as_str(),
            };
            if !tip.is_empty() {
                resp.on_hover_text(tip);
            }
        });
    }
}

const TEXT: &str = "text/plain;charset=utf-8";
// Linux evdev keycodes, as the portal expects.
const KEY_LEFTSHIFT: i32 = 42;
const KEY_INSERT: i32 = 110;

/// Paste worker. Puts each text in clipboard and primary selection, presses
/// Shift+Insert through the RemoteDesktop portal (apps paste the clipboard,
/// terminals the primary), then restores both. KDE asks for permission once;
/// the restore token keeps it. If the portal is refused, text is only copied.
fn paster(texts: Receiver<String>) {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(async {
        if let Err(e) = paste_via_portal(&texts).await {
            eprintln!("portal unavailable, clipboard only: {e}");
        }
        for text in texts.iter() {
            write_selection(false, TEXT, text.as_bytes());
        }
    });
}

async fn paste_via_portal(texts: &Receiver<String>) -> ashpd::Result<()> {
    let token_file = concat!(env!("CARGO_MANIFEST_DIR"), "/.portal-token");
    let token = std::fs::read_to_string(token_file).ok();
    let proxy = RemoteDesktop::new().await?;
    let session = proxy.create_session(Default::default()).await?;
    let opts = SelectDevicesOptions::default()
        .set_devices(ashpd::enumflags2::BitFlags::from(DeviceType::Keyboard))
        .set_persist_mode(PersistMode::ExplicitlyRevoked)
        .set_restore_token(token.as_deref());
    proxy.select_devices(&session, opts).await?;
    let started = proxy.start(&session, None, Default::default()).await?.response()?;
    if let Some(t) = started.restore_token() {
        let _ = std::fs::write(token_file, t);
    }

    // ponytail: blocking recv inside a current-thread runtime; fine, nothing else runs here.
    for text in texts.iter() {
        let saved = [read_selection(false), read_selection(true)];
        write_selection(false, TEXT, text.as_bytes());
        write_selection(true, TEXT, text.as_bytes());
        for (code, state) in [
            (KEY_LEFTSHIFT, KeyState::Pressed),
            (KEY_INSERT, KeyState::Pressed),
            (KEY_INSERT, KeyState::Released),
            (KEY_LEFTSHIFT, KeyState::Released),
        ] {
            proxy.notify_keyboard_keycode(&session, code, state, Default::default()).await?;
        }
        // ponytail: fixed delay; a very slow app could paste the restored content instead.
        tokio::time::sleep(std::time::Duration::from_millis(800)).await;
        for (primary, sel) in [false, true].into_iter().zip(saved) {
            if let Some((mime, data)) = sel {
                write_selection(primary, &mime, &data);
            }
        }
    }
    Ok(())
}

/// Preferred type to save from a `wl-paste --list-types` listing.
// ponytail: restores a single mime type, not every format the owner offered.
fn pick_mime(types: &str) -> Option<&str> {
    types
        .lines()
        .find(|t| t.starts_with("text/plain") || t.starts_with("image/"))
        .or_else(|| types.lines().next())
}

/// `wl-paste`/`wl-copy`, targeting the primary selection if asked.
fn wl(bin: &str, primary: bool) -> Command {
    let mut cmd = Command::new(bin);
    if primary {
        cmd.arg("--primary");
    }
    cmd
}

fn read_selection(primary: bool) -> Option<(String, Vec<u8>)> {
    let types = wl("wl-paste", primary).arg("--list-types").output().ok()?;
    let mime = pick_mime(&String::from_utf8_lossy(&types.stdout))?.to_string();
    let out = wl("wl-paste", primary).args(["--no-newline", "--type", &mime]).output().ok()?;
    out.status.success().then_some((mime, out.stdout))
}

fn write_selection(primary: bool, mime: &str, data: &[u8]) {
    if let Ok(mut c) = wl("wl-copy", primary).args(["--type", mime]).stdin(std::process::Stdio::piped()).spawn() {
        let _ = c.stdin.take().unwrap().write_all(data);
        let _ = c.wait();
    }
}

fn transcriber(model_dir: PathBuf, jobs: Receiver<PathBuf>, out: Sender<Result<String, String>>, ctx: egui::Context) {
    let send = |r| {
        let _ = out.send(r);
        ctx.request_repaint();
    };
    let mut model = match ParakeetModel::load(&model_dir, &Quantization::Int8) {
        Ok(m) => m,
        Err(e) => return send(Err(format!("model {}: {e}", model_dir.display()))),
    };
    send(Ok(String::new())); // model ready: an empty transcript moves Loading to Idle
    for wav in jobs {
        send(transcribe(&mut model, &wav));
    }
}

fn transcribe(model: &mut ParakeetModel, wav: &Path) -> Result<String, String> {
    let samples = transcribe_rs::audio::read_wav_samples(wav).map_err(|e| e.to_string())?;
    let result = model.transcribe_with(&samples, &ParakeetParams::default()).map_err(|e| e.to_string())?;
    Ok(result.text.trim().to_string())
}

fn main() -> eframe::Result {
    let sock = runtime_path("lorito.sock");

    // `lorito toggle`: bind this to a global shortcut in KDE settings.
    if std::env::args().nth(1).as_deref() == Some("toggle") {
        // The connection itself is the toggle signal.
        UnixStream::connect(&sock).expect("lorito is not running");
        return Ok(());
    }

    let model = std::env::var_os("LORITO_MODEL").map_or_else(|| DEFAULT_MODEL.into(), PathBuf::from);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Lorito")
            .with_app_id("lorito")
            .with_inner_size([96.0, 40.0])
            .with_decorations(false)
            .with_transparent(true)
            .with_resizable(false)
            .with_always_on_top(),
        ..Default::default()
    };

    eframe::run_native(
        "Lorito",
        options,
        Box::new(move |cc| {
            let (job_tx, job_rx) = channel();
            let (res_tx, res_rx) = channel();
            let (tog_tx, tog_rx) = channel();

            let c = cc.egui_ctx.clone();
            std::thread::spawn(move || transcriber(model, job_rx, res_tx, c));
            let (paste_tx, paste_rx) = channel();
            std::thread::spawn(move || paster(paste_rx));

            let _ = std::fs::remove_file(&sock);
            let listener = UnixListener::bind(&sock).expect("bind socket");
            let c = cc.egui_ctx.clone();
            std::thread::spawn(move || {
                for _ in listener.incoming().flatten() {
                    let _ = tog_tx.send(());
                    c.request_repaint();
                }
            });

            Ok(Box::new(App {
                state: State::Loading,
                last_text: String::new(),
                jobs: job_tx,
                results: res_rx,
                toggles: tog_rx,
                paste: paste_tx,
                wav: runtime_path("lorito.wav"),
                levels: [0.0; BARS],
                last_sample: 0.0,
            }))
        }),
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn transcribes_fixture() {
        let mut model = super::ParakeetModel::load(super::DEFAULT_MODEL.as_ref(), &super::Quantization::Int8).unwrap();
        for (wav, expected) in [("tests/hello.wav", "hello world"), ("tests/hola.wav", "prueba de dictado")] {
            let text = super::transcribe(&mut model, wav.as_ref()).unwrap().to_lowercase();
            assert!(text.contains(expected), "{text}");
        }
    }

    #[test]
    fn rms_level_scales() {
        assert_eq!(super::rms_level(&[0; 1600]), 0.0);
        let loud: Vec<u8> = (0..800).flat_map(|i| (if i % 2 == 0 { 12000i16 } else { -12000 }).to_le_bytes()).collect();
        assert_eq!(super::rms_level(&loud), 1.0);
        let quiet: Vec<u8> = (0..800).flat_map(|i| (if i % 2 == 0 { 300i16 } else { -300 }).to_le_bytes()).collect();
        let q = super::rms_level(&quiet);
        assert!(q > 0.1 && q < 0.5, "{q}");
    }

    fn app(state: super::State) -> (super::App, std::sync::mpsc::Receiver<std::path::PathBuf>) {
        use std::sync::mpsc::channel;
        let (jobs, job_rx) = channel();
        let (_, results) = channel();
        let (_, toggles) = channel();
        let (paste, _) = channel();
        let app = super::App {
            state,
            last_text: String::new(),
            jobs,
            results,
            toggles,
            paste,
            wav: "/nonexistent.wav".into(),
            levels: [0.0; super::BARS],
            last_sample: 0.0,
        };
        (app, job_rx)
    }

    #[test]
    fn stop_without_transcriber_is_error_not_panic() {
        let child = std::process::Command::new("sleep").arg("10").spawn().unwrap();
        let (mut app, job_rx) = app(super::State::Recording(child));
        drop(job_rx); // transcriber exited after a failed model load
        app.toggle();
        assert!(matches!(app.state, super::State::Error(_)));
    }

    #[test]
    fn picks_restorable_mime() {
        assert_eq!(super::pick_mime("x-kde-foo\ntext/plain;charset=utf-8\nTEXT"), Some("text/plain;charset=utf-8"));
        assert_eq!(super::pick_mime("chromium/x-src\nimage/png"), Some("image/png"));
        assert_eq!(super::pick_mime("x-special/gnome"), Some("x-special/gnome"));
        assert_eq!(super::pick_mime(""), None);
    }
}
