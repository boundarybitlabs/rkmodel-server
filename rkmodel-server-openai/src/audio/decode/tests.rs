use super::*;
use tokio_stream::StreamExt;

use crate::audio::{sine, wav};

async fn pcm(bytes: Vec<u8>, name: &str) -> (Vec<i16>, u64) {
    let source = Source::open(bytes, Some(name)).expect("the file should decode");
    let (mut stream, counter) = source.into_stream();

    let mut out = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("decoding should not fail");
        for pair in chunk.as_chunks::<2>().0 {
            out.push(i16::from_le_bytes(*pair));
        }
    }
    let counted = counter.load(Ordering::Relaxed);
    (out, counted)
}

/// The dominant frequency, by walking a few candidates and taking the one with
/// the most energy. Enough to tell a resampled 440 Hz tone from an aliased one
/// without pulling in an FFT.
fn dominant_hz(samples: &[i16], rate: f32, candidates: &[f32]) -> f32 {
    let mut best = (0.0f32, 0.0f32);
    for &hz in candidates {
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (i, s) in samples.iter().enumerate() {
            let angle = std::f32::consts::TAU * hz * i as f32 / rate;
            let v = *s as f32;
            re += v * angle.cos();
            im += v * angle.sin();
        }
        let power = re * re + im * im;
        if power > best.1 {
            best = (hz, power);
        }
    }
    best.0
}

// ---- probing ---------------------------------------------------------------

#[test]
fn a_wav_upload_reports_its_own_rate() {
    let source = Source::open(wav(44_100, 1, &sine(44_100, 440.0, 0.1)), Some("a.wav")).unwrap();
    assert_eq!(source.rate(), 44_100);
}

#[test]
fn something_that_is_not_audio_is_refused() {
    let err = Source::open(b"this is not audio at all".to_vec(), Some("a.wav"))
        .expect_err("plain text should not probe as audio");
    assert!(err.0.contains("could not be read"), "{}", err.0);
}

#[test]
fn a_wrong_extension_still_decodes_from_the_content() {
    // The hint only narrows the guess, so a mislabelled upload still works.
    Source::open(wav(16_000, 1, &sine(16_000, 440.0, 0.05)), Some("a.mp3"))
        .expect("content sniffing should find the wav");
}

// ---- decoding --------------------------------------------------------------

#[tokio::test]
async fn audio_already_at_the_target_rate_passes_through_unchanged() {
    let samples = sine(16_000, 440.0, 0.25);
    let (out, counted) = pcm(wav(16_000, 1, &samples), "a.wav").await;

    assert_eq!(out, samples);
    assert_eq!(counted, samples.len() as u64);
}

#[tokio::test]
async fn stereo_is_averaged_into_one_channel() {
    // Left and right cancel exactly, so an averaged downmix is silence and
    // anything that picks a single channel is not.
    let mut interleaved = Vec::new();
    for i in 0..16_000 {
        let v = ((i % 100) as i16 - 50) * 100;
        interleaved.push(v);
        interleaved.push(-v);
    }
    let (out, _) = pcm(wav(16_000, 2, &interleaved), "a.wav").await;

    assert_eq!(out.len(), 16_000);
    assert!(
        out.iter().all(|s| s.abs() <= 1),
        "the two channels should have cancelled, got {:?}",
        &out[..8]
    );
}

#[tokio::test]
async fn a_44100_upload_is_resampled_to_16000() {
    let (out, counted) = pcm(wav(44_100, 1, &sine(44_100, 440.0, 1.0)), "a.wav").await;

    // One second in, one second out, give or take the filter's edges.
    let expected = 16_000i64;
    assert!(
        (out.len() as i64 - expected).abs() < 400,
        "expected about {expected} samples, got {}",
        out.len()
    );
    assert_eq!(counted, out.len() as u64);
}

#[tokio::test]
async fn resampling_keeps_the_tone_rather_than_aliasing_it() {
    let (out, _) = pcm(wav(44_100, 1, &sine(44_100, 440.0, 1.0)), "a.wav").await;

    // Skip the very start and end, where the filter ramps.
    let middle = &out[2_000..out.len() - 2_000];
    let hz = dominant_hz(
        middle,
        16_000.0,
        &[220.0, 330.0, 440.0, 550.0, 880.0, 1320.0],
    );
    assert_eq!(hz, 440.0, "the resampled tone changed pitch");
}

#[tokio::test]
async fn a_48000_upload_is_resampled_to_16000() {
    let (out, _) = pcm(wav(48_000, 1, &sine(48_000, 440.0, 0.5)), "a.wav").await;

    let expected = 8_000i64;
    assert!(
        (out.len() as i64 - expected).abs() < 400,
        "expected about {expected} samples, got {}",
        out.len()
    );
}

#[tokio::test]
async fn the_counter_measures_the_clip_the_daemon_was_sent() {
    // Two seconds of 44.1 kHz becomes two seconds at 16 kHz, which is the
    // duration `verbose_json` reports.
    let (_, counted) = pcm(wav(44_100, 1, &sine(44_100, 440.0, 2.0)), "a.wav").await;
    let seconds = counted as f32 / TARGET_RATE as f32;
    assert!((seconds - 2.0).abs() < 0.05, "got {seconds}s");
}

// ---- the pure conversions --------------------------------------------------

#[test]
fn samples_encode_as_little_endian_pairs() {
    let mut out = Vec::new();
    encode(&[0.0, 1.0, -1.0], &mut out);
    assert_eq!(out, vec![0, 0, 0xFF, 0x7F, 0x01, 0x80]);
}

#[test]
fn a_sample_past_full_scale_saturates_rather_than_wrapping() {
    let mut out = Vec::new();
    encode(&[2.0, -2.0], &mut out);
    let values: Vec<i16> = out
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| i16::from_le_bytes(*p))
        .collect();
    assert_eq!(values, vec![i16::MAX, -i16::MAX]);
}

#[test]
fn one_channel_is_left_alone() {
    let mut out = Vec::new();
    downmix(&[0.1, 0.2, 0.3], 1, &mut out);
    assert_eq!(out, vec![0.1, 0.2, 0.3]);
}

#[test]
fn channels_are_averaged_frame_by_frame() {
    let mut out = Vec::new();
    downmix(&[1.0, 0.0, 0.5, -0.5], 2, &mut out);
    assert_eq!(out, vec![0.5, 0.0]);
}
