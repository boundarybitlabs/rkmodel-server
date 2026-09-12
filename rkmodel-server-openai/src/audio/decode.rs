//! The upload, decoded to what rkwhisperd takes: 16 kHz mono s16le.
//!
//! Decoding runs on a blocking thread and yields PCM as it goes, so audio
//! reaches the daemon while the rest of the file is still being read rather
//! than after the whole clip is in memory twice.
//!
//! Nothing here touches the network, so every part of it is testable on any
//! host with a file of samples.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use rkmodel_server_protocol::{ByteStream, Error as ProtocolError};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Indexing, Resampler};
use symphonia::core::codecs::audio::{AudioDecoder, AudioDecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// What rkwhisper takes, and what `TranscribeInput::pcm_s16le` carries.
pub const TARGET_RATE: usize = 16_000;

const BYTES_PER_SAMPLE: usize = 2;

/// The resampler's fixed input block. Small enough that a short clip still
/// produces several chunks, large enough that the FFT is not mostly overhead.
const BLOCK: usize = 1024;

/// PCM bytes per message to the daemon, half a second at the target rate. A
/// long upload then moves in steady pieces rather than one burst.
const CHUNK_BYTES: usize = TARGET_RATE * BYTES_PER_SAMPLE / 2;

/// How many decoded chunks may wait ahead of the daemon. Small, because the
/// point of decoding on the fly is not to hold the clip in memory.
const QUEUE: usize = 4;

/// An upload this frontend cannot decode.
///
/// Always the caller's problem rather than the daemon's: a container symphonia
/// does not read, a codec it has no decoder for, or a file with no audio in it.
#[derive(Debug)]
pub struct Unsupported(pub String);

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A probed upload, ready to decode but with no audio read yet.
pub struct Source {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn AudioDecoder>,
    track_id: u32,
    rate: usize,
    channels: usize,
}

impl std::fmt::Debug for Source {
    /// The reader and decoder behind this are trait objects with no `Debug` of
    /// their own, so this reports what the upload turned out to be.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Source")
            .field("rate", &self.rate)
            .field("channels", &self.channels)
            .finish_non_exhaustive()
    }
}

impl Source {
    /// Reads the upload's headers and prepares a decoder for its audio track.
    ///
    /// Separate from decoding so an unreadable upload is refused with a status
    /// rather than becoming an error partway through a stream the daemon has
    /// already started reading.
    pub fn open(bytes: Vec<u8>, filename: Option<&str>) -> Result<Source, Unsupported> {
        // A cursor over the bytes already in memory, so no part of decoding
        // waits on a disk.
        let stream =
            MediaSourceStream::new(Box::new(std::io::Cursor::new(bytes)), Default::default());

        // The extension only narrows the probe's guess. A file named wrongly
        // still decodes, since symphonia falls back to sniffing the content.
        let mut hint = Hint::new();
        if let Some(extension) = filename.and_then(|f| f.rsplit_once('.')).map(|(_, e)| e) {
            hint.with_extension(extension);
        }

        let format = symphonia::default::get_probe()
            .probe(
                &hint,
                stream,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .map_err(|e| Unsupported(format!("The audio file could not be read: {e}")))?;

        let track = format
            .default_track(TrackType::Audio)
            .ok_or_else(|| Unsupported("The file contains no audio track.".into()))?;
        let track_id = track.id;

        let params = track
            .codec_params
            .as_ref()
            .and_then(|p| p.audio())
            .ok_or_else(|| Unsupported("The audio track declares no codec.".into()))?;

        let rate = params
            .sample_rate
            .ok_or_else(|| Unsupported("The audio track declares no sample rate.".into()))?
            as usize;
        let channels = params.channels.as_ref().map_or(1, |c| c.count()).max(1);

        let decoder = symphonia::default::get_codecs()
            .make_audio_decoder(params, &AudioDecoderOptions::default())
            .map_err(|e| Unsupported(format!("The audio codec is not supported: {e}")))?;

        Ok(Source {
            format,
            decoder,
            track_id,
            rate,
            channels,
        })
    }

    /// The upload's own sample rate, before resampling.
    pub fn rate(&self) -> usize {
        self.rate
    }

    /// Decodes on a blocking thread, yielding 16 kHz mono s16le as it goes.
    ///
    /// The counter holds how many samples have been produced. It reaches the
    /// clip's full length once the stream ends, which is where the reported
    /// duration comes from: the upload's own header may be absent, wrong, or
    /// describe a length the file does not actually contain.
    pub fn into_stream(self) -> (ByteStream, Arc<AtomicU64>) {
        let produced = Arc::new(AtomicU64::new(0));
        let counter = produced.clone();
        let (tx, rx) = mpsc::channel(QUEUE);

        // Decoding is CPU-bound and symphonia is synchronous, so it belongs off
        // the runtime's worker threads. The bounded channel is the backpressure:
        // decoding stops when the daemon stops keeping up.
        tokio::task::spawn_blocking(move || {
            if let Err(e) = self.run(&tx, &counter) {
                let _ = tx.blocking_send(Err(e));
            }
        });

        (Box::pin(ReceiverStream::new(rx)), produced)
    }

    fn run(
        mut self,
        tx: &mpsc::Sender<Result<Vec<u8>, ProtocolError>>,
        counter: &AtomicU64,
    ) -> Result<(), ProtocolError> {
        let mut resampler = match self.rate == TARGET_RATE {
            true => None,
            false => Some(Resampling::new(self.rate)?),
        };

        let mut interleaved: Vec<f32> = Vec::new();
        let mut mono: Vec<f32> = Vec::new();
        let mut chunk: Vec<u8> = Vec::with_capacity(CHUNK_BYTES);

        loop {
            let packet = match self.format.next_packet() {
                Ok(Some(packet)) => packet,
                Ok(None) => break,
                // A truncated upload still transcribes what did arrive, which
                // is better than failing a mostly complete recording.
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break
                }
                Err(e) => {
                    return Err(ProtocolError::InvalidInput(format!(
                        "reading the audio failed: {e}"
                    )))
                }
            };
            if packet.track_id != self.track_id {
                continue;
            }

            let decoded = match self.decoder.decode(&packet) {
                Ok(buffer) => buffer,
                // Symphonia documents these as recoverable: skip the packet and
                // keep going rather than losing the whole recording to one.
                Err(SymphoniaError::DecodeError(_)) | Err(SymphoniaError::IoError(_)) => continue,
                Err(e) => {
                    return Err(ProtocolError::InvalidInput(format!(
                        "decoding the audio failed: {e}"
                    )))
                }
            };

            interleaved.resize(decoded.samples_interleaved(), 0.0);
            decoded.copy_to_slice_interleaved(&mut interleaved);
            downmix(&interleaved, self.channels, &mut mono);

            match &mut resampler {
                None => {
                    encode(&mono, &mut chunk);
                    mono.clear();
                }
                Some(r) => r.push(&mut mono, &mut chunk)?,
            }

            while chunk.len() >= CHUNK_BYTES {
                let rest = chunk.split_off(CHUNK_BYTES);
                send(tx, std::mem::replace(&mut chunk, rest), counter)?;
            }
        }

        // Whatever the last block left behind, plus the resampler's own tail.
        if let Some(r) = &mut resampler {
            r.finish(&mono, &mut chunk)?;
        }
        if !chunk.is_empty() {
            send(tx, chunk, counter)?;
        }
        Ok(())
    }
}

fn send(
    tx: &mpsc::Sender<Result<Vec<u8>, ProtocolError>>,
    chunk: Vec<u8>,
    counter: &AtomicU64,
) -> Result<(), ProtocolError> {
    let samples = (chunk.len() / BYTES_PER_SAMPLE) as u64;
    // A closed channel means the request went away, which is not a failure to
    // report anywhere.
    if tx.blocking_send(Ok(chunk)).is_err() {
        return Err(ProtocolError::InvalidInput("the request ended".into()));
    }
    counter.fetch_add(samples, Ordering::Relaxed);
    Ok(())
}

/// The resampler, plus the bookkeeping that keeps its output as long as the
/// input implies and no longer.
///
/// rubato always writes a whole output chunk, even for a short final block, so
/// the tail of a clip would otherwise arrive padded with the filter's own
/// silence. Tracking how much input has been taken says exactly how much output
/// is owed, and the padding past that is dropped.
struct Resampling {
    inner: Fft<f32>,
    /// One chunk of the resampler's output, reused across blocks.
    out: Vec<f32>,
    /// Frames of silence the filter emits before the real audio. Dropping them
    /// keeps segment timestamps honest.
    skip: usize,
    /// Input frames taken from the recording, not counting the flush's silence.
    taken: u64,
    /// Output frames written so far.
    written: u64,
    rate: usize,
}

impl Resampling {
    fn new(rate: usize) -> Result<Resampling, ProtocolError> {
        let inner =
            Fft::<f32>::new(rate, TARGET_RATE, BLOCK, 1, FixedSync::Input).map_err(|e| {
                ProtocolError::InvalidInput(format!(
                    "resampling {rate}Hz to {TARGET_RATE}Hz is not possible: {e}"
                ))
            })?;
        Ok(Resampling {
            out: vec![0f32; inner.output_frames_max()],
            skip: inner.output_delay(),
            inner,
            taken: 0,
            written: 0,
            rate,
        })
    }

    /// How many output frames the input taken so far is worth.
    fn owed(&self) -> u64 {
        (self.taken as f64 * TARGET_RATE as f64 / self.rate as f64).round() as u64
    }

    /// Runs every whole block waiting in `mono`, and keeps the rest.
    fn push(&mut self, mono: &mut Vec<f32>, chunk: &mut Vec<u8>) -> Result<(), ProtocolError> {
        let mut consumed = 0;
        while mono.len() - consumed >= self.inner.input_frames_next() {
            consumed += self.block(&mono[consumed..], None, chunk)?;
        }
        mono.drain(..consumed);
        Ok(())
    }

    /// The last, short block, and then the delay's worth of silence the filter
    /// is still holding.
    fn finish(&mut self, mono: &[f32], chunk: &mut Vec<u8>) -> Result<(), ProtocolError> {
        if !mono.is_empty() {
            self.block(mono, Some(mono.len()), chunk)?;
        }
        // `Some(0)` is a block of pure silence, which pushes out what the filter
        // still holds. Without it the recording loses its last few milliseconds.
        // It takes no input, so it adds nothing to what is owed and can only
        // deliver the remainder.
        let silence = vec![0f32; self.inner.input_frames_next()];
        self.block(&silence, Some(0), chunk)?;
        Ok(())
    }

    /// One block through the resampler, appended to `chunk` as s16le.
    ///
    /// Returns how many input frames it consumed.
    fn block(
        &mut self,
        input: &[f32],
        partial_len: Option<usize>,
        chunk: &mut Vec<u8>,
    ) -> Result<usize, ProtocolError> {
        let frames = input.len().min(self.inner.input_frames_next());
        let adapter = InterleavedSlice::new(&input[..frames], 1, frames)
            .map_err(|e| ProtocolError::InvalidInput(format!("resampling failed: {e}")))?;

        let capacity = self.out.len();
        let mut out = InterleavedSlice::new_mut(&mut self.out, 1, capacity)
            .map_err(|e| ProtocolError::InvalidInput(format!("resampling failed: {e}")))?;

        let indexing = Indexing {
            partial_len,
            ..Indexing::default()
        };
        let (consumed, produced) = self
            .inner
            .process_into_buffer(&adapter, &mut out, Some(&indexing))
            .map_err(|e| ProtocolError::InvalidInput(format!("resampling failed: {e}")))?;

        // rubato reports a whole block consumed even for a partial one, since
        // it reads the rest as silence. Counting that would make the clip look
        // longer than it is and let the padding through, so a partial block
        // counts only the frames it was actually given. `Some(0)` is the flush,
        // which is silence rather than any of the recording.
        self.taken += partial_len.unwrap_or(consumed) as u64;

        let dropped = self.skip.min(produced);
        self.skip -= dropped;

        let available = (produced - dropped) as u64;
        let room = self.owed().saturating_sub(self.written);
        let writing = available.min(room) as usize;

        encode(&self.out[dropped..dropped + writing], chunk);
        self.written += writing as u64;
        Ok(consumed)
    }
}

/// Averages the channels into one.
///
/// Whisper takes mono, and averaging keeps both sides of a stereo recording
/// rather than throwing one away.
fn downmix(interleaved: &[f32], channels: usize, out: &mut Vec<f32>) {
    if channels <= 1 {
        out.extend_from_slice(interleaved);
        return;
    }
    out.reserve(interleaved.len() / channels);
    for frame in interleaved.chunks_exact(channels) {
        out.push(frame.iter().sum::<f32>() / channels as f32);
    }
}

/// f32 samples to signed 16-bit little-endian.
///
/// Clamping rather than wrapping, so a sample past full scale saturates the way
/// every other decoder does instead of flipping sign.
fn encode(samples: &[f32], out: &mut Vec<u8>) {
    out.reserve(samples.len() * BYTES_PER_SAMPLE);
    for sample in samples {
        let scaled = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        out.extend_from_slice(&scaled.to_le_bytes());
    }
}

#[cfg(test)]
mod tests;
