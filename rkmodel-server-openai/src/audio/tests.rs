use super::*;

fn segments() -> Vec<Segment> {
    vec![
        Segment {
            text: " the sky".into(),
            start_s: 0.0,
            end_s: 1.25,
        },
        Segment {
            text: " is blue".into(),
            start_s: 1.25,
            end_s: 3.5,
        },
    ]
}

#[test]
fn json_carries_only_the_text() {
    let body = body(
        ResponseFormat::Json,
        "the sky is blue",
        &segments(),
        3.5,
        None,
    );
    let value: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(value, json!({"text": "the sky is blue"}));
}

#[test]
fn text_is_the_transcript_and_nothing_else() {
    let body = body(
        ResponseFormat::Text,
        "the sky is blue",
        &segments(),
        3.5,
        None,
    );
    assert_eq!(body, "the sky is blue\n");
}

#[test]
fn verbose_json_reports_the_duration_and_every_segment() {
    let body = body(
        ResponseFormat::VerboseJson,
        "the sky is blue",
        &segments(),
        3.5,
        Some("fr"),
    );
    let value: Value = serde_json::from_str(&body).unwrap();

    assert_eq!(value["task"], "transcribe");
    assert_eq!(value["language"], "fr");
    assert_eq!(value["duration"], 3.5);
    assert_eq!(value["text"], "the sky is blue");

    let segments = value["segments"].as_array().unwrap();
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0]["id"], 0);
    assert_eq!(segments[0]["start"], 0.0);
    assert_eq!(segments[1]["end"], 3.5);
    assert_eq!(segments[1]["text"], " is blue");
}

#[test]
fn verbose_json_carries_every_field_the_sdk_requires() {
    let body = body(ResponseFormat::VerboseJson, "hello", &segments(), 3.5, None);
    let value: Value = serde_json::from_str(&body).unwrap();
    let segment = &value["segments"][0];

    // The SDK's TranscriptionSegment declares all of these as required, so a
    // missing one raises rather than parsing to None.
    for field in [
        "id",
        "seek",
        "start",
        "end",
        "text",
        "tokens",
        "temperature",
        "avg_logprob",
        "compression_ratio",
        "no_speech_prob",
    ] {
        assert!(!segment[field].is_null(), "{field} is missing");
    }
}

#[test]
fn an_unset_language_is_reported_as_rkwhispers_default() {
    let body = body(ResponseFormat::VerboseJson, "hello", &[], 1.0, None);
    let value: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(value["language"], "en");
}

#[test]
fn srt_numbers_its_cues_from_one_and_uses_a_comma() {
    let body = body(ResponseFormat::Srt, "ignored", &segments(), 3.5, None);
    assert_eq!(
        body,
        "1\n00:00:00,000 --> 00:00:01,250\nthe sky\n\n\
         2\n00:00:01,250 --> 00:00:03,500\nis blue\n\n"
    );
}

#[test]
fn vtt_opens_with_its_header_and_uses_a_full_stop() {
    let body = body(ResponseFormat::Vtt, "ignored", &segments(), 3.5, None);
    assert_eq!(
        body,
        "WEBVTT\n\n\
         00:00:00.000 --> 00:00:01.250\nthe sky\n\n\
         00:00:01.250 --> 00:00:03.500\nis blue\n\n"
    );
}

#[test]
fn a_timestamp_past_an_hour_still_reads_correctly() {
    assert_eq!(timestamp(3661.5, ','), "01:01:01,500");
    assert_eq!(timestamp(3661.5, '.'), "01:01:01.500");
}

#[test]
fn a_negative_timestamp_clamps_to_zero() {
    assert_eq!(timestamp(-1.0, ','), "00:00:00,000");
}

#[test]
fn subtitles_from_no_segments_are_still_well_formed() {
    assert_eq!(body(ResponseFormat::Srt, "", &[], 0.0, None), "");
    assert_eq!(body(ResponseFormat::Vtt, "", &[], 0.0, None), "WEBVTT\n\n");
}

#[test]
fn every_format_parses_from_its_name() {
    assert_eq!(ResponseFormat::parse("json").unwrap(), ResponseFormat::Json);
    assert_eq!(ResponseFormat::parse("text").unwrap(), ResponseFormat::Text);
    assert_eq!(
        ResponseFormat::parse("verbose_json").unwrap(),
        ResponseFormat::VerboseJson
    );
    assert_eq!(ResponseFormat::parse("srt").unwrap(), ResponseFormat::Srt);
    assert_eq!(ResponseFormat::parse("vtt").unwrap(), ResponseFormat::Vtt);
}

#[test]
fn an_unknown_format_is_refused_naming_the_field() {
    let err = ResponseFormat::parse("yaml").expect_err("yaml is not a transcript format");
    let body = err.as_error_body();
    assert_eq!(body["error"]["param"], "response_format");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("verbose_json"),
        "the message should list what is accepted"
    );
}

#[test]
fn json_formats_say_so_in_their_content_type() {
    assert_eq!(ResponseFormat::Json.content_type(), "application/json");
    assert_eq!(
        ResponseFormat::VerboseJson.content_type(),
        "application/json"
    );
    assert!(ResponseFormat::Srt.content_type().starts_with("text/plain"));
    assert!(ResponseFormat::Vtt.content_type().starts_with("text/plain"));
    assert!(ResponseFormat::Text
        .content_type()
        .starts_with("text/plain"));
}
