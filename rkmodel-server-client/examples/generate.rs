//! Drives one generation against a running daemon, for checking a model by
//! hand.
//!
//!     cargo run -p rkmodel-server-client --example generate -- \
//!         --model minicpm4-0.5b "why is the sky blue?"

use std::time::Instant;

use rkmodel_server_client::RkModelClient;
use rkmodel_server_protocol::{
    Event, GenerateInput, Input, Message, Operation, Output, RkModelServer, Role,
};
use tokio_stream::StreamExt;

#[derive(Debug)]
struct Args {
    daemon: String,
    model: String,
    prompt: String,
    system: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    reasoning: Option<bool>,
    stream: bool,
    list: bool,
}

fn usage() -> ! {
    eprintln!(
        "usage: generate [--daemon URL] [--model ID] [--system TEXT] [--max-tokens N] \\\n\
         \x20      [--temperature F] [--reasoning on|off] [--no-stream] [--list] PROMPT"
    );
    std::process::exit(2)
}

fn parse() -> Args {
    let mut args = Args {
        daemon: "http://127.0.0.1:7070".into(),
        model: String::new(),
        prompt: String::new(),
        system: None,
        max_tokens: None,
        temperature: None,
        reasoning: None,
        stream: true,
        list: false,
    };
    let mut rest = std::env::args().skip(1);
    while let Some(arg) = rest.next() {
        let mut value = || rest.next().unwrap_or_else(|| usage());
        match arg.as_str() {
            "--daemon" => args.daemon = value(),
            "--model" => args.model = value(),
            "--system" => args.system = Some(value()),
            "--max-tokens" => args.max_tokens = value().parse().ok(),
            "--temperature" => args.temperature = value().parse().ok(),
            "--reasoning" => args.reasoning = Some(value() == "on"),
            "--no-stream" => args.stream = false,
            "--list" => args.list = true,
            "-h" | "--help" => usage(),
            other if other.starts_with("--") => usage(),
            other => args.prompt = other.to_string(),
        }
    }
    args
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse();
    let client = RkModelClient::new(&args.daemon, None)?;

    if args.list {
        for m in client.models().await? {
            let ops: Vec<_> = m.operations.iter().map(|o| o.as_str()).collect();
            println!(
                "{:<24} {:<12} {} reasoning={}",
                m.id,
                m.state.as_str(),
                ops.join(","),
                m.reasoning
            );
        }
        return Ok(());
    }

    if args.model.is_empty() || args.prompt.is_empty() {
        usage();
    }

    let mut messages = Vec::new();
    if let Some(system) = &args.system {
        messages.push(Message::text(Role::System, system));
    }
    messages.push(Message::text(Role::User, &args.prompt));

    let input = Input::Generate(GenerateInput {
        messages,
        temperature: args.temperature,
        top_p: None,
        max_tokens: args.max_tokens,
        reasoning: args.reasoning,
    });

    let started = Instant::now();
    if !args.stream {
        let outputs = client
            .invoke(Operation::Generate, &args.model, vec![input])
            .await?;
        for output in outputs {
            if let Output::Generated {
                text,
                reasoning,
                finish,
                usage,
            } = output
            {
                if let Some(r) = reasoning {
                    println!("[reasoning] {r}");
                }
                println!("{text}");
                println!("\n[{finish:?}] {usage:?} in {:?}", started.elapsed());
            }
        }
        return Ok(());
    }

    let mut stream = client
        .invoke_stream(Operation::Generate, &args.model, input)
        .await?;
    let mut first_token = None;
    let mut in_reasoning = false;

    while let Some(event) = stream.next().await {
        match event? {
            Event::ReasoningDelta(s) => {
                if !in_reasoning {
                    print!("[reasoning] ");
                    in_reasoning = true;
                }
                first_token.get_or_insert_with(|| started.elapsed());
                print!("{s}");
                flush();
            }
            Event::TextDelta(s) => {
                if in_reasoning {
                    println!();
                    in_reasoning = false;
                }
                first_token.get_or_insert_with(|| started.elapsed());
                print!("{s}");
                flush();
            }
            Event::Segment(_) => {}
            Event::Done { finish, usage } => {
                let elapsed = started.elapsed();
                let tps = usage.output_tokens as f64 / elapsed.as_secs_f64();
                println!("\n\n[{finish:?}] {usage:?}");
                println!(
                    "first token in {:?}, {} tokens in {:?}, {tps:.1} tok/s",
                    first_token.unwrap_or(elapsed),
                    usage.output_tokens,
                    elapsed
                );
            }
        }
    }
    Ok(())
}

fn flush() {
    use std::io::Write;
    let _ = std::io::stdout().flush();
}
