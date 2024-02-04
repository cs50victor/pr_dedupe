mod bert;
mod supabase;
mod upstash;

use std::{env, process::exit};

use clap::Parser;
use futures::stream::StreamExt;
use serde_json::json;
use upstash::Upstash;

#[derive(Clone, Copy)]
enum FileAction {
    Added,
    Modified,
    AddedModified,
    Removed,
    Renamed,
}

impl From<FileAction> for char {
    fn from(val: FileAction) -> Self {
        match val {
            FileAction::Added => '+',
            FileAction::Modified => 'M',
            FileAction::AddedModified => '*',
            FileAction::Removed => '-',
            FileAction::Renamed => '^',
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "finds duplicate or similar prs in a repo", long_about = None)]
struct Args {
    #[arg(long = "added")]
    added_files: String,

    #[arg(long = "modified")]
    modified_files: String,

    #[arg(long = "a-or-m")]
    added_or_modified_files: String,

    #[arg(long = "removed")]
    removed_files: String,

    #[arg(long = "renamed")]
    renamed_files: String,

    #[arg(long = "db", default_value = "upstash")]
    vector_db: String,

    /// Number similar matches to return
    #[arg(short = 'k', default_value_t = 10)]
    top_k: u8,

    /// Minimum similarity, in percentage to match for
    #[arg(short = 'm', default_value_t = 80)]
    min_similarity: u8,
}

#[tokio::main]
async fn main() {
    env_logger::init();
    let args = Args::parse();

    let Args {
        min_similarity: _,
        added_files,
        modified_files,
        added_or_modified_files,
        removed_files,
        renamed_files,
        top_k,
        vector_db,
    } = args;

    if ![
        &added_files,
        &modified_files,
        &added_or_modified_files,
        &removed_files,
        &renamed_files,
    ]
    .iter()
    .any(|arg| !arg.is_empty())
    {
        return;
    };

    let (rest_url, token) = match vector_db.as_str() {
        "supabase" => {
            let (supabase_url, supabase_service_role_key) = (
                env::var("SUPABASE_URL"),
                env::var("SUPABASE_SERVICE_ROLE_KEY"),
            );

            if supabase_url.is_err() || supabase_service_role_key.is_err() {
                eprintln!("both SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY env variables need to be set to use supabase's vector database");
                exit(1);
            }

            (supabase_url.unwrap(), supabase_service_role_key.unwrap())
        }
        "upstash" => {
            let (upstash_vector_rest_url, upstash_vector_rest_token) = (
                env::var("UPSTASH_VECTOR_REST_URL"),
                env::var("UPSTASH_VECTOR_REST_TOKEN"),
            );

            if upstash_vector_rest_url.is_err() || upstash_vector_rest_token.is_err() {
                eprintln!("both UPSTASH_VECTOR_REST_URL and UPSTASH_VECTOR_REST_TOKEN env variables need to be set to use supabase's vector database");
                exit(1);
            }

            (
                upstash_vector_rest_url.unwrap(),
                upstash_vector_rest_token.unwrap(),
            )
        }
        _ => {
            eprintln!(
                "Unsupported vector database name. Supported names are 'supabase', 'upstash' "
            );
            exit(1);
        }
    };

    let pr_files = added_files
        .split(',')
        .map(|file| (file, FileAction::Added))
        .chain(
            modified_files
                .split(',')
                .map(|file| (file, FileAction::Modified)),
        )
        .chain(
            added_or_modified_files
                .split(',')
                .map(|file| (file, FileAction::AddedModified)),
        );

    let mut pr_content = futures::stream::iter(pr_files.map(|(path, file_type)| async move {
        match reqwest::get(path).await {
            Ok(resp) => match resp.bytes().await {
                Ok(resp_bytes) => {
                    let content = std::str::from_utf8(&resp_bytes).unwrap();

                    match file_type {
                        FileAction::Added | FileAction::Modified | FileAction::AddedModified => {
                            parse(file_type, path, Some(content))
                        }
                        _ => {
                            let symbol: char = file_type.into();
                            eprintln!("Unexpected Filetype. Symbol {symbol}");
                            "".to_owned()
                        }
                    }
                }
                Err(e) => {
                    eprintln!("{e}");
                    "".to_owned()
                }
            },
            Err(e) => {
                eprintln!("Couldn't download {path} | Reason {e:?}");
                "".to_owned()
            }
        }
    }))
    .buffer_unordered(10)
    .collect::<Vec<String>>()
    .await;

    pr_content.extend(
        removed_files
            .split(',')
            .map(|file| parse(FileAction::Removed, file, None)),
    );
    pr_content.extend(
        renamed_files
            .split(',')
            .map(|file| parse(FileAction::Renamed, file, None)),
    );

    let embedding = match bert::generate_embeddings(&pr_content.join(" "), 399).await {
        Ok(embedding) => embedding,
        Err(e) => {
            eprintln!("Error: {e}");
            exit(1);
        }
    };

    let db_client = match Upstash::new(rest_url, token) {
        Ok(db_client) => db_client,
        Err(e) => {
            eprintln!("Error: {e}");
            exit(1);
        }
    };

    if let Err(e) = db_client.save_embedding(&embedding).await {
        eprintln!("Error: {e}");
        exit(1);
    };

    let _c = db_client.query(&embedding, top_k).await;
    // check for similar PRs

    // output a json of similar prs
    let similar_prs = json!({"smi":""}).to_string();
    std::fs::write(
        env::var("GITHUB_OUTPUT").unwrap(),
        format!("similar_prs={similar_prs}"),
    )
    .unwrap();
}

fn parse(file_type: FileAction, path: &str, content: Option<&str>) -> String {
    let symbol: char = file_type.into();
    match content {
        Some(c) => format!("{symbol} : {path}\n{c}\n"),
        None => format!("{symbol} : {path}\n"),
    }
}
