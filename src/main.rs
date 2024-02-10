mod bert;
mod files_to_ignore;
mod supabase;
mod upstash;

use std::{env, process::exit};

use clap::Parser;
use futures::stream::StreamExt;
use log::{error, info};

use serde::{Deserialize, Serialize};
use upstash::Upstash;

use crate::files_to_ignore::FILES_TO_IGNORE;

#[derive(Serialize, Deserialize, Debug)]
pub struct SimilarPRsInner {
    pub pr_url:String,
    pub percentage: f32
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SimilarPRs {
    pub data: Vec<SimilarPRsInner>,
}

#[derive(Clone, Copy, Debug)]
enum FileAction {
    Added,
    Modified,
    Removed,
    Renamed,
}

impl From<FileAction> for char {
    fn from(val: FileAction) -> Self {
        match val {
            FileAction::Added => '+',
            FileAction::Modified => 'M',
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
    pretty_env_logger::formatted_builder()
        .filter_module("pr_dedupe", log::LevelFilter::Info)
        .init();

    let args = Args::parse();

    let Args {
        min_similarity: _,
        added_files,
        modified_files,
        removed_files,
        renamed_files,
        top_k,
        vector_db,
    } = args;

    if ![
        &added_files,
        &modified_files,
        &removed_files,
        &renamed_files,
    ]
    .iter()
    .any(|arg| !arg.is_empty())
    {
        return;
    };

    let raw_url_prefix = format!(
        "https://github.com/{}/raw/{}/",
        env::var("REPO_NAME").unwrap(),
        env::var("GITHUB_SHA").unwrap()
    );

    info!("raw_url_prefix {}", &raw_url_prefix);

    let (rest_url, token) = match vector_db.as_str() {
        "supabase" => {
            let (supabase_url, supabase_service_role_key) = (
                env::var("SUPABASE_URL"),
                env::var("SUPABASE_SERVICE_ROLE_KEY"),
            );

            if supabase_url.is_err() || supabase_service_role_key.is_err() {
                error!("both SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY env variables need to be set to use supabase's vector database");
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
                error!("both UPSTASH_VECTOR_REST_URL and UPSTASH_VECTOR_REST_TOKEN env variables need to be set to use supabase's vector database");
                exit(1);
            }

            (
                upstash_vector_rest_url.unwrap(),
                upstash_vector_rest_token.unwrap(),
            )
        }
        _ => {
            error!("Unsupported vector database name. Supported names are 'supabase', 'upstash' ");
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
        .filter(|(file, _)| {
            !file.is_empty() && !FILES_TO_IGNORE.iter().any(|&suffix| file.ends_with(suffix))
        })
        .map(|(file, action)| (format!("{}{file}", &raw_url_prefix), action));

    info!(
        "downloading PR files | {:?}",
        pr_files.clone().collect::<Vec<_>>()
    );

    let mut pr_content = futures::stream::iter(pr_files.map(|(path, file_type)| async move {
        match reqwest::get(&path).await {
            Ok(resp) => match resp.bytes().await {
                Ok(resp_bytes) => {
                    let content = std::str::from_utf8(&resp_bytes).unwrap();

                    match file_type {
                        FileAction::Added | FileAction::Modified => {
                            parse(file_type, &path, Some(content))
                        }
                        _ => {
                            let symbol: char = file_type.into();
                            error!("Unexpected Filetype. Symbol {symbol}");
                            "".to_owned()
                        }
                    }
                }
                Err(e) => {
                    error!("{e}");
                    exit(1);
                }
            },
            Err(e) => {
                error!("Couldn't download {path} | Reason {e:?}");
                exit(1);
            }
        }
    }))
    .buffer_unordered(10)
    .collect::<Vec<String>>()
    .await;

    pr_content.extend(
        removed_files
            .split(',')
            .map(|file| (file, FileAction::Removed))
            .chain(
                renamed_files
                    .split(',')
                    .map(|file| (file, FileAction::Renamed)),
            )
            .filter(|(file, _)| !file.is_empty())
            .map(|(file, file_action)| {
                parse(file_action, &format!("{}{file}", &raw_url_prefix), None)
            }),
    );

    let embedding = match bert::generate_embeddings(pr_content, 399).await {
        Ok(embedding) => embedding,
        Err(e) => {
            error!("{e}");
            exit(1);
        }
    };

    let db_client = match Upstash::new(rest_url, token) {
        Ok(db_client) => db_client,
        Err(e) => {
            error!("{e}");
            exit(1);
        }
    };

    info!("Created db client");

    if let Err(e) = db_client.save_embedding(&embedding).await {
        error!("{e}");
        exit(1);
    }

    info!("Saved embedding");

    let similar_prs = match db_client.query(&embedding, top_k).await {
        Ok(resp) => serde_json::to_string(&resp).unwrap(),
        Err(e) => {
            error!("{e}");
            exit(1);
        }
    };

    info!("Queried for similar PRs");

    info!("Similar PRs {similar_prs:?}");
    // check for similar PRs
    std::fs::write(
        env::var("GITHUB_OUTPUT").unwrap(),
        format!("similar_prs={similar_prs}"),
    )
    .unwrap();
}

fn parse(file_type: FileAction, path: &str, content: Option<&str>) -> String {
    let symbol: char = file_type.into();
    match content {
        Some(c) => {
            info!("parsed {path}'s content");
            format!("{symbol} : {path}\n{c}\n")
        }
        None => format!("{symbol} : {path}\n"),
    }
}
