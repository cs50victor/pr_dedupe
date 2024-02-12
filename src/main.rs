mod bert;
mod files_to_ignore;
mod supabase;
mod upstash;
mod utils;

use std::env;

use clap::Parser;
use futures::stream::StreamExt;
use log::info;

use serde::{Deserialize, Serialize};
use upstash::Upstash;

use crate::{
    files_to_ignore::FILES_TO_IGNORE,
    utils::{get_upstash_envs, log_err_and_exit, set_hf_home_env},
};

#[derive(Serialize, Deserialize, Debug)]
pub struct SimilarPRsInner {
    pub pr_url: String,
    pub percentage: f32,
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
    #[arg(long)]
    closed: String,

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
    set_hf_home_env();

    pretty_env_logger::formatted_builder()
        .filter_module("pr_dedupe", log::LevelFilter::Info)
        .init();

    let args = Args::parse();

    let Args {
        closed,
        min_similarity: _,
        added_files,
        modified_files,
        removed_files,
        renamed_files,
        top_k,
        vector_db,
    } = args;

    let (rest_url, token) = match vector_db.as_str() {
        "upstash" => match get_upstash_envs() {
            Ok(envs) => envs,
            Err(e) => {
                log_err_and_exit(format!("{e}"));
            }
        },
        "supabase" => match get_upstash_envs() {
            Ok(envs) => envs,
            Err(e) => {
                log_err_and_exit(format!("{e}"));
            }
        },
        _ => {
            log_err_and_exit(
                "Unsupported vector database name. Supported names are 'supabase', 'upstash' ",
            );
        }
    };

    let db_client = match Upstash::new(rest_url, token) {
        Ok(db_client) => db_client,
        Err(e) => {
            log_err_and_exit(format!("{e}"));
        }
    };

    info!("Created vector db client");

    if closed.trim().parse::<bool>().unwrap() {
        if let Err(e) = db_client.remove_pr_from_db().await {
            log_err_and_exit(format!("{e}"));
        }
        info!("Deleted PR from vector db");
        return;
    }

    let pr_content = match [
        &added_files,
        &modified_files,
        &removed_files,
        &renamed_files,
    ]
    .iter()
    .all(|arg| arg.is_empty())
    {
        true => {
            info!("this pr has no content, it's probably a bot or spam");
            [" ".to_string()].to_vec()
        }
        false => {
            let raw_url_prefix = format!(
                "https://github.com/{}/raw/{}/",
                env::var("REPO_NAME").unwrap(),
                env::var("GITHUB_SHA").unwrap()
            );

            info!("raw_url_prefix {}", &raw_url_prefix);

            let pr_files = added_files
                .split(',')
                .map(|file| (file, FileAction::Added))
                .chain(
                    modified_files
                        .split(',')
                        .map(|file| (file, FileAction::Modified)),
                )
                .filter(|(file, _)| {
                    !file.is_empty()
                        && !FILES_TO_IGNORE.iter().any(|&suffix| file.ends_with(suffix))
                })
                .map(|(file, action)| (format!("{}{file}", &raw_url_prefix), action));

            info!(
                "downloading PR files | {:?}",
                pr_files.clone().collect::<Vec<_>>()
            );

            let mut pr_content =
                futures::stream::iter(pr_files.map(|(path, file_type)| async move {
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
                                        log_err_and_exit(format!(
                                            "Unexpected Filetype. Symbol {}",
                                            symbol
                                        ));
                                    }
                                }
                            }
                            Err(e) => {
                                log_err_and_exit(format!("{e}"));
                            }
                        },
                        Err(e) => {
                            log_err_and_exit(format!("Couldn't download {path} | Reason {e:?}"));
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
            pr_content
        }
    };

    let embedding = match bert::generate_embeddings(pr_content, 384).await {
        Ok(embedding) => embedding,
        Err(e) => {
            log_err_and_exit(format!("{e}"));
        }
    };

    if let Err(e) = db_client.save_embedding(&embedding).await {
        log_err_and_exit(format!("{e}"));
    }

    info!("Saved embedding");

    let similar_prs = match db_client.query(&embedding, top_k).await {
        Ok(resp) => serde_json::to_string(&resp).unwrap(),
        Err(e) => {
            log_err_and_exit(format!("{e}"));
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
