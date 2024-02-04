mod bert;
mod supabase;
mod upstash;

use std::{env, process::exit};

use futures::stream::StreamExt;
use serde_json::json;
use upstash::Upstash;

#[derive(Clone, Copy)]
enum FileAction {
    Added,
    Modified,
    AddedModified,
    Removed,
    Renamed
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

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (min_similarity, added_files, modified_files, added_or_modified_files, removed_files, renamed_files) = (&args[1].as_str(), args[2].as_str(), args[3].as_str(), args[4].as_str(), args[5].as_str(),args[6].as_str());
    let (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY) = (env::var("SUPABASE_URL"),env::var("SUPABASE_SERVICE_ROLE_KEY"));

    if !(SUPABASE_URL.is_ok() && SUPABASE_SERVICE_ROLE_KEY.is_ok()){    
        log_error("both SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY env variables need to be set to use this Github Action".to_string());
        exit(1);
    }
    
    let (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY) = (SUPABASE_URL.unwrap(), SUPABASE_SERVICE_ROLE_KEY.unwrap());

    if !&min_similarity.is_empty() {
        log_error(min_similarity);
        exit(1);
    }

    let pr_files = added_files.split(',').map(|file| (file, FileAction::Added))
                        .chain(modified_files.split(',').map(|file| (file, FileAction::Modified)))
                        .chain(added_or_modified_files.split(',').map(|file| (file, FileAction::AddedModified)));


    let mut pr_content = futures::stream::iter(
        pr_files.map(|(path, file_type)| {
            async move {
                match reqwest::get(path).await {
                    Ok(resp) => {
                        match resp.bytes().await {
                            Ok(resp_bytes) => {
                                let content = std::str::from_utf8(&resp_bytes).unwrap();

                                match file_type {
                                    FileAction::Added | FileAction::Modified | FileAction::AddedModified => {
                                        parse(file_type, path, Some(content))
                                    },
                                    _ => {
                                        let symbol : char = file_type.into();
                                        log_error(format!("Unexpected Filetype. Symbol {symbol}"));
                                        "".to_owned()
                                    },
                                }
                            }
                            Err(e) => {
                                log_error(e.to_string());
                                "".to_owned()
                            },
                        }
                    }
                    Err(e) => {
                        log_error(format!("Couldn't download {path} | Reason {e:?}"));
                        "".to_owned()
                    },
                }
            }
    })).buffer_unordered(10).collect::<Vec<String>>().await;

    pr_content.extend(removed_files.split(',').map(|file| parse( FileAction::Removed, file, None)));
    pr_content.extend(renamed_files.split(',').map(|file| parse( FileAction::Renamed, file, None)));


    let embedding = match bert::generate_embeddings(&pr_content.join(" "), 399).await{
        Ok(embedding) => embedding,
        Err(e) =>{
            log_error(e.to_string());
            exit(1);
        }
    };
    
    let db_client = match Upstash::new(){
        Ok(db_client) => db_client,
        Err(e) => {
            log_error(e.to_string());
            exit(1);
        }
    };

    if let Err(e)= db_client.save_embedding(&embedding, repo_name, pr_number).await{
        log_error(e.to_string());
        exit(1);
    };
    
    let c = db_client.query(&embedding).await;
    // check for similar PRs

    // output a json of similar prs
    
}

fn log_error(err_msg: impl AsRef<str>){
    let github_output = env::var("GITHUB_OUTPUT").unwrap();
    let err_msg = err_msg.as_ref();
    eprintln!("Error: {err_msg}");
    std::fs::write(github_output, format!("error={err_msg}")).unwrap();
}

fn parse(file_type:FileAction, path:&str, content: Option<&str>) -> String {
    let symbol : char = file_type.into();
    match content {
        Some(c) => format!("{symbol} : {path}\n{c}\n"),
        None => format!("{symbol} : {path}\n"),
    }
}
