mod bert;

use futures::stream::StreamExt;

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

    if !&min_similarity.is_empty() {
        log_error(min_similarity.to_string());
        std::process::exit(1);
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


    let embeddings = bert::generate_embeddings(&pr_content.join(" "), 399).await;
    
    // save embeddings
    // repo_name, pr_id, embeddings

    // check for similar PRs

    // output a json of similar prs
    
}

fn log_error(err_msg:String){
    let github_output = std::env::var("GITHUB_OUTPUT").unwrap();
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
