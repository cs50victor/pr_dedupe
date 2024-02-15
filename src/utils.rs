use std::{
    env,
    fs::File,
    io::{self, Write},
    path::Path,
    process::exit,
};

use log::error;

use anyhow::Result;

use crate::SimilarPRs;

pub trait VectorDB {
    async fn save_embedding(&self, embedding: &[f32]) -> Result<()>;
    async fn remove_pr(&self) -> Result<()>;
    async fn query(&self, embedding: &[f32], top_k: u8, min_similarity: u8) -> Result<SimilarPRs>;
}

pub fn uuid(repo_name: &str, pr_number: &str) -> String {
    format!("{repo_name}:{pr_number}")
}

pub fn uuid_to_pr_number(uuid: &str) -> &str {
    uuid.split(':').next_back().unwrap()
}

pub fn uuid_to_repo_name(uuid: &str) -> &str {
    uuid.split(':').next().unwrap()
}

pub fn log_err_and_exit(msg: impl AsRef<str>) -> ! {
    error!("{}", msg.as_ref());
    exit(1);
}

/// sets HF HOME env if it doesn't exist
pub fn set_hf_home_env() {
    let key = "HF_HOME";
    if env::var(key).is_err() {
        env::set_var(key, ".");
    };
}

pub fn write_append<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> io::Result<()> {
    fn inner(path: &Path, contents: &[u8]) -> io::Result<()> {
        File::options().append(true).open(path)?.write_all(contents)
    }
    inner(path.as_ref(), contents.as_ref())
}
pub fn set_output(key: &str, value: &str) {
    std::fs::write(env::var("GITHUB_OUTPUT").unwrap(), format!("{key}={value}")).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_encode_and_decode_pr_number() {
        let repo_name = "cs50victor/pr_dedupe";
        let pr_number = "2";

        let uuid = uuid(repo_name, pr_number);

        assert_eq!(uuid_to_pr_number(&uuid), pr_number);
    }

    #[test]
    fn uuid_encode_and_decode_repo_name() {
        let repo_name = "cs50victor/pr_dedupe";
        let pr_number = "2";

        let uuid = uuid(repo_name, pr_number);

        assert_eq!(uuid_to_repo_name(&uuid), repo_name);
    }
}
