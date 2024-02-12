use std::{env, process::exit};

use anyhow::{bail, Result};
use log::error;

pub fn get_upstash_envs() -> Result<(String, String)> {
    let (upstash_vector_rest_url, upstash_vector_rest_token) = (
        env::var("UPSTASH_VECTOR_REST_URL"),
        env::var("UPSTASH_VECTOR_REST_TOKEN"),
    );

    if upstash_vector_rest_url.is_err() || upstash_vector_rest_token.is_err() {
        bail!("both UPSTASH_VECTOR_REST_URL and UPSTASH_VECTOR_REST_TOKEN env variables need to be set to use supabase's vector database");
    }

    Ok((
        upstash_vector_rest_url.unwrap(),
        upstash_vector_rest_token.unwrap(),
    ))
}

pub fn get_supabase_envs() -> Result<(String, String)> {
    let (supabase_url, supabase_service_role_key) = (
        env::var("SUPABASE_URL"),
        env::var("SUPABASE_SERVICE_ROLE_KEY"),
    );

    if supabase_url.is_err() || supabase_service_role_key.is_err() {
        bail!("both SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY env variables need to be set to use supabase's vector database");
    }

    Ok((supabase_url.unwrap(), supabase_service_role_key.unwrap()))
}

pub fn uuid(repo_name: &str, pr_number: &str) -> String {
    format!("[{repo_name}]:{pr_number}")
}

pub fn uuid_to_pr_number(uuid: &str) -> &str {
    uuid.split(':').next_back().unwrap()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_encode_and_decode() {
        let repo_name = "cs50victor/pr_dedupe";
        let pr_number = "2";

        let uuid = uuid(repo_name, pr_number);

        assert_eq!(uuid_to_pr_number(&uuid), pr_number);
    }
}
