use std::env;

use anyhow::{bail, Result};

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
