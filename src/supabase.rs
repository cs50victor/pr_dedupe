// add supabase later
use std::env;

use anyhow::bail;
use postgrest::Postgrest;
use serde_json::json;

use anyhow::Result;

struct SB {
    client: Postgrest,
}

impl SB {
    pub fn new() -> Result<Self> {
        let (supabase_url, supabase_service_role_key) = (
            env::var("SUPABASE_URL"),
            env::var("SUPABASE_SERVICE_ROLE_KEY"),
        );

        if supabase_url.is_err() || supabase_service_role_key.is_err() {
            bail!("both SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY env variables need to be set to use supabase's vector database");
        }

        let (supabase_url, supabase_service_role_key) =
            (supabase_url.unwrap(), supabase_service_role_key.unwrap());

        Ok(Self {
            // TODO: add later
            client: Postgrest::new(supabase_url).insert_header("apikey", supabase_service_role_key),
        })
    }

    pub async fn save_embedding(&self, embedding: Vec<f32>) {
        let body = json!({
            "pr_num": "pr_num",
            "name": "repo_name",
            "embedding": embedding
        });

        let _resp = self
            .client
            .from("repos")
            .upsert(body.to_string())
            .execute()
            .await;
    }
}
