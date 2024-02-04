use postgrest::Postgrest;
use serde_json::json;

// add later

struct SB {
    client: Postgrest
}

impl SB {
    pub fn new<T>(supabase_url: T, supabase_service_role_key: impl AsRef<str>) -> Self
    where
        T: Into<String>,
    {
        Self { 
            client: Postgrest::new(supabase_url).insert_header("apikey", supabase_service_role_key)
            // TODO: add later 
            // {
            //     auth: {
            //       persistSession: false,
            //       autoRefreshToken: false,
            //     },
            //   }
        }
    }

    pub async fn save_embedding(&self, embedding: Vec<f32>){
        let body = json!({
            "pr_num": "pr_num",
            "name": "repo_name",
            "embedding": embedding
        });
        
        let resp = self.client.from("repos").upsert(body.to_string()).execute().await;
        
    }
}
