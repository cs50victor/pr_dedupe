use anyhow::Result;
use reqwest::{header, Client, Response, Url};
use serde_json::json;


pub struct Upstash {
    client: Client,
    url_endpoint: Url
}

impl Upstash {
    pub fn new(upstash_vector_rest_url:String, upstash_vector_rest_token:String) -> Result<Self>{
        let mut value = header::HeaderValue::from_str(&format!("Bearer {upstash_vector_rest_token}"))?;

        value.set_sensitive(true);
        let mut headers = header::HeaderMap::new();
        headers.insert(header::AUTHORIZATION, value);

        // initialize reqwest client
        let client = reqwest::Client::builder().default_headers(headers).build()?;

        let url_endpoint = Url::parse(&upstash_vector_rest_url)?;

        Ok(
            Self {
                client,
                url_endpoint
            }
        )
    }

    pub async fn save_embedding(&self,embedding: &Vec<f32>, repo_name:String, pr_number:u16) -> Result<Response> {
        
        let data = json!({
            "id": uuid(repo_name,pr_number), 
            "vector": embedding,
        });

        let resp  = self.client.post(format!("{}/upsert",self.url_endpoint)).body(
            data.to_string()
        ).send().await?;

        Ok(resp)
    }

    pub async fn query(&self, embedding: &Vec<f32>) -> Result<Response>{
        let data = json!({
            "topK": 10,
            "vector": &embedding,
            "includeVectors": false, 
            "includeMetadata": false
        });

        let resp  = self.client.post(format!("{}/query",self.url_endpoint)).body(
            data.to_string()
        ).send().await?;

        Ok(resp)
    }

}

fn uuid(repo_name:String, pr_number: u16) -> String {
    format!("[{repo_name}]:{pr_number}")
}