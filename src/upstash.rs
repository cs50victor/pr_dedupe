use std::env;

use anyhow::{bail, Result};

use reqwest::{header, Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{SimilarPRs, SimilarPRsInner};

pub struct Upstash {
    client: Client,
    url_endpoint: Url,
}

#[derive(Serialize, Deserialize, Debug)]
struct Data {
    id: String,
    score: f32,
}

#[derive(Serialize, Deserialize, Debug)]
struct QueryResult {
    result: Vec<Data>,
}

impl From<QueryResult> for SimilarPRs {
    fn from(val: QueryResult) -> Self {
        let add_pr_prefix = |pr_number: &str| {
            format!(
                "https://github.com/{}/pull/{pr_number}",
                env::var("REPO_NAME").unwrap(),
            )
        };

        SimilarPRs {
            data: val
                .result
                .iter()
                .map(|d| {
                    let pr_number = d.id.split(':').next_back().unwrap();
                    SimilarPRsInner{
                        pr_url: add_pr_prefix(pr_number),
                        percentage: d.score * 100.0
                    }
                })
                .collect::<Vec<_>>(),
        }
    }
}

impl Upstash {
    pub fn new(upstash_vector_rest_url: String, upstash_vector_rest_token: String) -> Result<Self> {
        let mut value =
            header::HeaderValue::from_str(&format!("Bearer {upstash_vector_rest_token}"))?;

        value.set_sensitive(true);
        let mut headers = header::HeaderMap::new();
        headers.insert(header::AUTHORIZATION, value);

        // initialize reqwest client
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .build()?;

        let url_endpoint = Url::parse(&upstash_vector_rest_url)?;

        Ok(Self {
            client,
            url_endpoint,
        })
    }

    pub async fn save_embedding(&self, embedding: &Vec<f32>) -> Result<()> {
        let (repo_name, pr_number) = (env::var("REPO_NAME")?, env::var("PR_NUMBER")?);

        let data = json!({
            "id": uuid(&repo_name,&pr_number),
            "vector": embedding,
        })
        .to_string();

        let uri = self.url_endpoint.join("upsert")?;

        let resp = self.client.post(uri).body(data).send().await?;

        if resp.status().as_u16() != 200 {
            bail!(
                "Couldn't save embedding | Reason {}",
                resp.text().await.unwrap()
            );
        }

        Ok(())
    }

    pub async fn query(&self, embedding: &Vec<f32>, top_k: u8) -> Result<SimilarPRs> {
        let data = json!({
            "topK": top_k,
            "vector": &embedding,
        })
        .to_string();

        let uri = self.url_endpoint.join("query")?;

        let resp = self.client.post(uri).body(data).send().await?;

        if resp.status().as_u16() != 200 {
            bail!(
                "Couldn't query db for similar PRs | Reason {}",
                resp.text().await.unwrap()
            );
        }

        let results = serde_json::from_str::<QueryResult>(&resp.text().await.unwrap())?;

        Ok(results.into())
    }
}

fn uuid(repo_name: &str, pr_number: &str) -> String {
    format!("[{repo_name}]:{pr_number}")
}
