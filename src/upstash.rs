use std::env;

use anyhow::{bail, Result};

use log::info;
use reqwest::{header, Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    utils::{uuid, uuid_to_pr_number, uuid_to_repo_name, VectorDB},
    SimilarPRs, SimilarPRsInner,
};

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
        let repo_name = env::var("REPO_NAME").unwrap();

        // ask upstash team to provide feature using api?
        let add_pr_prefix =
            |pr_number: &str| format!("https://github.com/{}/pull/{pr_number}", &repo_name,);

        let curr_pr = add_pr_prefix(&env::var("PR_NUMBER").unwrap());

        SimilarPRs {
            data: val
                .result
                .iter()
                //
                .filter(|d| {
                    repo_name == uuid_to_repo_name(&d.id)
                        && add_pr_prefix(uuid_to_pr_number(&d.id)) != curr_pr
                })
                .map(|d| {
                    let pr_number = uuid_to_pr_number(&d.id);
                    SimilarPRsInner {
                        pr_url: add_pr_prefix(pr_number),
                        percentage: d.score * 100.0,
                    }
                })
                .collect::<Vec<_>>(),
        }
    }
}

impl Upstash {
    pub fn new() -> Result<Self> {
        let (upstash_vector_rest_url, upstash_vector_rest_token) = (
            env::var("UPSTASH_VECTOR_REST_URL"),
            env::var("UPSTASH_VECTOR_REST_TOKEN"),
        );

        if upstash_vector_rest_url.is_err() || upstash_vector_rest_token.is_err() {
            bail!("both UPSTASH_VECTOR_REST_URL and UPSTASH_VECTOR_REST_TOKEN env variables need to use supabase's vector database");
        }

        let (upstash_vector_rest_url, upstash_vector_rest_token) = (
            upstash_vector_rest_url.unwrap(),
            upstash_vector_rest_token.unwrap(),
        );

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
}

impl VectorDB for Upstash {
    async fn save_embedding(&self, embedding: &[f32]) -> Result<()> {
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

    async fn remove_pr(&self) -> Result<()> {
        let (repo_name, pr_number) = (env::var("REPO_NAME")?, env::var("PR_NUMBER")?);

        let data = format!("{:?}", [uuid(&repo_name, &pr_number)]);

        println!("data {data}");

        let uri = self.url_endpoint.join("delete")?;

        let resp = self.client.delete(uri).body(data).send().await?;

        let status = resp.status();
        let resp_data = resp.text().await.unwrap();

        if status.as_u16() != 200 {
            bail!(
                "Couldn't remove PR embedding from vector db | Reason {}",
                resp_data
            );
        }
        info!("response data after removing PR from db, {resp_data}");

        Ok(())
    }

    async fn query(&self, embedding: &[f32], top_k: u8, min_similarity: u8) -> Result<SimilarPRs> {
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

        // ask upstash team to provide feature using api?
        let mut similar_prs: SimilarPRs = results.into();
        similar_prs
            .data
            .retain(|d| d.percentage >= min_similarity as f32);
        Ok(similar_prs)
    }
}
