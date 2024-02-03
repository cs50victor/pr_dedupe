// Credit : https://github.com/scrippt-tech/orca

use anyhow::{anyhow, Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::tokio::Api, Cache, Repo, RepoType};
use rayon::prelude::*;
use std::{fmt::Display, sync::{Arc, Mutex}};
use tokenizers::{PaddingParams, Tokenizer};
use tokio::sync::RwLock;

#[async_trait::async_trait]
pub trait Embedding {
    async fn generate_embeddings(&self, prompts: Vec<&str>) -> Result<EmbeddingResponse>;
}

#[derive(Debug)]
pub enum EmbeddingResponse {
    /// Bert embedding response
    Bert(Tensor),

    /// Empty response; usually used to initialize a pipeline result when
    /// no response is available.
    Empty,
}

impl EmbeddingResponse {
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        match self {
            EmbeddingResponse::Bert(embedding) => {
                // perform avg-pooling to get the embedding
                let (_n, n_tokens, _hidden_size) = embedding.dims3()?;
                let embedding = (embedding.sum(1)? / (n_tokens as f64))?;
                let embedding = embedding.to_vec2()?;

                match embedding.len() {
                    1 => Ok(embedding[0].clone()),
                    _ => Err(anyhow::anyhow!(format!(
                        "expected 1 embedding, got {}",
                        embedding.len()
                    ))),
                }
            }
            EmbeddingResponse::Empty => Err(anyhow::anyhow!("empty response does not have an embedding")),
        }
    }

    /// Get the embedding from an OpenAIEmbeddingResponse
    pub fn to_vec2(&self) -> Result<Vec<Vec<f32>>> {
        match self {
            EmbeddingResponse::Bert(embedding) => {
                // perform avg-pooling to get the embedding
                let (_n, n_tokens, _hidden_size) = embedding.dims3()?;
                let embedding = (embedding.sum(1)? / (n_tokens as f64))?;
                let embedding = embedding.to_vec2()?;

                Ok(embedding.clone())
            }
            EmbeddingResponse::Empty => Err(anyhow::anyhow!("empty response does not have an embedding")),
        }
    }

    pub fn to_tensor(&self) -> Option<Tensor> {
        match self {
            EmbeddingResponse::Bert(tensor) => Some(tensor.clone()),
            EmbeddingResponse::Empty => None,
        }
    }
}

impl Display for EmbeddingResponse {
    /// Display the response content from an EmbeddingResponse
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingResponse::Bert(response) => {
                write!(f, "{:?}", response)
            }
            EmbeddingResponse::Empty => write!(f, ""),
        }
    }
}

impl Default for EmbeddingResponse {
    /// Default EmbeddingResponse is Empty
    fn default() -> Self {
        EmbeddingResponse::Empty
    }
}

pub struct Bert {
    /// Run offline (you must have the files already cached)
    offline: bool,

    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    model_id: Option<String>,

    /// Model weights.
    model: Option<Arc<BertModel>>,

    /// Tokenizer.
    tokenizer: Option<RwLock<Tokenizer>>,

    revision: Option<String>,

    /// L2 normalization for embeddings.
    normalize_embeddings: bool,
}

impl Default for Bert {
    /// Provides default values for `Bert`.
    fn default() -> Self {
        Self {
            offline: false,
            model_id: Some("sentence-transformers/all-MiniLM-L6-v2".to_string()),
            model: None,
            tokenizer: None,
            revision: Some("refs/pr/21".to_string()),
            normalize_embeddings: false,
        }
    }
}

impl AsRef<Self> for Bert {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl Bert {
    /// Creates a new `Bert` instance with a specified prompt.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds the model and tokenizer.
    pub async fn build_model_and_tokenizer(mut self) -> Result<Self> {
        let device = match Device::new_metal(0){
            Ok(device) => device,
            Err(_) => {
                log::error!("Couldn't use Metal as default device, defaulting to CPU");
                Device::Cpu
            },
        };

        let repo = Repo::with_revision(self.model_id.clone().unwrap(), RepoType::Model, self.revision.clone().unwrap());

        // 
        let (config_filename, tokenizer_filename, weights_filename) = if self.offline {
            let cache = Cache::default().repo(repo);
            (
                cache.get("config.json").ok_or(anyhow!("Missing config file in cache"))?,
                cache.get("tokenizer.json").ok_or(anyhow!("Missing tokenizer file in cache"))?,
                cache.get("model.safetensors").ok_or(anyhow!("Missing weights file in cache"))?,
            )
        } else {
            let api = Api::new()?;
            let api = api.repo(repo);
            (
                api.get("config.json").await?,
                api.get("tokenizer.json").await?,
                api.get("model.safetensors").await?,
            )
        };
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;
        self.model = Some(Arc::new(model));
        self.tokenizer = Some(RwLock::new(tokenizer));
        Ok(self)
    }
}

#[async_trait::async_trait]
impl Embedding for Bert {
    async fn generate_embeddings(&self, prompts: Vec<&str>) -> Result<EmbeddingResponse> {

        if self.model.is_none() || self.tokenizer.is_none() {
            return Err(anyhow!("Model or tokenizer not initialized"));
        }

        let model: Arc<BertModel> = self.model.as_ref().unwrap().clone();
        let mut tokenizer: tokio::sync::RwLockWriteGuard<'_, Tokenizer> =
            self.tokenizer.as_ref().unwrap().write().await;
        let device = &model.device;

        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        let tokens = tokenizer
            .encode_batch(prompts, true)
            .map_err(E::msg)?;
        let token_ids = tokens
            .iter()
            .enumerate()
            .map(|(i, tokens)| {
                let tokens = tokens.get_ids().to_vec();
                let tensor = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
                Ok((i, tensor))
            })
            .collect::<Result<Vec<_>>>()?;

        let embeddings = vec![Tensor::ones((2, 3), candle_core::DType::F32, device)?; token_ids.len()];
        // Wrap the embeddings vector in an Arc<Mutex<_>> for thread-safe access
        let embeddings_arc = Arc::new(Mutex::new(embeddings));

        // Use rayon to compute embeddings in parallel
        log::info!("Computing embeddings");
        let start = std::time::Instant::now();
        token_ids.par_iter().try_for_each_with(embeddings_arc.clone(), |embeddings_arc, (i, token_ids)| {
            let token_type_ids = token_ids.zeros_like()?;
            let embedding = model.forward(token_ids, &token_type_ids)?.squeeze(0)?;

            // Lock the mutex and write the embedding to the correct index
            let mut embeddings = embeddings_arc.lock().map_err(|e| anyhow!("Mutex error: {}", e))?;
            embeddings[*i] = embedding;

            Ok::<(), anyhow::Error>(())
        })?;
        log::info!("Done computing embeddings");
        log::info!("Embeddings took {:?} to generate", start.elapsed());

        // Retrieve the final ordered embeddings
        let embeddings_arc = Arc::try_unwrap(embeddings_arc)
            .map_err(|_| anyhow!("Arc unwrap failed"))?
            .into_inner()
            .map_err(|e| anyhow!("Mutex error: {}", e))?;

        let stacked_embeddings = Tensor::stack(&embeddings_arc, 0)?;

        Ok(EmbeddingResponse::Bert(stacked_embeddings))
    }
}

pub async fn generate_embeddings(content: &str, max_token: usize) -> Result<Vec<f32>>{
    let bert = Bert::new().build_model_and_tokenizer().await?;

    let chunked_pr_content = chunk_large_text(content, max_token);

    Ok(bert.generate_embeddings(chunked_pr_content).await.unwrap().to_vec().unwrap())
}

fn chunk_large_text(content:&str, max_tokens: usize, ) -> Vec<&str> {
    // use huggingface tokenizer for text splitter?
    let splitter = text_splitter::TextSplitter::default().with_trim_chunks(true);
    splitter.chunks(content, max_tokens).collect::<Vec<&str>>()
}

#[cfg(test)]
mod test {
    use super::*;

    #[tokio::test]
    async fn test_batch() {
        let bert = Bert::new().build_model_and_tokenizer().await.unwrap();
        let response = bert.generate_embeddings(["Hello World", "Goodbye World"].to_vec()).await;
        let response = response.unwrap();
        let vec = response.to_vec2().unwrap();
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0].len(), 384);
        assert_eq!(vec[1].len(), 384);
    }
}
