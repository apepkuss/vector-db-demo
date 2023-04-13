use milvus::{
    client::Client,
    data::FieldColumn,
    index::{IndexParams, IndexType},
    options::CreateCollectionOptions,
    proto::common::ConsistencyLevel,
    schema::CollectionSchemaBuilder,
    schema::FieldSchema, index::MetricType,
    collection::SearchOption,
};

use std::{collections::HashMap, vec};

use async_openai::{
    types::{CreateEmbeddingRequestArgs, Embedding},
    Client as AIClient,
};

// defualt grpc port: 19530
const MILVUS_SERVER_URL: &str = "http://localhost:19530";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    // ======== step-1: create embeddings for a document with openai api ========
    let openai_client = AIClient::new();

    let book_contents = vec![
        "Why do programmers hate nature? It has too many bugs.",
        "Why was the computer cold? It left its Windows open.",
    ];
    let book_embeddings = gen_embeddings(&openai_client, book_contents).await?;

    // ========== step-2: store the document embeddings in Milvus ==========

    // create a milvus client
    let client = Client::new(MILVUS_SERVER_URL).await?;

    // define field schemas
    let book_id =
        FieldSchema::new_primary_int64("book_id", "This is the required primary key field", true);
    let book_name =
        FieldSchema::new_varchar("book_name", "This is the required primary key field", 200);
    let book_intro =
        FieldSchema::new_float_vector("book_intro", "This is the required vector field", 1536);

    // define a collection schema
    let collection_name = "flows_network_book";
    let collection_schema =
        CollectionSchemaBuilder::new(collection_name, "a guide example for milvus rust SDK")
            .add_field(book_id)
            .add_field(book_name)
            .add_field(book_intro)
            .build()?;

    // create a collection option
    let collection_option = CreateCollectionOptions::default()
        .shard_num(2)
        .consistency_level(ConsistencyLevel::Strong);

    // create a collection
    if client.has_collection(collection_name).await? {
        client.drop_collection(collection_name).await?;
    }
    let collection = client
        .create_collection(collection_schema.clone(), Some(collection_option))
        .await?;
    
    // create a field for book name
    let book_name_column = FieldColumn::new(
        collection.schema().get_field("book_name").unwrap(),
        vec!["book1".to_owned(), "book2".to_owned()],
    );

    // create a field for book intro and store the embedding data in this field
    let mut book_data: Vec<f32> = vec![];
    for embed in book_embeddings {
        book_data.extend_from_slice(&embed.embedding);
    }
    let book_intro_column = FieldColumn::new(
        collection.schema().get_field("book_intro").unwrap(),
        book_data,
    );
    collection
        .insert(vec![book_name_column, book_intro_column], None)
        .await?;

    // create an index on book_intro for fast search
    let index_params = IndexParams::new(
        "book_intro_index".to_owned(),
        IndexType::IvfFlat,
        milvus::index::MetricType::L2,
        HashMap::from([("nlist".to_owned(), "1024".to_owned())]),
    );
    collection.create_index("book_intro", index_params).await?;

    // load collection to the local memory
    collection.load(1).await?;

    
    // confirm the collection is created successfully
    let expr = "book_name in [\"book1\", \"book2\"]";
    let columns = collection.query::<_, [&str; 0]>(expr, []).await?;
    println!(
        "row num: {}",
        columns.first().map(|c| c.len()).unwrap_or(0),
    );
    println!("book2: name: {}", columns[2].name);
    println!("book2: length: {}", columns[2].value.len());
    println!("book2: dims: {}", columns[2].dim);
    // println!("{:?}", columns);
    // println!("{:?}", columns[2]);


    // release collection from memory after query
    collection.release().await?;

    // ============== step-3: perform a vector query ==============

    // load the newly created collection to memory before query
    let collection = client.get_collection("flows_network_book").await?;
    collection.load(1).await?;

    // generate a vector from a user input text
    let text = vec!["the windows of the computer are open."];
    let embeddings = gen_embeddings(&openai_client, text).await?;
    let query_embedding = embeddings[0].embedding.clone();

    // perform vector query
    let result = collection
        .search(
            vec![query_embedding.into()],
            "book_intro",
            1,
            MetricType::L2,
            vec!["book_name"],
            &SearchOption::default(),
        )
        .await?;

    // display the search result
    println!("search result: {:?}", result.len());
    println!("result[0].field: {:?}", result[0].field);


    Ok(())
}

// fn gen_random_f32_vector(n: i64) -> Vec<f32> {
//     // use rand::prelude::*;
//     let mut data = Vec::<f32>::with_capacity(n as usize);
//     let mut rng = rand::thread_rng();
//     for _ in 0..n {
//         data.push(rng.gen());
//     }
//     data
// }

async fn gen_embeddings(client: &AIClient, text: Vec<&str>) -> Result<Vec<Embedding>, Box<dyn std::error::Error>> {
    let request = CreateEmbeddingRequestArgs::default()
        .model("text-embedding-ada-002")
        .input(text)
        .build()?;

    let response = client.embeddings().create(request).await?;

    Ok(response.data)
}
