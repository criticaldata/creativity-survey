-- Single combined query for all 5 clusters, 50 papers each, deduplicated
WITH 
q1_embedding AS (
  SELECT ml_generate_embedding_result AS embedding_col
  FROM ML.GENERATE_EMBEDDING(
    MODEL models.textembed,
    (SELECT "neurosymbolic AI creativity" AS content),
    STRUCT(TRUE AS flatten_json_output)
  )
),
q1_results AS (
  SELECT
    base.pmc_id,
    base.title,
    base.author,
    LEFT(base.article_text, 5000) AS abstract,
    base.pmc_link,
    distance AS semantic_distance,
    'neurosymbolic AI creativity' AS query_cluster
  FROM VECTOR_SEARCH(
    TABLE `bigquery-public-data.pmc_open_access_commercial.articles`,
    'ml_generate_embedding_result',
    (SELECT embedding_col FROM q1_embedding),
    top_k => 100
  )
  QUALIFY ROW_NUMBER() OVER (PARTITION BY pmc_id ORDER BY distance ASC) = 1
  LIMIT 50
),

q2_embedding AS (
  SELECT ml_generate_embedding_result AS embedding_col
  FROM ML.GENERATE_EMBEDDING(
    MODEL models.textembed,
    (SELECT "psychedelic creative cognition psilocybin default mode network creativity" AS content),
    STRUCT(TRUE AS flatten_json_output)
  )
),
q2_results AS (
  SELECT
    base.pmc_id,
    base.title,
    base.author,
    LEFT(base.article_text, 5000) AS abstract,
    base.pmc_link,
    distance AS semantic_distance,
    'psychedelic creative cognition' AS query_cluster
  FROM VECTOR_SEARCH(
    TABLE `bigquery-public-data.pmc_open_access_commercial.articles`,
    'ml_generate_embedding_result',
    (SELECT embedding_col FROM q2_embedding),
    top_k => 100
  )
  QUALIFY ROW_NUMBER() OVER (PARTITION BY pmc_id ORDER BY distance ASC) = 1
  LIMIT 50
),

q3_embedding AS (
  SELECT ml_generate_embedding_result AS embedding_col
  FROM ML.GENERATE_EMBEDDING(
    MODEL models.textembed,
    (SELECT "dual process language models divergent thinking convergent thinking AI" AS content),
    STRUCT(TRUE AS flatten_json_output)
  )
),
q3_results AS (
  SELECT
    base.pmc_id,
    base.title,
    base.author,
    LEFT(base.article_text, 5000) AS abstract,
    base.pmc_link,
    distance AS semantic_distance,
    'dual process language models' AS query_cluster
  FROM VECTOR_SEARCH(
    TABLE `bigquery-public-data.pmc_open_access_commercial.articles`,
    'ml_generate_embedding_result',
    (SELECT embedding_col FROM q3_embedding),
    top_k => 100
  )
  QUALIFY ROW_NUMBER() OVER (PARTITION BY pmc_id ORDER BY distance ASC) = 1
  LIMIT 50
),

q4_embedding AS (
  SELECT ml_generate_embedding_result AS embedding_col
  FROM ML.GENERATE_EMBEDDING(
    MODEL models.textembed,
    (SELECT "compositional generation multimodal" AS content),
    STRUCT(TRUE AS flatten_json_output)
  )
),
q4_results AS (
  SELECT
    base.pmc_id,
    base.title,
    base.author,
    LEFT(base.article_text, 5000) AS abstract,
    base.pmc_link,
    distance AS semantic_distance,
    'compositional generation multimodal' AS query_cluster
  FROM VECTOR_SEARCH(
    TABLE `bigquery-public-data.pmc_open_access_commercial.articles`,
    'ml_generate_embedding_result',
    (SELECT embedding_col FROM q4_embedding),
    top_k => 100
  )
  QUALIFY ROW_NUMBER() OVER (PARTITION BY pmc_id ORDER BY distance ASC) = 1
  LIMIT 50
),

q5_embedding AS (
  SELECT ml_generate_embedding_result AS embedding_col
  FROM ML.GENERATE_EMBEDDING(
    MODEL models.textembed,
    (SELECT "mixture of experts creativity" AS content),
    STRUCT(TRUE AS flatten_json_output)
  )
),
q5_results AS (
  SELECT
    base.pmc_id,
    base.title,
    base.author,
    LEFT(base.article_text, 5000) AS abstract,
    base.pmc_link,
    distance AS semantic_distance,
    'mixture of experts creativity' AS query_cluster
  FROM VECTOR_SEARCH(
    TABLE `bigquery-public-data.pmc_open_access_commercial.articles`,
    'ml_generate_embedding_result',
    (SELECT embedding_col FROM q5_embedding),
    top_k => 100
  )
  QUALIFY ROW_NUMBER() OVER (PARTITION BY pmc_id ORDER BY distance ASC) = 1
  LIMIT 50
)

SELECT * FROM q1_results
UNION ALL
SELECT * FROM q2_results
UNION ALL
SELECT * FROM q3_results
UNION ALL
SELECT * FROM q4_results
UNION ALL
SELECT * FROM q5_results
ORDER BY query_cluster, semantic_distance ASC;