docker run -e FOR_MATCHING_FILEPATH="/app/data/c1_list.csv" -e OUTPUT_FILEPATH="/app/data/c1_list_matches.csv" -e SNOWFLAKE_USER="" -e SNOWFLAKE_PASSWORD="" \
-v $(pwd)/data:/app/data  \
 account_matcher_app:test

docker run -e FOR_MATCHING_FILEPATH="/app/data/inflow_list.csv" -e OUTPUT_FILEPATH="/app/data/inflow_list_matches.csv" -e SNOWFLAKE_USER="" -e SNOWFLAKE_PASSWORD="" \
-v $(pwd)/data:/app/data  \
 account_matcher_app:test