#Linux

docker run -e FOR_MATCHING_FILEPATH="/app/data/test.csv" -e OUTPUT_FILEPATH="/app/data/output.csv" -e SNOWFLAKE_USER="TABLEAU_SERVER_MARKETINGOPERATIONS" -e SNOWFLAKE_PASSWORD="DATteam0ps1" \
-v $(pwd)/data:/app/data  \
 account_matcher_tool:test

docker run -e FOR_MATCHING_FILEPATH="/app/data/inflow_list.csv" -e OUTPUT_FILEPATH="/app/data/inflow_list_matches.csv" -e SNOWFLAKE_USER="" -e SNOWFLAKE_PASSWORD="" \
-v $(pwd)/data:/app/data  \
 account_matcher_app:test

#Windows
 docker run -e "FOR_MATCHING_FILEPATH=/app/data/test.csv" ^
 -e "OUTPUT_FILEPATH=/app/data/output.csv" ^
 -e "SNOWFLAKE_USER=" ^
 -e "SNOWFLAKE_PASSWORD=" ^
 -v "C:/Users/Charles Tan/Documents/GitHub/mops-account-matcher-tool/data:/app/data" ^
 account_matcher_tool:test
