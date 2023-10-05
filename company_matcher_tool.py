from name_matching.name_matcher import NameMatcher
import pandas as pd
import re
import unicodedata
from cleanco import basename
import snowflake_login
import os



def extract_domain_name(website_string):
    if website_string is None:
        return ''  # Return an empty string for None values
    pattern = r"(?:https?://)?(?:www\.)?([a-zA-Z0-9.-]+)\b"
    match = re.search(pattern, website_string)
    if match:
        return match.group(1)
    else:
        return ''  # Return an empty string for no match
    
    
def cleanName(company_name, suffixes:list = None):
    if suffixes == None:
        suffixes = [
            "inc",
            "corp",
            "ltd",
            "llc",
            "plc"
            "corp,"
            "co",
            "group",
            "grp",
            "nb",
            "bv",
            "sa",
            "na",
            "international",
            "intl",
            "gmbh",
            "ag",
            "holdings",
            "enterprises",
            "industries",
            "solutions",
            "services",
            "systems",
            "company",
            "companies"

        ]
        
    suffix_pattern = '|'.join(re.escape(suffix) for suffix in suffixes)
    company_base_name = basename(unicodedata.normalize('NFKD',company_name.lower()).encode('ASCII','ignore').decode())
    clean_name =  re.sub(fr'(?i)\s*({suffix_pattern})\s*$', '', company_base_name)
    return clean_name    


def match_scorer(master_dataset,matching_data):
    print('Configuring Match Scorer...')
    matcher = NameMatcher(top_n=10,
    lowercase=True,
    punctuations=True,
    remove_ascii=True,
    legal_suffixes=True,
    common_words=False,
    verbose=True)

    # adjust the distance metrics to use
    matcher.set_distance_metrics(['discounted_levenshtein', 'fuzzy_wuzzy_partial_string', 'refined_soundex'])


    print('Loading Master Data...')
    # load the data to which the names should be matched
    matcher.load_and_process_master_data(column='SFDC_NAME_CLEAN',
                                     df_matching_data=master_dataset, 
                                     transform=True)
    
        
        
    print('Performing matches...')
    # perform the name matching on the data you want matched
    matches = matcher.match_names(to_be_matched=matching_data, 
                            column_matching='COMPANY_NAME_CLEAN')
    
    # combine the datasets based on the matches
    combined = pd.merge(master_dataset, matches, how='left', left_index=True, right_on='match_index')
    combined = pd.merge(combined, matching_data, how='left', left_index=True, right_index=True)


    result = combined[['MATCHING_ID','COMPANY_NAME','COMPANY_NAME_CLEAN','WEBSITE','COUNTRY','SFDC_ID','SFDC_NAME','SFDC_NAME_CLEAN','SFDC_WEBSITE','SFDC_COUNTRY','score']]
    # result = combined[combined['score'] >= 90 ][['MATCHING_ID','COMPANY_NAME','COMPANY_NAME_CLEAN','WEBSITE','COUNTRY','SFDC_ID','SFDC_NAME','SFDC_NAME_CLEAN','SFDC_WEBSITE','SFDC_COUNTRY','score']]
    result = result.rename(columns={'score':'NAME_SCORE'})
    result = result[~result['MATCHING_ID'].isna() ]
    result['NAME_SCORE'] = (result['NAME_SCORE'] / 100) * 24
    
    


    return result

def get_best_match(master_dataset,matching_data):
    
    
    scored_set = match_scorer(master_dataset,matching_data)
    
    print('Initializing Output Dataframe...')
        # Initialize the output DataFrame
    output = pd.DataFrame(columns=[
        'MATCHING_ID', 'COMPANY_NAME', 'COMPANY_NAME_CLEAN','WEBSITE', 'COUNTRY','SFDC_ID','SFDC_NAME','SFDC_NAME_CLEAN','SFDC_WEBSITE','SFDC_COUNTRY',
        'NAME_SCORE', 'WEBSITE_SCORE', 'COUNTRY_SCORE', 'TOTAL_MATCHING_SCORE'
    ])
    
    
    print('Getting exact matches on name, website,country....')
    # Exact matches on name, website, and country
    match_candidates = matching_data[(matching_data['COMPANY_NAME'] != '') & (matching_data['WEBSITE'] != '') & (matching_data['COUNTRY'] != '')  ]
    if len(match_candidates) > 0:
        exact_matches = pd.merge(match_candidates, master_dataset, left_on=['COMPANY_NAME', 'WEBSITE', 'COUNTRY'],
                                right_on=['SFDC_NAME', 'SFDC_WEBSITE', 'SFDC_COUNTRY'], how='inner')
        
        exact_matches['NAME_SCORE'] = 25
        exact_matches['WEBSITE_SCORE'] = 15
        exact_matches['COUNTRY_SCORE'] = 10
        
        # Exclude exact matches from matching_data
        matching_data = matching_data[~matching_data['MATCHING_ID'].isin(exact_matches['MATCHING_ID'])]
    else:
        exact_matches =output
    
    
    print('Getting exact matches on website and country....')
    # Exact matches on website and country, then name similarity scoring
    match_candidates = matching_data[(matching_data['WEBSITE'] != '') & (matching_data['COUNTRY'] != '')  ]
    if len(match_candidates) > 0:
        website_country_matches = pd.merge(match_candidates, master_dataset[['SFDC_WEBSITE', 'SFDC_COUNTRY']], left_on=['WEBSITE', 'COUNTRY'],
                                        right_on=['SFDC_WEBSITE', 'SFDC_COUNTRY'], how='inner')
        
        website_country_matches = website_country_matches.merge(scored_set[['MATCHING_ID','NAME_SCORE','SFDC_ID','SFDC_NAME','SFDC_NAME_CLEAN']],how='left',suffixes=('',''),left_on=['MATCHING_ID'],right_on=['MATCHING_ID'] )
        website_country_matches['WEBSITE_SCORE'] = 15
        website_country_matches['COUNTRY_SCORE'] = 10

        # Exclude exact website/country matches from matching_data
        matching_data = matching_data[~matching_data['MATCHING_ID'].isin(website_country_matches['MATCHING_ID'])]
    else:
        website_country_matches =output
    
    print('Getting exact matches on website and name....')
    # Exact matches on website and name
    match_candidates = matching_data[(matching_data['COMPANY_NAME'] != '') & (matching_data['WEBSITE'] != '')  ]
    if len(match_candidates) > 0:
        website_name_matches = pd.merge(match_candidates, master_dataset, left_on=['WEBSITE', 'COMPANY_NAME'],
                                        right_on=['SFDC_WEBSITE', 'SFDC_NAME'], how='inner')
        website_name_matches['NAME_SCORE'] = 25
        website_name_matches['WEBSITE_SCORE'] = 15
        website_name_matches['COUNTRY_SCORE'] = 0
        
        # Exclude exact website/name matches from matching_data
        matching_data = matching_data[~matching_data['MATCHING_ID'].isin(website_name_matches['MATCHING_ID'])]
    else:
        website_name_matches = output
        
    print('Getting exact matches on country and name....')
    # Exact matches on country and name
    match_candidates = matching_data[(matching_data['COMPANY_NAME'] != '') & (matching_data['COUNTRY'] != '')  ]
    if len(match_candidates) > 0:
        country_name_matches = pd.merge(match_candidates, master_dataset, left_on=['COUNTRY', 'COMPANY_NAME'],
                                        right_on=['SFDC_COUNTRY', 'SFDC_NAME'], how='inner')
        country_name_matches['NAME_SCORE'] = 25
        country_name_matches['COUNTRY_SCORE'] = 10
        country_name_matches['WEBSITE_SCORE'] = 0
        
        # Exclude exact country/name matches from matching_data
        matching_data = matching_data[~matching_data['MATCHING_ID'].isin(country_name_matches['MATCHING_ID'])]
    else:
        country_name_matches = output
    
    
    print('Getting exact matches on website....')
    # Exact matches on website, then name similarity scoring
    match_candidates = matching_data[(matching_data['WEBSITE'] != '')   ]
    if len(match_candidates) > 0:
        website_matches = pd.merge(match_candidates, master_dataset[['SFDC_WEBSITE']], left_on=['WEBSITE'],
                                    right_on=['SFDC_WEBSITE'], how='inner')
        
        website_matches = website_matches.merge(scored_set[['MATCHING_ID','NAME_SCORE','SFDC_ID','SFDC_NAME','SFDC_NAME_CLEAN','SFDC_COUNTRY']],how='left',suffixes=('',''),left_on=['MATCHING_ID'],right_on=['MATCHING_ID'] )
        website_matches['WEBSITE_SCORE'] = 15
        website_matches['COUNTRY_SCORE'] = 0
        
        # Exclude exact website matches from matching_data
        matching_data = matching_data[~matching_data['MATCHING_ID'].isin(website_matches['MATCHING_ID'])]
    else:
        website_matches = output
    
    
    print('Getting exact matches on country....')
    # Exact matches on country, then name similarity scoring
    match_candidates = matching_data[(matching_data['COUNTRY'] != '')]
    if len(match_candidates) > 0:
        country_matches = pd.merge(match_candidates, master_dataset[['SFDC_COUNTRY']], left_on=['COUNTRY'],
                                    right_on=['SFDC_COUNTRY'], how='inner')
        
        country_matches = country_matches.merge(scored_set[['MATCHING_ID','NAME_SCORE','SFDC_ID','SFDC_NAME','SFDC_NAME_CLEAN','SFDC_WEBSITE']],how='left',suffixes=('',''),left_on=['MATCHING_ID'],right_on=['MATCHING_ID'] )
        country_matches['WEBSITE_SCORE'] = 0
        country_matches['COUNTRY_SCORE'] = 10
        
        # Exclude exact country matches from matching_data
        matching_data = matching_data[~matching_data['MATCHING_ID'].isin(country_matches['MATCHING_ID'])]
    else:
        country_matches = output
    
    
    print('Getting exact matches on name....')
    # Exact matches on name, then name similarity scoring
    name_matches = pd.merge(matching_data, master_dataset, left_on=['COMPANY_NAME'],
                                right_on=['SFDC_NAME'], how='inner')
    name_matches['NAME_SCORE'] = 25
    name_matches['WEBSITE_SCORE'] = 0
    name_matches['COUNTRY_SCORE'] = 0
    
    # Exclude exact name matches from matching_data
    matching_data = matching_data[~matching_data['MATCHING_ID'].isin(name_matches['MATCHING_ID'])]
    
    
    print('Getting partial match scores on name....')
    partial_matches = pd.merge(matching_data, scored_set[['MATCHING_ID','NAME_SCORE','SFDC_ID','SFDC_NAME','SFDC_NAME_CLEAN','SFDC_WEBSITE','SFDC_COUNTRY']], left_on=['MATCHING_ID'],
                                right_on=['MATCHING_ID'], how='left',suffixes=('',''))
    partial_matches['WEBSITE_SCORE']=0
    partial_matches['COUNTRY_SCORE']=0
    
    
    print('Consolidating matches...')
    all_matches = pd.concat([output,partial_matches,name_matches,country_matches,website_matches,website_country_matches,website_name_matches,country_name_matches,exact_matches],ignore_index=True)
    all_matches['NAME_SCORE'] = all_matches['NAME_SCORE'].fillna(0)
    all_matches['WEBSITE_SCORE'] = all_matches['WEBSITE_SCORE'].fillna(0)
    all_matches['COUNTRY_SCORE'] = all_matches['COUNTRY_SCORE'].fillna(0)
    
    
    
    print('Getting Total Scores...')
    all_matches['TOTAL_MATCHING_SCORE'] = all_matches['NAME_SCORE'] + all_matches['WEBSITE_SCORE'] + all_matches['COUNTRY_SCORE']
    
    print('Getting top score per record')
    # sorted_all_matches = all_matches.sort_values(by=['MATCHING_ID','TOTAL_MATCHING_SCORE'],ascending=[True, False])
    # max_score_records = sorted_all_matches.groupby('MATCHING_ID').first()
    # Group by MATCHING_ID and select the record with the highest TOTAL_MATCHING_SCORE
    # max_score_records = all_matches.groupby('MATCHING_ID').apply(lambda group: group.loc[group['TOTAL_MATCHING_SCORE'].idxmax()])
    max_score_records = all_matches.groupby('MATCHING_ID')['TOTAL_MATCHING_SCORE'].idxmax().apply(lambda idx: all_matches.loc[idx])

    
    
    output = max_score_records.reset_index(drop=True)
    output = output[['MATCHING_ID','COMPANY_NAME','COMPANY_NAME_CLEAN','WEBSITE','COUNTRY','SFDC_ID','SFDC_NAME','SFDC_NAME_CLEAN','SFDC_WEBSITE','SFDC_COUNTRY','NAME_SCORE','WEBSITE_SCORE','COUNTRY_SCORE','TOTAL_MATCHING_SCORE']]
    

    return output

def account_matcher(master_dataset,matching_data,chunk_size:int = None):
    
    output = pd.DataFrame(columns=[
        'MATCHING_ID', 'COMPANY_NAME', 'COMPANY_NAME_CLEAN','WEBSITE', 'COUNTRY','SFDC_ID','SFDC_NAME','SFDC_NAME_CLEAN','SFDC_WEBSITE','SFDC_COUNTRY',
        'NAME_SCORE', 'WEBSITE_SCORE', 'COUNTRY_SCORE', 'TOTAL_MATCHING_SCORE'
    ])
    
    master_dataset['SFDC_NAME'] = master_dataset['ACCOUNT_NAME'].apply(str.upper).apply(str.strip)
    master_dataset['SFDC_NAME_CLEAN'] = master_dataset['ACCOUNT_NAME'].apply(str.strip).apply(cleanName)
    master_dataset['SFDC_WEBSITE'] = master_dataset['WEBSITE'].apply(extract_domain_name)
    master_dataset = master_dataset[['SFDC_ID','SFDC_NAME','SFDC_NAME_CLEAN','SFDC_WEBSITE','SFDC_COUNTRY']]
    
    matching_data = matching_data.where(pd.notna(matching_data),'')
    matching_data['COMPANY_NAME'] = matching_data['COMPANY_NAME'].astype('object')
    matching_data['COMPANY_NAME'] = matching_data['COMPANY_NAME'].apply(str.upper).apply(str.strip)
    matching_data['COMPANY_NAME_CLEAN'] = matching_data['COMPANY_NAME'].apply(str.strip).apply(cleanName)
    matching_data['WEBSITE'] = matching_data['WEBSITE'].astype('object')
    matching_data['COUNTRY'] = matching_data['COUNTRY'].astype('object')

    if chunk_size == None:
        chunk_size = len(matching_data) + 1
    
    for start in range(0,len(matching_data),chunk_size):
        chunk = matching_data[start:start  + chunk_size]
        print(f'Processing chunk {start//chunk_size+1}...')
        chunk_output = get_best_match(master_dataset,chunk)
        
        output = pd.concat([output,chunk_output],ignore_index=True)
    
    
    print('Exporting results...')
    output.to_csv(output_file_path,index=False)
    print('Success!')
    

if __name__ == '__main__':
    snow = snowflake_login.connect()
    
    account_option = pd.DataFrame(snow.execute_dict('''
        select an.ACCOUNT_ID SFDC_ID,
       an.ACCOUNT_NAME,
       al.WEBSITE,
       al.COUNTRY_NAME SFDC_COUNTRY
        from MARKETINGOPERATIONS.DM_CORE.T_RPT_ACCOUNT_NAME_OPTION an
        left join MARKETINGOPERATIONS.DM_CORE.T_RPT_ACCOUNTLEAD al
            on an.ACCOUNT_ID = al.ID
            ORDER BY al.COUNTRY_NAME
                                                '''
                                                ))
    
    for_matching_file_path = os.getenv('FOR_MATCHING_FILEPATH')
    output_file_path = os.getenv('OUTPUT_FILEPATH')
    chunk_size = os.getenv('BATCH_CHUNK_SIZE')
    
    for_matching = pd.read_csv(for_matching_file_path)
    
    account_matcher(master_dataset=account_option,matching_data=for_matching,chunk_size=chunk_size)