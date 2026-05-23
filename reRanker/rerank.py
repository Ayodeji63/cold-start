import pandas as pd
import json
from pprint import pprint
from google import genai
from google.genai import errors as genai_errors
import json 
import os 
from tqdm import tqdm 
import time 
import numpy as np
from utils import *
import random
import argparse
import time
import socket
import httpx
import ast
import re
import tiktoken
import yaml

seed = 12
random.seed(seed)

sleep_int = 100
sleep_time = 10

def prompt(client, type_llm, inputPrompt, request_delay=0):
    if request_delay > 0:
        time.sleep(request_delay)
    if type_llm == 'gemini_pro':
        generation_config = {'candidate_count':1, "max_output_tokens": 8192, "temperature": 0, "top_p": 0.99, "top_k": 32}
        responses = client.models.generate_content(
            model="models/gemini-3.1-flash-lite",
            contents = inputPrompt,
            config=generation_config
        )
        return responses.text
    return inputPrompt
    # elif type_llm =='chatGPT':
    #     response = client.chat.completions.create(
    #         model="GPT-35-turbo-WebQSP-Vanilla", # model = "deployment_name"
    #         messages=[
    #             {"role": "user", "content": inputPrompt}
    #         ],
    #         temperature=0,
    #         max_tokens=800,
    #         top_p=0.95,
    #         frequency_penalty=0,
    #         presence_penalty=0,
    #         stop=None
    #     )
    #     return response.choices[0].message.content


### FewShot
def fewshot(model, city, type_method, type_llm, data_user_test, user_kws_train, map_rest_id2int, new_results_res_kw, num_kws_user, num_kws_rest, samples, label_samples, root_dir):

    temp_fewshot = all_prompts[type_llm][type_method]
    user_rank = dict()
    counterRequest = 0
    len_rank = None
    folder_path_result_rerank = f'{root_dir}results_rerank/{city}'
    if not os.path.exists(folder_path_result_rerank):
        os.makedirs(folder_path_result_rerank)
    file_path_ = folder_path_result_rerank+ f"/{type_method}_{num_kws_user}_{num_kws_rest}.json"
    print(file_path_)
    if os.path.isfile(file_path_):
        print("File exists")
        with open(file_path_, "r") as file:
            user_rank = json.load(file)
        len_rank = len(user_rank.keys())
    for uid in tqdm(data_user_test.keys(), total=len(data_user_test)):
        if len_rank is not None:
            if counterRequest <= len_rank:
                counterRequest += 1 
                continue
        if (counterRequest+1) % sleep_int == 0:
            time.sleep(sleep_time)
        user_kw = data_user_test[uid]['kw'][: num_kws_user]  # use 5 kws for user
        res_candidate = list(map(int,[str(map_rest_id2int[cand]) for cand in data_user_test[uid]['candidate']]))
        ### selecting randomly users for examples
        userKeys = list(user_kws_train.keys())
        user_train = random.choice(userKeys)
        user_train_2 = random.choice(userKeys)
        user_train_3 = random.choice(userKeys)

        user_train_kw = list(user_kws_train[user_train])[: num_kws_user]
        candidate_train = samples[user_train]
        labels = label_samples[user_train]
        res_candidate_train = list(map(int,[str(map_rest_id2int[cand]) for cand in candidate_train]))
        label_res = list(map(int,[str(map_rest_id2int[cand]) for cand in labels]))

        if type_method == '2_shots':
            user_train_kw_2 = list(user_kws_train[user_train_2])[: num_kws_user]
            # random.shuffle(user_train_kw_2)
            candidate_train_2 = samples[user_train_2]
            labels_2 = label_samples[user_train_2]
            res_candidate_train_2 = list(map(int,[str(map_rest_id2int[cand]) for cand in candidate_train_2]))
            label_res_2 = list(map(int,[str(map_rest_id2int[cand]) for cand in labels_2]))
        if type_method == '3_shots':
            user_train_kw_2 = list(user_kws_train[user_train_2])[: num_kws_user]
            # random.shuffle(user_train_kw_2)
            candidate_train_2 = samples[user_train_2]
            labels_2 = label_samples[user_train_2]
            res_candidate_train_2 = list(map(int,[str(map_rest_id2int[cand]) for cand in candidate_train_2]))
            label_res_2 = list(map(int,[str(map_rest_id2int[cand]) for cand in labels_2]))

            user_train_kw_3 = list(user_kws_train[user_train_3])[: num_kws_user]
            # random.shuffle(user_train_kw_3)
            candidate_train_3 = samples[user_train_3]
            labels_3 = label_samples[user_train_3]
            res_candidate_train_3 = list(map(int,[str(map_rest_id2int[cand]) for cand in candidate_train_3]))
            label_res_3 = list(map(int,[str(map_rest_id2int[cand]) for cand in labels_3]))

        if type_method == '1_shot':
            inputPrompt = temp_fewshot.format(', '.join(user_train_kw), res_candidate_train , cand_kw_fn_fewshot(user_train, new_results_res_kw,samples, map_rest_id2int, 20, num_kws_rest),label_res,
                                     ', '.join(user_kw),res_candidate, cand_kw_fn(uid, new_results_res_kw,data_user_test, map_rest_id2int, 20, num_kws_rest))
        elif type_method == '2_shots':
            inputPrompt = temp_fewshot.format(', '.join(user_train_kw), res_candidate_train , cand_kw_fn_fewshot(user_train, new_results_res_kw,samples, map_rest_id2int, 20, num_kws_rest),label_res,
                                        ', '.join(user_train_kw_2), res_candidate_train_2 , cand_kw_fn_fewshot(user_train_2, new_results_res_kw,samples, map_rest_id2int, 20, num_kws_rest),label_res_2,
                                     ', '.join(user_kw),res_candidate, cand_kw_fn(uid, new_results_res_kw,data_user_test, map_rest_id2int, 20, num_kws_rest))
        elif type_method == '3_shots':
            inputPrompt = temp_fewshot.format(', '.join(user_train_kw), res_candidate_train , cand_kw_fn_fewshot(user_train, new_results_res_kw,samples, map_rest_id2int, 20, num_kws_rest),label_res,
                                        ', '.join(user_train_kw_2), res_candidate_train_2 , cand_kw_fn_fewshot(user_train_2, new_results_res_kw,samples, map_rest_id2int, 20, num_kws_rest),label_res_2,
                                        ', '.join(user_train_kw_3), res_candidate_train_3 , cand_kw_fn_fewshot(user_train_3, new_results_res_kw,samples, map_rest_id2int, 20, num_kws_rest),label_res_3,
                                     ', '.join(user_kw),res_candidate, cand_kw_fn(uid, new_results_res_kw,data_user_test, map_rest_id2int, 20, num_kws_rest))

        predictions = prompt(model,type_llm, inputPrompt)
               
        pred = predictions.split(',')
        user_rank[uid] = list(map(int, pred))
        with open(file_path_, "w") as json_file:
            json.dump(user_rank, json_file)
    return user_rank


def extract_numbers(input_string):
    numbers = ''
    for char in input_string:
        if char.isdigit():
            numbers += char
    return numbers


def retry_delay_seconds(error, fallback=65):
    match = re.search(r"retryDelay': '(\d+)s", str(error))
    if match:
        return int(match.group(1)) + 5
    return fallback


def parse_ranking(predictions, candidates, top_k=20):
    candidate_set = set(candidates)
    ranking = []
    for value in re.findall(r"\d+", predictions or ""):
        item = int(value)
        if item in candidate_set and item not in ranking:
            ranking.append(item)
        if len(ranking) >= top_k:
            break
    for item in candidates:
        if item not in ranking:
            ranking.append(item)
        if len(ranking) >= top_k:
            break
    return ranking


def extract_json_payload(text):
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and start < end:
        return json.loads(cleaned[start:end + 1])

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and start < end:
        return json.loads(cleaned[start:end + 1])

    raise ValueError("No JSON object or array found in model output.")


def normalize_scores(score_by_item):
    if not score_by_item:
        return {}
    values = list(score_by_item.values())
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return {item: 1.0 for item in score_by_item}
    return {
        item: (score - min_score) / (max_score - min_score)
        for item, score in score_by_item.items()
    }


def parse_scored_ranking(predictions, candidates, top_k=20):
    candidate_set = set(candidates)
    score_by_item = {}
    try:
        payload = extract_json_payload(predictions)
        if isinstance(payload, dict):
            payload = payload.get("rankings", payload.get("items", payload.get("scores", [])))
        if not isinstance(payload, list):
            raise ValueError("Scored output must be a JSON list.")

        for entry in payload:
            if not isinstance(entry, dict):
                continue
            item = entry.get("id", entry.get("item_id", entry.get("restaurant_id", entry.get("hotel_id"))))
            score = entry.get("score", entry.get("relevance", entry.get("relevance_score")))
            if item is None or score is None:
                continue
            item = int(item)
            if item not in candidate_set:
                continue
            score_by_item[item] = float(score)
    except Exception as e:
        print(f"Could not parse scored JSON output: {e}. Falling back to ID-order parsing.")
        ranking = parse_ranking(predictions, candidates, top_k)
        return ranking, {item: float(top_k - idx) / max(top_k, 1) for idx, item in enumerate(ranking)}

    score_by_item = normalize_scores(score_by_item)
    ranking = sorted(score_by_item, key=lambda item: score_by_item[item], reverse=True)
    for item in candidates:
        if item not in ranking:
            ranking.append(item)
            score_by_item[item] = 0.0
        if len(ranking) >= top_k:
            break
    return ranking[:top_k], score_by_item


def hybrid_rank(candidates, llm_ranking, llm_scores, alpha, top_k, allow_pool_replacement=False):
    if alpha is None:
        return llm_ranking[:top_k]

    beta = 1.0 - alpha
    retrieval_top = candidates[:top_k]
    eligible_items = candidates if allow_pool_replacement else retrieval_top
    norm = max(len(candidates), 1)
    combined_scores = {}
    ordered_items = []
    for item in eligible_items + llm_ranking:
        if item not in ordered_items:
            ordered_items.append(item)
    ordered_items = [item for item in ordered_items if item in set(eligible_items)]

    for idx, item in enumerate(candidates):
        if item not in set(eligible_items):
            continue
        combined_scores[item] = combined_scores.get(item, 0.0) + alpha * (norm - idx) / norm
    for item in eligible_items:
        combined_scores[item] = combined_scores.get(item, 0.0) + beta * llm_scores.get(item, 0.0)

    return sorted(ordered_items, key=lambda item: combined_scores.get(item, 0.0), reverse=True)[:top_k]


def load_restaurant_metadata(metadata_file, map_rest_id2int):
    if not metadata_file or not os.path.isfile(metadata_file):
        return {}
    df = pd.read_csv(metadata_file)
    id_col = 'business_id' if 'business_id' in df.columns else 'rest_id' if 'rest_id' in df.columns else None
    if id_col is None:
        return {}

    useful_cols = [
        'name',
        'city',
        'categories',
        'cuisine_family',
        'price',
        'halal',
        'stars',
        'review_count',
        'west_african_similarity',
        'spice_signal',
        'value_signal',
        'service_signal',
        'family_style_signal',
        'semantic_profile',
    ]
    metadata = {}
    for _, row in df.iterrows():
        rest_id = row[id_col]
        if rest_id not in map_rest_id2int:
            continue
        parts = []
        for col in useful_cols:
            if col in df.columns and not pd.isna(row[col]):
                parts.append(f"{col}={row[col]}")
        metadata[map_rest_id2int[rest_id]] = "; ".join(parts)
    return metadata


def cand_meta_fn(uid_, data, map_rest_id2int, rest_metadata, topCandidates=20):
    if not rest_metadata:
        return "None"
    cand_meta = {}
    for cand in data[uid_]['candidate'][:topCandidates]:
        cand_int = map_rest_id2int[cand]
        if cand_int in rest_metadata:
            cand_meta[cand_int] = rest_metadata[cand_int]
    if not cand_meta:
        return "None"
    return ', '.join(f'{key} ({value})' for key, value in cand_meta.items())


def ensure_rest_mapping(map_rest_id2int, rest_kws, data_user_test):
    next_id = max(map_rest_id2int.values(), default=-1) + 1
    for rest_id in rest_kws.keys():
        if rest_id not in map_rest_id2int:
            map_rest_id2int[rest_id] = next_id
            next_id += 1
    for user_data in data_user_test.values():
        for rest_id in user_data.get('candidate', []):
            if rest_id not in map_rest_id2int:
                map_rest_id2int[rest_id] = next_id
                next_id += 1
    return map_rest_id2int


def build_zeroshot_template(city, rerank_mode):
    if city == 'tripAdvisor':
        target = "hotels"
        user_text = "There are the keywords that I often mention when wanting to choose hotels: {}."
        candidate_text = "The candidate hotel set for me is enclosed in square brackets, with the hotels separated by commas (Format: [hotel_id_1, hotel_id_2,...]) is: {}."
        keyword_text = "Keywords associated with candidate hotels are in the following format: hotel_id_1 (keyword 1, keyword 2,...) are {}."
        preference_text = "according to the user and candidate hotel keywords I provided above"
    elif city.startswith('naija_yelp'):
        target = "restaurants"
        user_text = "You are a Nigerian diaspora restaurant recommendation agent.\nThere are the keywords that I often mention when choosing restaurants: {}."
        candidate_text = "The candidate restaurant set for me is enclosed in square brackets, with the restaurants separated by commas (Format: [restaurant_id_1, restaurant_id_2,...]) is: {}."
        keyword_text = "Keywords associated with candidate restaurants have the following form: restaurant_id_1 (keyword 1, keyword 2,...) are {}."
        preference_text = "according to the user keywords, candidate keywords, cultural food preferences, and metadata. Consider pepper level, jollof/rice dishes, suya or grilled meat, halal needs, portion size, value for money, family-style dining, service warmth, and Nigerian or West African similarity when those signals are present"
    else:
        target = "items"
        user_text = "There are the keywords that I often mention when wanting to choose items on Amazon: {}."
        candidate_text = "The candidate item set for me is enclosed in square brackets, with the items separated by commas (Format: [item_id_1, item_id_2,...]) is: {}."
        keyword_text = "Keywords associated with candidate items have the following form: item_id_1 (keyword 1, keyword 2,...) are {}."
        preference_text = "according to the user and candidate item keywords I provided above"

    if rerank_mode == "scored":
        return f'''
        {user_text}
        {candidate_text}
        {keyword_text}
        Candidate metadata is: {{}}.
        Input: Assess relevance for every candidate {target} {preference_text}. Assign each candidate a score from 0.0 to 1.0, where 1.0 is the best match.
        Output: Return valid JSON only. Return a list of objects sorted by score descending. Each object must have exactly these keys: "id", "score". Include every candidate id once. Desired format: [{{{{"id": 1, "score": 0.91}}}}, {{{{"id": 2, "score": 0.37}}}}]
        '''

    return f'''
        {user_text}
        {candidate_text}
        {keyword_text}
        Candidate metadata is: {{}}.
        Input: Please suggest the {{}} most suitable {target} from the candidate set, {preference_text}.
        Output: Must include {{}} {target} in the candidate set. No explanation. Desired format is string: id_1, id_2, ...
        '''


def zeroshot(model, type_llm, data_user_test,  map_rest_id2int, new_results_res_kw, num_kws_user, num_kws_rest, root_dir, city, rest_metadata=None, request_delay=4.5, max_users=None, rerank_top_k=20, candidate_pool_k=20, rerank_mode="scored", hybrid_alpha=None, allow_pool_replacement=False):
    temp = build_zeroshot_template(city, rerank_mode)

    user_rank = dict()
    # user_shuffle = dict()
    folder_path_result_rerank = f'{root_dir}results_rerank/{city}'
    if not os.path.exists(folder_path_result_rerank):
        os.makedirs(folder_path_result_rerank)
    alpha_suffix = "llm" if hybrid_alpha is None else f"hybrid_alpha_{hybrid_alpha:g}"
    replacement_suffix = "replace" if allow_pool_replacement else "preserve"
    file_path_ = folder_path_result_rerank+ f"/zeroshot_{rerank_mode}_{num_kws_user}_{num_kws_rest}_pool{candidate_pool_k}_top{rerank_top_k}_{alpha_suffix}_{replacement_suffix}_{seed}.json"
    # file_shuffle = folder_path_result_rerank+ f"/shuffle_cadidate_zeroshot_{num_kws_user}_{num_kws_rest}_{seed}.json"
    if os.path.isfile(file_path_):
        print("File exists")
        with open(file_path_, "r") as file:
            user_rank = json.load(file)
        print(f"Resuming from {len(user_rank)} saved users")
    counterRequest = 0
    for uid in tqdm(data_user_test.keys(), total=len(data_user_test)):
        if uid in user_rank:
            continue
        if max_users is not None and len(user_rank) >= max_users:
            break
        counterRequest = counterRequest+1
        user_kw = data_user_test[uid]['kw'][:num_kws_user] 
        res_candidate = list(map(int,[str(map_rest_id2int[cand]) for cand in data_user_test[uid]['candidate'][:candidate_pool_k]]))
        
        # ablation random
        # random.shuffle(res_candidate)
        # user_shuffle[uid] = res_candidate
        
        prompt_args = [
            ', '.join(user_kw),
            res_candidate,
            cand_kw_fn(uid, new_results_res_kw, data_user_test, map_rest_id2int, candidate_pool_k, num_kws_rest),
            cand_meta_fn(uid, data_user_test, map_rest_id2int, rest_metadata, candidate_pool_k),
        ]
        if rerank_mode == "listwise":
            prompt_args.extend([rerank_top_k, rerank_top_k])
        inputPrompt = temp.format(*prompt_args)
        flag = False
        while flag is False:
            try:
                predictions = prompt(model, type_llm, inputPrompt, request_delay)
                print(predictions)
                flag = True
                if rerank_mode == "scored":
                    llm_rank, llm_scores = parse_scored_ranking(predictions, res_candidate, candidate_pool_k)
                    pred = hybrid_rank(res_candidate, llm_rank, llm_scores, hybrid_alpha, rerank_top_k, allow_pool_replacement)
                else:
                    pred = parse_ranking(predictions, res_candidate, rerank_top_k)
            except httpx.ReadTimeout:
                print("Timeout occurred. Connection timed out.")
                time.sleep(120)    
            except genai_errors.ClientError as e:
                delay = retry_delay_seconds(e)
                print(f"Gemini API error: {e}. Sleeping {delay}s before retry.")
                time.sleep(delay)
            except RuntimeError as e:
                print(f"{e}")
                time.sleep(120) 

        user_rank[uid] = pred
        with open(file_path_, "w") as json_file:
            json.dump(user_rank, json_file)
        # with open(file_shuffle, "w") as json_file:
        #     json.dump(user_shuffle, json_file)

    return user_rank


if __name__ == '__main__':
    listcity = ['edinburgh', 'london', 'singapore', 'amazonGrocery', 'amazonBaby', 'amazonVideo', 'naija_yelp', 'naija_yelp_cold_start', 'naija_yelp_paper']
    parser = argparse.ArgumentParser('LLM re-ranking RecSys')
    parser.add_argument('--type_method', type=str, default= 'zeroshot', help='zeroshot, 1_shot, 2_shots, 3_shots')
    parser.add_argument('--num_kws_user', type=int, default= 3)
    parser.add_argument('--num_kws_rest', type=int, default= 5)
    parser.add_argument('--city', type=str, default='singapore', help=f'choose city{listcity}')
    parser.add_argument('--type_LLM', type=str, default='gemini_pro', help='gemini_pro, chatGPT')
    parser.add_argument('--api_key', type=str, default=None, help='API key')
    parser.add_argument('--groundtruth_file', type=str, default=None, help='optional holdout labels csv for evaluation')
    parser.add_argument('--metadata_file', type=str, default=None, help='optional restaurant metadata csv for reranking')
    parser.add_argument('--test_candidate_file', type=str, default=None, help='optional data/out2LLMs test candidate json')
    parser.add_argument('--train_candidate_file', type=str, default=None, help='optional data/out2LLMs train candidate json for few-shot samples')
    parser.add_argument('--request_delay', type=float, default=4.5, help='seconds to sleep before each LLM request')
    parser.add_argument('--max_users', type=int, default=None, help='optional cap for smoke tests or budgeted runs')
    parser.add_argument('--rerank_top_k', type=int, default=20, help='number of items the LLM should return')
    parser.add_argument('--candidate_pool_k', type=int, default=50, help='number of retrieved candidates to send to the LLM reranker')
    parser.add_argument('--rerank_mode', type=str, default='scored', choices=['scored', 'listwise'], help='scored asks for JSON relevance scores; listwise asks for an ordered id list')
    parser.add_argument('--hybrid_alpha', type=float, default=None, help='optional MPG retrieval-order weight. 0.3 is a useful starting point; omit for LLM-only ranking')
    parser.add_argument('--allow_pool_replacement', action='store_true', help='allow candidates beyond rerank_top_k to replace original MPG top-k items')


    args = parser.parse_args()
    root_dir = 'reRanker/'

    # if few-shots
    with open(root_dir+"prompts.yaml", "r") as f:
        all_prompts = yaml.safe_load(f)

    run_list_kws_for_user = [args.num_kws_user]
    run_list_kws_for_rest = [args.num_kws_rest]
    list_method = [args.type_method]

    ### Load data
    city = args.city
    test_candidate_file = args.test_candidate_file or f"data/out2LLMs/{args.city}_knn2rest.json"
    train_candidate_file = args.train_candidate_file or f"data/out2LLMs/{args.city}_user2candidate.json"
    data_user_test = read_json(test_candidate_file)
    rest_kws = read_json(f"data/score/{args.city}-keywords-TFIUF.json")
    user_kws_train = read_json(train_candidate_file) #user-train

    
    ## review filesS
    if city == 'tripAdvisor':
        is_tripAdvisor = True
    else: 
        is_tripAdvisor= False
    gt_file = args.groundtruth_file or 'data/reviews/{}.csv'.format(args.city)
    gt, u2rs, map_rest_id2int = prepare_user2rests(gt_file, is_tripAdvisor = is_tripAdvisor)
    map_rest_id2int = ensure_rest_mapping(map_rest_id2int, rest_kws, data_user_test)
    metadata_file = args.metadata_file or f'data/metadata/{args.city}_restaurant_detail.csv'
    rest_metadata = load_restaurant_metadata(metadata_file, map_rest_id2int)

    new_results_res_kw = get_kw_for_rest(rest_kws, map_rest_id2int)

    # # sample for fewshot
    if '1_shot' in list_method or '2_shots' in list_method or '3_shots'in list_method:
        samples = read_json(f'data/fewshot_samples/{city}_5.json')
        label_samples = read_json(f'data/fewshot_samples/{city}_label_5.json')

    folder_path = root_dir + 'results'
    if not os.path.exists(folder_path):
        # If it doesn't exist, create it
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    file_path = f'{folder_path}/{args.city}_{seed}.txt'
    file_path_json = f'{folder_path}/{args.city}_{seed}.json'

    print('Begin run LLM tests')
    result_json = dict()
    for met in list_method:
        result_json[met] = dict()
        for num_kws_user in run_list_kws_for_user:
            result_json[met][num_kws_user] = dict()
            for num_kws_rest in run_list_kws_for_rest:
                result_json[met][num_kws_user][num_kws_rest] = dict()
                print('\nargs: ', args)
                print(f"Method: {met}, number of user keyword: {num_kws_user}, number of rest keyword: {num_kws_rest}")
                data_args = vars(args)
                with open(file_path, 'a') as file:
                # Write log
                    file.write("\n")
                    json.dump(data_args, file)
                    file.write("\n")
                    file.write(f"Method: {met}, number of user keyword: {num_kws_user}, number of rest keyword: {num_kws_rest}")
                    file.write("\n")

                if args.type_LLM == 'gemini_pro':
                    client = genai.Client(api_key=args.api_key)
                    if met in ['1_shot', '2_shots', '3_shots']:
                        user_rank = fewshot(client, city, met, args.type_LLM, data_user_test, user_kws_train, map_rest_id2int, new_results_res_kw, num_kws_user, num_kws_rest, samples, label_samples, root_dir)
                    elif met == 'zeroshot':
                        user_rank = zeroshot(
                            client,
                            args.type_LLM,
                            data_user_test,
                            map_rest_id2int,
                            new_results_res_kw,
                            num_kws_user,
                            num_kws_rest,
                            root_dir,
                            city,
                            rest_metadata,
                            args.request_delay,
                            args.max_users,
                            args.rerank_top_k,
                            args.candidate_pool_k,
                            args.rerank_mode,
                            args.hybrid_alpha,
                            args.allow_pool_replacement,
                        )
                pre, rec, f1, ndcg = evalAll(user_rank, u2rs)
                result_json[met][num_kws_user][num_kws_rest]['prec'] = pre
                result_json[met][num_kws_user][num_kws_rest]['recall'] = rec
                result_json[met][num_kws_user][num_kws_rest]['f1'] = f1
                result_json[met][num_kws_user][num_kws_rest]['ndcg'] = ndcg

                with open(file_path_json, "a") as json_file:
                    json.dump(result_json, json_file)
                with open(file_path, 'a') as file:
                    file.write(str(result_json))
                    file.write("\n")
                print('slepping ...')
