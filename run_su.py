import argparse
import os
import pathlib
import pickle
from lib2to3.pgen2.tokenize import tokenize
import csv

import config
import datasets
import evaluate
import accelerate
from accelerate import Accelerator
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import json
import pandas as pd
import sklearn
import datasets
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification


parser = argparse.ArgumentParser()
parser.add_argument('--type_of_question', type=str)
parser.add_argument('--num_generations_per_prompt', type=int, default=5)
parser.add_argument('--fraction_of_data_to_use', type=float, default=0.9)
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--temperature', type=float, default='0.5')
parser.add_argument('--num_beams', type=int, default='5')
parser.add_argument('--model', type=str, default='opt-2.7b')
parser.add_argument('--decoding_method', type=str, default='beam_search')
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--device', type=str, default="cuda:2")
parser.add_argument('--encoder_device', type=str, default="cuda:3")
args = parser.parse_args()

print("INFO: ------setting device------")

device = args.device
encoder_device = args.encoder_device
'''if torch.cuda.is_available():
    device = 'cuda' #adjust to gpu number
elif torch.device('mps') != None:
    device = 'mps'
else:
    device = 'cpu'''

dtype = torch.float16
run_id = args.run_id

if device == 'mps' or device == 'cpu':
    dtype = torch.bfloat16

wandb.login()
wandb.init(project='nlg_uncertainty', id=run_id, config=args, resume='allow')

run_name = wandb.run.name


# Set a seed value
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

#Fix torch random seed
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

encoder_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
encoder_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(encoder_device)
print("INFO: ------loading data------")

with open(f'{config.data_dir}/coqa-dev-v1.0.json', 'r') as infile:
    data = json.load(infile)['data']

rouge = evaluate.load('rouge')



dataset = {}

if not os.path.exists(f'{config.data_dir}/coqa_dataset'):

    if args.fraction_of_data_to_use < 1.0:
        data = data[:int(len(data) * args.fraction_of_data_to_use)]


    dataset['story'] = []
    dataset['question'] = []
    dataset['answer'] = []
    dataset['additional_answers'] = []
    dataset['rouge1'] = []
    dataset['rouge2'] = []
    dataset['rougeL'] = []
    dataset['semantic_variability'] = []
    dataset['id'] = []

    for sample_id, sample in enumerate(tqdm(data, desc='Encoding data')):
        story = sample['story']
        questions = sample['questions']
        answers = sample['answers']
        additional_answers = sample['additional_answers']
        for question_index, question in enumerate(questions):
            dataset['story'].append(story)
            dataset['question'].append(question['input_text'])
            dataset['answer'].append({
                'text': answers[question_index]['input_text'],
                'answer_start': answers[question_index]['span_start'] # where answer starts in story
            })
            dataset['id'].append(sample['id'] + '_' + str(question_index))
            additional_answers_list = []

            for i in range(3):
                additional_answers_list.append(additional_answers[str(i)][question_index]['input_text'])

            dataset['additional_answers'].append(additional_answers_list)
            story = story + ' Q: ' + question['input_text'] + ' A: ' + answers[question_index]['input_text']
            if not story[-1] == '.':
                story = story + '.'
            all_answers = [answers[question_index]['input_text']] + additional_answers_list

            answer_list_1 = []
            answer_list_2 = []
            has_semantically_different_answers = False
            inputs = []

            # This computes the syntactic similarity across the reference answers
            for i, reference_answer in enumerate(all_answers):
                for j in range(4):
                    if i != j:
                        answer_list_1.append(all_answers[i])
                        answer_list_2.append(all_answers[j])

                        qa_1 = question['input_text'] + ' ' + all_answers[i]
                        qa_2 = question['input_text'] + ' ' + all_answers[j]

                        input = qa_1 + ' [SEP] ' + qa_2

                        inputs.append(input)
                        #print(encoded_input)

            encoded_input = encoder_tokenizer.batch_encode_plus(inputs, padding=True)

            prediction = encoder_model(torch.tensor(encoded_input['input_ids'], device=encoder_device))['logits']

            predicted_label = torch.argmax(prediction, dim=1)
            if 0 in predicted_label:
                has_semantically_different_answers = True

            dataset['semantic_variability'].append(has_semantically_different_answers)

            results = rouge.compute(predictions=answer_list_1, references=answer_list_2)
            #print(results)
            dataset['rouge1'].append(results['rouge1']) # changed to value only
            dataset['rouge2'].append(results['rouge2'])
            dataset['rougeL'].append(results['rougeL'])

    dataset_df = pd.DataFrame.from_dict(dataset)

    dataset = Dataset.from_pandas(dataset_df)

    dataset.save_to_disk(f'{config.data_dir}/coqa_dataset')

    print("INFO: ------saved data to disk------")

if not os.path.exists(f'{config.output_dir}/sequences/' + run_name):
    print("INFO: ------loading model------")

    os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

    #os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

    model = AutoModelForCausalLM.from_pretrained(f"facebook/{args.model}",
                                                torch_dtype=dtype,
                                                device_map="auto",
                                                cache_dir=config.hf_cache_dir)#.cuda()

accelerator = Accelerator()
#accelerate.dispatch_model(model, device_map=config.device_map)
device = accelerator.device



tokenizer = AutoTokenizer.from_pretrained(f"facebook/{args.model}", use_fast=False, cache_dir=config.hf_cache_dir)

opt_models = ['opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b', 'opt-13b', 'opt-30b'] 

dataset = datasets.load_from_disk(f'{config.data_dir}/coqa_dataset')
id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))

if args.fraction_of_data_to_use < 1.0:
    train_dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed_value)['train']
else:
    train_dataset = dataset


def encode(examples):
    return tokenizer(examples['story'] + ' Q: ' + examples['question'] + ' A:', truncation=False, padding=False)


def encode_and_format_dataset(dataset):
    dataset = dataset.map(encode, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset


period_token_id = tokenizer('. ')['input_ids'][1]
eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
question_framing_ids = [[tokenizer(eos_token)['input_ids'][1]] for eos_token in eos_tokens]
squad_metric = evaluate.load("squad")
rouge = evaluate.load('rouge')
exact_match_metric = evaluate.load("exact_match")


def get_generations(model, dataloader, number_of_generations, sequences = [], temp_generations_path = f'{config.output_dir}/sequences/temp/generations.pkl'):
    """For a given model, produce a number of generation """

    with torch.no_grad():
        max_length_of_generated_sequence = 128
        try:
            for batch in tqdm(dataloader, desc='Generating sequences'):

                input_ids = batch['input_ids'].to(device)
                if args.decoding_method == 'beam_search':
                    most_likely_generation = model.generate(input_ids,
                                                            num_beams=5,
                                                            num_return_sequences=2,
                                                            do_sample=False,
                                                            max_length=input_ids.shape[1] +
                                                            max_length_of_generated_sequence,
                                                            eos_token_id=period_token_id,
                                                            bad_words_ids=question_framing_ids)
                elif args.decoding_method == 'greedy':
                    most_likely_generation = model.generate(input_ids,
                                                            num_beams=1,
                                                            do_sample=False,
                                                            max_length=input_ids.shape[1] +
                                                            max_length_of_generated_sequence,
                                                            eos_token_id=period_token_id,
                                                            bad_words_ids=question_framing_ids)

                input_length = batch['input_ids'].shape[1]
                generations = torch.ones((number_of_generations, input_length + max_length_of_generated_sequence),
                                        dtype=torch.long,
                                        device=device)
                for i in range(number_of_generations):

                    generation = model.generate(input_ids,
                                                do_sample=True,
                                                num_return_sequences=1,
                                                num_beams=args.num_beams,
                                                max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                                                eos_token_id=period_token_id,
                                                temperature=args.temperature,
                                                bad_words_ids=question_framing_ids,
                                                top_p=args.top_p)
                    generations[i, :generation.shape[1]] = generation

                generations = torch.reshape(generations, (-1, number_of_generations, generations.shape[-1]))
                for i in range(generations.shape[0]):
                    sequence_dict = {
                        'prompt': batch['input_ids'][i].to('cpu'),
                        'generations': generations[i].to('cpu'),
                        'id': batch['id'],
                        'question': id_to_question_mapping[batch['id'][0]]
                    }


                    # Decode the generations
                    generated_texts = []
                    for generation in generations[i]:
                        generated_texts.append(
                            tokenizer.decode(generation[len(batch['input_ids'][i]):], skip_special_tokens=True))
                        
                    # TODO: cannot directly send list to cpu, check everything in sequence_dict and send to cpu
                    most_likely_generation = most_likely_generation.to('cpu')

                    sequence_dict['generated_texts'] = generated_texts
                    wandb.log({'generated_texts': str(generated_texts)})
                    sequence_dict['most_likely_generation_ids'] = most_likely_generation[0].to('cpu')
                    sequence_dict['most_likely_generation'] = tokenizer.decode(
                        most_likely_generation[0][len(batch['input_ids'][i]):], skip_special_tokens=True)

                    sequence_dict['second_most_likely_generation_ids'] = most_likely_generation[1].to('cpu')
                    sequence_dict['second_most_likely_generation'] = tokenizer.decode(
                        most_likely_generation[1][len(batch['input_ids'][i]):], skip_special_tokens=True)

                    sequence_dict['semantic_variability_reference_answers'] = batch[
                        'semantic_variability'] if 'semantic_variability' in batch else None
                    rouge_types = ['rouge1', 'rouge2', 'rougeL']
                    for rouge_type in rouge_types:
                        if rouge_type in batch:
                            sequence_dict[rouge_type + '_reference_answers'] = batch[rouge_type]

                        else:
                            sequence_dict[rouge_type + '_reference_answers'] = None

                        sequence_dict[rouge_type + '_to_target'] = 0.0

                    sequence_dict['answer'] = batch['answer']['text']
                    sequence_dict['additional_answers'] = [x[0] for x in batch['additional_answers']]

                    sequence_dict['exact_match'] = 0.0

                    reference_answers = batch['answer']['text'] + [x[0] for x in batch['additional_answers']]
                    # evaluating with exact match and rouge against the reference answers from dataset
                    for answer in reference_answers:
                        predictions = [sequence_dict['most_likely_generation'].lstrip()]
                        references = [answer]
                        results = exact_match_metric.compute(predictions=predictions,
                                                            references=references,
                                                            ignore_case=True,
                                                            ignore_punctuation=True)
                        sequence_dict['exact_match'] = max(results['exact_match'], sequence_dict['exact_match'])
                        rouge_results = rouge.compute(predictions=predictions, references=references)
                        for rouge_type in rouge_types:
                            sequence_dict[rouge_type + '_to_target'] = max(rouge_results[rouge_type],
                                                                        sequence_dict[rouge_type + '_to_target'])

                    sequences.append(sequence_dict)
                del generations
        except Exception as ex:
            print(ex)
            if os.path.exists(temp_generations_path):
                os.remove(temp_generations_path)
            else:
                pathlib.Path(f'{config.output_dir}/sequences/temp/').mkdir(parents=True, exist_ok=True)

            with open(temp_generations_path, 'wb') as outfile:
                    pickle.dump(sequences, outfile)

    return sequences

# check if generations already exist
# if so, load them
# else, generate them
if os.path.exists(f'{config.output_dir}/sequences/' + run_name):
    print("INFO: ------loading generations------")
    with open(f'{config.output_dir}/sequences/{run_name}/{args.model}_generations.pkl', 'rb') as infile:
        sequences = pickle.load(infile)
else:
    print("INFO: ------generating sequences------")
    temp_generations_path = f'{config.output_dir}/sequences/temp/generations.pkl'
    if os.path.exists(temp_generations_path):
        sequences = []
        with open(temp_generations_path, 'rb') as infile:
            sequences = pickle.load(infile)
        last_index = len(sequences) + 1
        train_subset = train_dataset.select(range(last_index, len(train_dataset))) 
        questions = encode_and_format_dataset(train_subset)
        dataloader = torch.utils.data.DataLoader(questions, batch_size=1)
        model, dataloader = accelerator.prepare(model, dataloader)
        sequences = get_generations(model, dataloader, args.num_generations_per_prompt, sequences, temp_generations_path)
    else:
        questions = encode_and_format_dataset(train_dataset)
        dataloader = torch.utils.data.DataLoader(questions, batch_size=1)
        model, dataloader = accelerator.prepare(model, dataloader)
        sequences = get_generations(model, dataloader, args.num_generations_per_prompt)
    pathlib.Path(f'{config.output_dir}/sequences/' + run_name).mkdir(parents=True, exist_ok=True)
    with open(f'{config.output_dir}/sequences/{run_name}/{args.model}_generations.pkl', 'wb') as outfile:
        pickle.dump(sequences, outfile)


generation_tokenizer = tokenizer

tokenizer = AutoTokenizer.from_pretrained(f"facebook/opt-350m", use_fast=False, cache_dir=config.data_dir) # TODO: change to 1.3B only? Different tokenizers?

#with open(f'{config.output_dir}/{run_name}/{args.generation_model}_generations.pkl', 'rb') as infile:
#    sequences = pickle.load(infile)

cleaned_sequences = []

if os.path.exists(f'{config.output_dir}/{run_name}/{args.model}_generations.pkl'):
    print("INFO: ------loading cleaned generations------")
    with open(f'{config.output_dir}/{run_name}/{args.model}_generations.pkl', 'rb') as infile:
        cleaned_sequences = pickle.load(infile)
else:
    print("INFO: ------saving cleaned generations------")
    for sample in tqdm(sequences, desc='Cleaning generations'):
        cleaned_generations = torch.ones_like(sample['generations'])
        question = sample['question']
        generated_texts = sample['generated_texts']
        cleaned_generated_texts = []

        max_len_of_generations = cleaned_generations.shape[-1]

        strings_to_filter_on = [
            '.', '\n', 'Q:', 'A:', 'question:', 'answer:', 'Question:', 'Answer:', 'Questions:', 'questions:', 'QUESTION:',
            'ANSWER:'
        ]

        for i, generated_text in enumerate(generated_texts):
            for string in strings_to_filter_on:
                if string in generated_text:
                    generated_text = generated_text.split(string)[0]
            cleaned_generated_texts.append(generated_text)
            clean_ids = torch.cat(
                [sample['prompt'].to(device),
                torch.tensor(tokenizer(generated_text)['input_ids'][1:], device=device)])
            cleaned_generations[i, :min(len(clean_ids), max_len_of_generations)] = clean_ids[:max_len_of_generations]

        sample['cleaned_generated_texts'] = cleaned_generated_texts
        sample['cleaned_generations'] = cleaned_generations
        cleaned_sequences.append(sample)

    pathlib.Path(f'{config.output_dir}/{run_name}/' + run_name).mkdir(parents=True, exist_ok=True)
    with open(f'{config.output_dir}/{run_name}/{args.model}_generations.pkl', 'wb') as outfile:
        pickle.dump(cleaned_sequences, outfile)


#tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
#model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(device)


#with open(f'{config.output_dir}/{run_name}/{args.generation_model}_generations.pkl', 'rb') as infile:
#   sequences = pickle.load(infile)

sequences = cleaned_sequences
result_dict = {}

if os.path.exists(f'{config.output_dir}/{run_name}/{args.model}_generations_similarities.pkl'):
    print("INFO: ------loading semantic and syntactic similarities------")
    with open(f'{config.output_dir}/{run_name}/{args.model}_generations_similarities.pkl', 'rb') as infile:
        result_dict = pickle.load(infile)
else:
    meteor = evaluate.load('meteor') # TODO: check meteor docs

    deberta_predictions = []

    for sample in tqdm(sequences, desc='Computing semantic entropy and syntactic'):
        question = sample['question']
        if 'cleaned_generated_texts' in sample:
            generated_texts = sample['cleaned_generated_texts']
        else:
            generated_texts = sample['generated_texts']

        id_ = sample['id'][0]

        unique_generated_texts = list(set(generated_texts))

        answer_list_1 = []
        answer_list_2 = []
        has_semantically_different_answers = False
        inputs = []
        syntactic_similarities = {}
        rouge_types = ['rouge1', 'rouge2', 'rougeL']
        for rouge_type in rouge_types:
            syntactic_similarities[rouge_type] = 0.0

        semantic_set_ids = {}
        for index, answer in enumerate(unique_generated_texts):
            semantic_set_ids[answer] = index

        #print('Number of unique answers:', len(unique_generated_texts))

        if len(unique_generated_texts) > 1:

            # Evaluate semantic similarity
            for i, reference_answer in enumerate(unique_generated_texts):
                for j in range(i + 1, len(unique_generated_texts)):

                    answer_list_1.append(unique_generated_texts[i])
                    answer_list_2.append(unique_generated_texts[j])

                    qa_1 = question + ' ' + unique_generated_texts[i]
                    qa_2 = question + ' ' + unique_generated_texts[j]

                    input = qa_1 + ' [SEP] ' + qa_2
                    inputs.append(input)
                    encoded_input = encoder_tokenizer.encode(input, padding=True)
                    prediction = encoder_model(torch.tensor(torch.tensor([encoded_input]), device=encoder_device))['logits']
                    predicted_label = torch.argmax(prediction, dim=1)

                    reverse_input = qa_2 + ' [SEP] ' + qa_1
                    encoded_reverse_input = encoder_tokenizer.encode(reverse_input, padding=True)
                    reverse_prediction = encoder_model(torch.tensor(torch.tensor([encoded_reverse_input]), device=encoder_device))['logits']
                    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)

                    deberta_prediction = 1
                    # print(qa_1, qa_2, predicted_label, reverse_predicted_label)
                    if 0 in predicted_label or 0 in reverse_predicted_label:
                        has_semantically_different_answers = True
                        deberta_prediction = 0

                    else:
                        semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]

                    deberta_predictions.append([unique_generated_texts[i], unique_generated_texts[j], deberta_prediction])

            rouge = evaluate.load('rouge')

            # Evaluate syntactic similarity
            answer_list_1 = []
            answer_list_2 = []
            for i in generated_texts:
                for j in generated_texts:
                    if i != j:
                        answer_list_1.append(i)
                        answer_list_2.append(j)

            results = rouge.compute(predictions=answer_list_1, references=answer_list_2)

            for rouge_type in rouge_types:
                syntactic_similarities[rouge_type] = results[rouge_type]

        result_dict[id_] = {
            'syntactic_similarities': syntactic_similarities,
            'has_semantically_different_answers': has_semantically_different_answers
        }
        list_of_semantic_set_ids = [semantic_set_ids[x] for x in generated_texts]
        result_dict[id_]['semantic_set_ids'] = list_of_semantic_set_ids

    with open('deberta_predictions_{}.csv'.format(args.run_id), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(['qa_1', 'qa_2', 'prediction'])
        writer.writerows(deberta_predictions)


    with open(f'{config.output_dir}/{run_name}/{args.model}_generations_similarities.pkl', 'wb') as outfile:
        pickle.dump(result_dict, outfile)

### p(true)
    
print("INFO:------calculating p(true)------")

if os.path.exists(f'{config.output_dir}/{run_name}/{args.model}_p_true_aurocs.pkl'):
    print("INFO:------loading p_true_aurocs------")
    with open(f'{config.output_dir}/{run_name}/{args.model}_p_true_aurocs.pkl', 'rb') as infile:
        p_true_auroc = pickle.load(infile)
else:

    with open(f'{config.output_dir}/{run_name}/{args.model}_generations.pkl', 'rb') as infile:
        sequences_for_few_shot_prompt = pickle.load(infile)


    # Build few shot prompt

    subset_of_sequences_for_few_shot_prompt = sequences_for_few_shot_prompt[-10:]
    number_of_few_shot_samples = 5

    prompt_template = 'Question: {} \n Here are some ideas that were brainstormed:{}\n Possible answer:{}\n Is the possible answer:\n (A) True\n (B) False\n The possible answer is:'
    few_shot_promopt = ''
    for sequence in subset_of_sequences_for_few_shot_prompt:
        question = sequence['question']
        question = question.split('Question: ')[-1].split('Answer: ')[0]
        prompt = sequence['prompt']
        generated_texts = '\n'.join(sequence['cleaned_generated_texts'][:number_of_few_shot_samples])

        most_likely_answer = sequence['most_likely_generation']
        correct = ' True' if sequence['rougeL_to_target'] > 0.3 else ' False'
        few_shot_promopt += prompt_template.format(question, generated_texts, most_likely_answer) + correct + '\n'

    # Build prompt for question
    labels_across_datasets = []
    p_trues_across_datasets = []

    n_samples_to_use = 2000

    with torch.no_grad():

        aurocs = []
        p_trues = []
        corrects = []
        for sequence in tqdm(sequences_for_few_shot_prompt[:n_samples_to_use]):

            question = sequence['question']
            if 'Question: ' in question:
                question = question.split('Question: ')[-1].split('Answer: ')[0]
            else:
                question = question.split('Q: ')[-1].split('A: ')[0]

            generated_texts = '\n'.join(sequence['cleaned_generated_texts'][:number_of_few_shot_samples])
            most_likely_answer = sequence['most_likely_generation']
            correct = 1.0 if sequence['rougeL_to_target'] > 0.3 else 0.0
            base_prompt = prompt_template.format(question, generated_texts, most_likely_answer)
            prompt_true = few_shot_promopt + prompt_template.format(question, generated_texts, most_likely_answer) + ' True'

            # This computation of the negative log likelihoods follows this tutorial: https://huggingface.co/docs/transformers/perplexity
            tokenized_base_prompt = generation_tokenizer(base_prompt)['input_ids']
            tokenized_prompt_true = torch.tensor(generation_tokenizer(prompt_true)['input_ids'], device=device)

            target_ids_true = tokenized_prompt_true.clone()
            target_ids_true[:len(tokenized_base_prompt)] = -100

            model_output_true = model(torch.reshape(tokenized_prompt_true, (1, -1)), labels=target_ids_true)
            loss_true = model_output_true.loss

            p_trues.append(loss_true.item())
            corrects.append(correct)

            labels_across_datasets += corrects
            p_trues_across_datasets += p_trues

        p_true_auroc = sklearn.metrics.roc_auc_score(1 - torch.tensor(corrects), torch.tensor(p_trues))

        # Store p_true aurocs in a pickle file
        with open(f'{config.output_dir}/{run_name}/{args.model}_p_true_aurocs.pkl', 'wb') as outfile:
            pickle.dump(p_true_auroc, outfile)

### likelihoods

if os.path.exists(f'{config.output_dir}/{run_name}/{args.model}_generations_{args.model}_likelihoods.pkl'):
    print("INFO:-----loading likelihoods------")
    with open(f'{config.output_dir}/{run_name}/{args.model}_generations_{args.model}_likelihoods.pkl', 'rb') as infile:
        likelihoods = pickle.load(infile)
else:

    print("INFO:-----calculating likelihoods------")


    with open(f'{config.output_dir}/{run_name}/{args.model}_generations.pkl', 'rb') as infile:
        sequences = pickle.load(infile)

    with open(f'{config.output_dir}/{run_name}/{args.model}_generations_similarities.pkl', 'rb') as infile:
        similarities_dict = pickle.load(infile)


    def get_neg_loglikelihoods(model, sequences):

        with torch.no_grad():
            result = []
            for sample in sequences:
                result_dict = {}
                prompt = sample['prompt']
                if 'cleaned_generations' in sample:
                    generations = sample['cleaned_generations'].to(device)
                else:
                    generations = sample['generations'].to(device)
                id_ = sample['id']

                average_neg_log_likelihoods = torch.zeros((generations.shape[0],))
                average_unconditioned_neg_log_likelihoods = torch.zeros((generations.shape[0],))
                neg_log_likelihoods = torch.zeros((generations.shape[0],))
                neg_unconditioned_log_likelihoods = torch.zeros((generations.shape[0],))
                pointwise_mutual_information = torch.zeros((generations.shape[0],))
                sequence_embeddings = []

                for generation_index in range(generations.shape[0]):
                    prompt = prompt[prompt != tokenizer.pad_token_id]
                    generation = generations[generation_index][generations[generation_index] != tokenizer.pad_token_id]

                    # This computation of the negative log likelihoods follows this tutorial: https://huggingface.co/docs/transformers/perplexity
                    target_ids = generation.clone()
                    target_ids[:len(prompt)] = -100
                    model_output = model(torch.reshape(generation, (1, -1)), labels=target_ids, output_hidden_states=True)
                    generation_only = generation.clone()[(len(prompt) - 1):]
                    unconditioned_model_output = model(torch.reshape(generation_only, (1, -1)),
                                                    labels=generation_only,
                                                    output_hidden_states=True)
                    hidden_states = model_output['hidden_states']
                    average_neg_log_likelihood = model_output['loss']

                    average_unconditioned_neg_log_likelihood = unconditioned_model_output['loss']
                    average_neg_log_likelihoods[generation_index] = average_neg_log_likelihood
                    average_unconditioned_neg_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood
                    neg_log_likelihoods[generation_index] = average_neg_log_likelihood * (len(generation) - len(prompt))
                    neg_unconditioned_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood * (
                        len(generation) - len(prompt))
                    pointwise_mutual_information[generation_index] = -neg_log_likelihoods[
                        generation_index] + neg_unconditioned_log_likelihoods[generation_index]

                    average_of_last_layer_token_embeddings = torch.mean(hidden_states[-1], dim=1)
                    sequence_embeddings.append(average_of_last_layer_token_embeddings)

                most_likely_generation = sample['most_likely_generation_ids'].to(device)
                target_ids = most_likely_generation.clone()
                target_ids[:len(prompt)] = -100
                model_output = model(torch.reshape(most_likely_generation, (1, -1)),
                                    labels=target_ids,
                                    output_hidden_states=True)
                hidden_states = model_output['hidden_states']
                average_neg_log_likelihood_of_most_likely_gen = model_output['loss']
                most_likely_generation_embedding = torch.mean(hidden_states[-1], dim=1)

                second_most_likely_generation = sample['second_most_likely_generation_ids'].to(device)
                target_ids = second_most_likely_generation.clone()
                target_ids[:len(prompt)] = -100
                model_output = model(torch.reshape(second_most_likely_generation, (1, -1)),
                                    labels=target_ids,
                                    output_hidden_states=True)
                hidden_states = model_output['hidden_states']
                average_neg_log_likelihood_of_second_most_likely_gen = model_output['loss']
                second_most_likely_generation_embedding = torch.mean(hidden_states[-1], dim=1)

                neg_log_likelihood_of_most_likely_gen = average_neg_log_likelihood_of_most_likely_gen * (
                    len(most_likely_generation) - len(prompt))

                sequence_embeddings = torch.stack(sequence_embeddings)
                result_dict['prompt'] = prompt
                result_dict['generations'] = generations
                result_dict['average_neg_log_likelihoods'] = average_neg_log_likelihoods
                result_dict['neg_log_likelihoods'] = neg_log_likelihoods
                result_dict['sequence_embeddings'] = most_likely_generation_embedding
                result_dict['most_likely_sequence_embedding'] = most_likely_generation
                result_dict['average_unconditioned_neg_log_likelihoods'] = average_unconditioned_neg_log_likelihoods
                result_dict['neg_unconditioned_log_likelihoods'] = neg_unconditioned_log_likelihoods
                result_dict['pointwise_mutual_information'] = pointwise_mutual_information
                result_dict['average_neg_log_likelihood_of_most_likely_gen'] = average_neg_log_likelihood_of_most_likely_gen
                result_dict[
                    'average_neg_log_likelihood_of_second_most_likely_gen'] = average_neg_log_likelihood_of_second_most_likely_gen
                result_dict['neg_log_likelihood_of_most_likely_gen'] = neg_log_likelihood_of_most_likely_gen
                result_dict['semantic_set_ids'] = torch.tensor(similarities_dict[id_[0]]['semantic_set_ids'], device=device)
                result_dict['id'] = id_
                result.append(result_dict)

            return result


    likelihoods = get_neg_loglikelihoods(model, sequences)

    with open(f'{config.output_dir}/{run_name}/{args.model}_generations_{args.model}_likelihoods.pkl',
            'wb') as outfile:
        pickle.dump(likelihoods, outfile)

### confidence measure

if os.path.exists(f'{config.output_dir}/{run_name}/aggregated_likelihoods_{args.model}_generations.pkl'):
    print("INFO:-----loading confidence measure------")
    with open(f'{config.output_dir}/{run_name}/aggregated_likelihoods_{args.model}_generations.pkl', 'rb') as infile:
        overall_results = pickle.load(infile)
else:

    print("INFO:-----calculating confidence measure------")

    llh_shift = torch.tensor(5.0)


    def get_overall_log_likelihoods(list_of_results):
        """Compute log likelihood of all generations under their given context.
        
        list_of_results: list of dictionaries with keys:
        
        returns: dictionary with keys: 'neg_log_likelihoods', 'average_neg_log_likelihoods'
                that contains tensors of shape (num_models, num_generations, num_samples_per_generation)
        """

        result_dict = {}

        list_of_keys = ['neg_log_likelihoods', 'average_neg_log_likelihoods', 'sequence_embeddings',\
                        'pointwise_mutual_information', 'average_neg_log_likelihood_of_most_likely_gen',\
                        'neg_log_likelihood_of_most_likely_gen', 'semantic_set_ids']

        for key in list_of_keys:
            list_of_ids = []
            overall_results = []
            for model_size, result in list_of_results:
                results_per_model = []
                for sample in result:
                    average_neg_log_likelihoods = sample[key]
                    list_of_ids.append(sample['id'][0])
                    results_per_model.append(average_neg_log_likelihoods)

                results_per_model = torch.stack(results_per_model)

                overall_results.append(results_per_model)

            if key != 'sequence_embeddings':
                overall_results = torch.stack(overall_results)

            result_dict[key] = overall_results

        result_dict['ids'] = list_of_ids
        return result_dict


    def get_mutual_information(log_likelihoods):
        """Compute confidence measure for a given set of likelihoods"""

        mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
        tiled_mean = mean_across_models.tile(log_likelihoods.shape[0], 1, 1)
        diff_term = torch.exp(log_likelihoods) * log_likelihoods - torch.exp(tiled_mean) * tiled_mean
        f_j = torch.div(torch.sum(diff_term, dim=0), diff_term.shape[0])
        mutual_information = torch.div(torch.sum(torch.div(f_j, mean_across_models), dim=1), f_j.shape[-1])

        return mutual_information


    def get_log_likelihood_variance(neg_log_likelihoods):
        """Compute log likelihood variance of approximate posterior predictive"""
        mean_across_models = torch.mean(neg_log_likelihoods, dim=0)
        variance_of_neg_log_likelihoods = torch.var(mean_across_models, dim=1)

        return variance_of_neg_log_likelihoods


    def get_log_likelihood_mean(neg_log_likelihoods):
        """Compute softmax variance of approximate posterior predictive"""
        mean_across_models = torch.mean(neg_log_likelihoods, dim=0)
        mean_of_neg_log_likelihoods = torch.mean(mean_across_models, dim=1)

        return mean_of_neg_log_likelihoods


    def get_mean_of_poinwise_mutual_information(pointwise_mutual_information):
        """Compute mean of pointwise mutual information"""
        mean_across_models = torch.mean(pointwise_mutual_information, dim=0)
        return torch.mean(mean_across_models, dim=1)


    def get_predictive_entropy(log_likelihoods):
        """Compute predictive entropy of approximate posterior predictive"""
        mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
        entropy = -torch.sum(mean_across_models, dim=1) / torch.tensor(mean_across_models.shape[1])
        return entropy


    def get_predictive_entropy_over_concepts(log_likelihoods, semantic_set_ids):
        """Compute the semantic entropy"""
        mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
        # This is ok because all the models have the same semantic set ids
        semantic_set_ids = semantic_set_ids.to('cpu')
        semantic_set_ids = semantic_set_ids[0]
        entropies = []
        for row_index in range(mean_across_models.shape[0]):
            aggregated_likelihoods = []
            row = mean_across_models[row_index]
            semantic_set_ids_row = semantic_set_ids[row_index]
            for semantic_set_id in torch.unique(semantic_set_ids_row):
                aggregated_likelihoods.append(torch.logsumexp(row[semantic_set_ids_row == semantic_set_id], dim=0))
            aggregated_likelihoods = torch.tensor(aggregated_likelihoods) - llh_shift
            entropy = - torch.sum(aggregated_likelihoods, dim=0) / torch.tensor(aggregated_likelihoods.shape[0])
            entropies.append(entropy)

        return torch.tensor(entropies)


    def get_margin_probability_uncertainty_measure(log_likelihoods):
        """Compute margin probability uncertainty measure"""
        mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
        topk_likelihoods, indices = torch.topk(mean_across_models, 2, dim=1, sorted=True)
        margin_probabilities = np.exp(topk_likelihoods[:, 0]) - np.exp(topk_likelihoods[:, 1])

        return margin_probabilities


    list_of_results = []

    with open(f'{config.output_dir}/{run_name}/{args.model}_generations_{args.model}_likelihoods.pkl',
            'rb') as infile:
        sequences = pickle.load(infile)
        list_of_results.append((args.model, sequences))

    overall_results = get_overall_log_likelihoods(list_of_results)
    mutual_information = get_mutual_information(-overall_results['neg_log_likelihoods'])
    predictive_entropy = get_predictive_entropy(-overall_results['neg_log_likelihoods'])
    predictive_entropy_over_concepts = get_predictive_entropy_over_concepts(-overall_results['average_neg_log_likelihoods'],
                                                                            overall_results['semantic_set_ids'])
    unnormalised_entropy_over_concepts = get_predictive_entropy_over_concepts(-overall_results['neg_log_likelihoods'],
                                                                            overall_results['semantic_set_ids'])

    margin_measures = get_margin_probability_uncertainty_measure(-overall_results['average_neg_log_likelihoods'])
    unnormalised_margin_measures = get_margin_probability_uncertainty_measure(-overall_results['neg_log_likelihoods'])


    def get_number_of_unique_elements_per_row(tensor):
        assert len(tensor.shape) == 2
        return torch.count_nonzero(torch.sum(torch.nn.functional.one_hot(tensor), dim=1), dim=1)


    number_of_semantic_sets = get_number_of_unique_elements_per_row(overall_results['semantic_set_ids'][0])
    average_predictive_entropy = get_predictive_entropy(-overall_results['average_neg_log_likelihoods'])
    average_predictive_entropy_on_subsets = []
    predictive_entropy_on_subsets = []
    semantic_predictive_entropy_on_subsets = []
    num_predictions = overall_results['average_neg_log_likelihoods'].shape[-1]
    number_of_semantic_sets_on_subsets = []
    for i in range(1, num_predictions + 1):
        offset = num_predictions * (i / 100)
        average_predictive_entropy_on_subsets.append(
            get_predictive_entropy(-overall_results['average_neg_log_likelihoods'][:, :, :int(i)]))
        predictive_entropy_on_subsets.append(get_predictive_entropy(-overall_results['neg_log_likelihoods'][:, :, :int(i)]))
        semantic_predictive_entropy_on_subsets.append(
            get_predictive_entropy_over_concepts(-overall_results['average_neg_log_likelihoods'][:, :, :int(i)],
                                                overall_results['semantic_set_ids'][:, :, :int(i)]))
        number_of_semantic_sets_on_subsets.append(
            get_number_of_unique_elements_per_row(overall_results['semantic_set_ids'][0][:, :i]))

    average_pointwise_mutual_information = get_mean_of_poinwise_mutual_information(
        overall_results['pointwise_mutual_information'])

    overall_results['mutual_information'] = mutual_information
    overall_results['predictive_entropy'] = predictive_entropy
    overall_results['predictive_entropy_over_concepts'] = predictive_entropy_over_concepts
    overall_results['unnormalised_entropy_over_concepts'] = unnormalised_entropy_over_concepts
    overall_results['number_of_semantic_sets'] = number_of_semantic_sets
    overall_results['margin_measures'] = margin_measures
    overall_results['unnormalised_margin_measures'] = unnormalised_margin_measures

    overall_results['average_predictive_entropy'] = average_predictive_entropy
    for i in range(len(average_predictive_entropy_on_subsets)):
        overall_results[f'average_predictive_entropy_on_subset_{i + 1}'] = average_predictive_entropy_on_subsets[i]
        overall_results[f'predictive_entropy_on_subset_{i + 1}'] = predictive_entropy_on_subsets[i]
        overall_results[f'semantic_predictive_entropy_on_subset_{i + 1}'] = semantic_predictive_entropy_on_subsets[i]
        overall_results[f'number_of_semantic_sets_on_subset_{i + 1}'] = number_of_semantic_sets_on_subsets[i]
    overall_results['average_pointwise_mutual_information'] = average_pointwise_mutual_information

    with open(f'{config.output_dir}/{run_name}/aggregated_likelihoods_{args.model}_generations.pkl',
            'wb') as outfile:
        pickle.dump(overall_results, outfile)

# Evaluation

print("INFO:-----Evaluation-----")

overall_result_dict = {}

aurocs_across_models = []

sequence_embeddings_dict = {}

run_ids_to_analyze = args.run_id
for run_id in run_ids_to_analyze:

    def get_similarities_df():
        """Get the similarities df from the pickle file"""
        with open(f'{config.output_dir}/{run_name}/{args.model}_generations_similarities.pkl', 'rb') as f:
            similarities = pickle.load(f)
            similarities_df = pd.DataFrame.from_dict(similarities, orient='index')
            similarities_df['id'] = similarities_df.index
            similarities_df['has_semantically_different_answers'] = similarities_df[
                'has_semantically_different_answers'].astype('int')
            similarities_df['rougeL_among_generations'] = similarities_df['syntactic_similarities'].apply(
                lambda x: x['rougeL'])

            return similarities_df

    def get_generations_df():
        """Get the generations df from the pickle file"""
        with open(f'{config.output_dir}/{run_name}/{args.model}_generations.pkl', 'rb') as infile:
            generations = pickle.load(infile)
            generations_df = pd.DataFrame(generations)
            generations_df['id'] = generations_df['id'].apply(lambda x: x[0])
            generations_df['id'] = generations_df['id'].astype('object')
            if not generations_df['semantic_variability_reference_answers'].isnull().values.any():
                generations_df['semantic_variability_reference_answers'] = generations_df[
                    'semantic_variability_reference_answers'].apply(lambda x: x[0].item())

            if not generations_df['rougeL_reference_answers'].isnull().values.any():
                generations_df['rougeL_reference_answers'] = generations_df['rougeL_reference_answers'].apply(
                    lambda x: x[0].item())
            generations_df['length_of_most_likely_generation'] = generations_df['most_likely_generation'].apply(
                lambda x: len(str(x).split(' ')))
            generations_df['length_of_answer'] = generations_df['answer'].apply(lambda x: len(str(x).split(' ')))
            generations_df['variance_of_length_of_generations'] = generations_df['generated_texts'].apply(
                lambda x: np.var([len(str(y).split(' ')) for y in x]))
            generations_df['correct'] = (generations_df['rougeL_to_target'] > 0.3).astype('int')

            return generations_df

    def get_likelihoods_df():
        """Get the likelihoods df from the pickle file"""

        with open(f'{config.output_dir}/{run_name}/aggregated_likelihoods_{args.model}_generations.pkl', 'rb') as f:
            likelihoods = pickle.load(f)
            print(likelihoods.keys())

            subset_keys = ['average_predictive_entropy_on_subset_' + str(i) for i in range(1, num_generations + 1)]
            subset_keys += ['predictive_entropy_on_subset_' + str(i) for i in range(1, num_generations + 1)]
            subset_keys += ['semantic_predictive_entropy_on_subset_' + str(i) for i in range(1, num_generations + 1)]
            subset_keys += ['number_of_semantic_sets_on_subset_' + str(i) for i in range(1, num_generations + 1)]

            keys_to_use = ('ids', 'predictive_entropy', 'mutual_information', 'average_predictive_entropy',\
                            'average_pointwise_mutual_information', 'average_neg_log_likelihood_of_most_likely_gen',\
                            'average_neg_log_likelihood_of_second_most_likely_gen', 'neg_log_likelihood_of_most_likely_gen',\
                            'predictive_entropy_over_concepts', 'number_of_semantic_sets', 'unnormalised_entropy_over_concepts')

            # filter keys_to_use if they are not in the likelihoods dict
            keys_to_use = tuple([key for key in keys_to_use if key in likelihoods.keys()])

            likelihoods_small = dict((k, likelihoods[k]) for k in keys_to_use + tuple(subset_keys))
            for key in likelihoods_small:
                if key == 'average_predictive_entropy_on_subsets':
                    likelihoods_small[key].shape
                if type(likelihoods_small[key]) is torch.Tensor:
                    likelihoods_small[key] = torch.squeeze(likelihoods_small[key].cpu())

            sequence_embeddings = likelihoods['sequence_embeddings']

            likelihoods_df = pd.DataFrame.from_dict(likelihoods_small)

            likelihoods_df.rename(columns={'ids': 'id'}, inplace=True)

            return likelihoods_df, sequence_embeddings

    similarities_df = get_similarities_df()
    generations_df = get_generations_df()
    num_generations = len(generations_df['generated_texts'][0])
    likelihoods_df, sequence_embeddings = get_likelihoods_df()
    result_df = generations_df.merge(similarities_df, on='id').merge(likelihoods_df, on='id')

    n_samples_before_filtering = len(result_df)
    result_df['len_most_likely_generation_length'] = result_df['most_likely_generation'].apply(lambda x: len(x.split()))

    # Begin analysis
    result_dict = {}
    result_dict['accuracy'] = result_df['correct'].mean()

    correct_nans = result_df['correct'][result_df['correct'].isna()].index
    avg_pred_entr_nans = result_df['average_predictive_entropy'][result_df['average_predictive_entropy'].isna()].index

    # combine the indices and drop them
    indices_to_drop = list(set(correct_nans).union(set(avg_pred_entr_nans)))
    result_df = result_df.drop(indices_to_drop)


    # Compute the auroc for the length normalized predictive entropy
    ln_predictive_entropy_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                                result_df['average_predictive_entropy'])
    result_dict['ln_predictive_entropy_auroc'] = ln_predictive_entropy_auroc

    predictive_entropy_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'], result_df['predictive_entropy'])
    result_dict['predictive_entropy_auroc'] = predictive_entropy_auroc

    entropy_over_concepts_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                                result_df['predictive_entropy_over_concepts'])
    result_dict['entropy_over_concepts_auroc'] = entropy_over_concepts_auroc

    if 'unnormalised_entropy_over_concepts' in result_df.columns:
        unnormalised_entropy_over_concepts_auroc = sklearn.metrics.roc_auc_score(
            1 - result_df['correct'], result_df['unnormalised_entropy_over_concepts'])
        result_dict['unnormalised_entropy_over_concepts_auroc'] = unnormalised_entropy_over_concepts_auroc

    aurocs_across_models.append(entropy_over_concepts_auroc)

    neg_llh_most_likely_gen_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                                  result_df['neg_log_likelihood_of_most_likely_gen'])
    result_dict['neg_llh_most_likely_gen_auroc'] = neg_llh_most_likely_gen_auroc

    number_of_semantic_sets_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                                  result_df['number_of_semantic_sets'])
    result_dict['number_of_semantic_sets_auroc'] = number_of_semantic_sets_auroc

    result_dict['number_of_semantic_sets_correct'] = result_df[result_df['correct'] ==
                                                               1]['number_of_semantic_sets'].mean()
    result_dict['number_of_semantic_sets_incorrect'] = result_df[result_df['correct'] ==
                                                                 0]['number_of_semantic_sets'].mean()

    result_dict['average_rougeL_among_generations'] = result_df['rougeL_among_generations'].mean()
    result_dict['average_rougeL_among_generations_correct'] = result_df[result_df['correct'] ==
                                                                        1]['rougeL_among_generations'].mean()
    result_dict['average_rougeL_among_generations_incorrect'] = result_df[result_df['correct'] ==
                                                                          0]['rougeL_among_generations'].mean()
    result_dict['average_rougeL_auroc'] = sklearn.metrics.roc_auc_score(result_df['correct'],
                                                                        result_df['rougeL_among_generations'])

    average_neg_llh_most_likely_gen_auroc = sklearn.metrics.roc_auc_score(
        1 - result_df['correct'], result_df['average_neg_log_likelihood_of_most_likely_gen'])
    result_dict['average_neg_llh_most_likely_gen_auroc'] = average_neg_llh_most_likely_gen_auroc
    result_dict['rougeL_based_accuracy'] = result_df['correct'].mean()

    '''result_dict['margin_measure_auroc'] = sklearn.metrics.roc_auc_score(
        1 - result_df['correct'], result_df['average_neg_log_likelihood_of_most_likely_gen'] +
        result_df['average_neg_log_likelihood_of_second_most_likely_gen'])'''


    # Measure the AURROCs when using different numbers of generations to compute our uncertainty measures.
    ln_aurocs = []
    aurocs = []
    semantic_aurocs = []
    average_number_of_semantic_sets = []
    average_number_of_semantic_sets_correct = []
    average_number_of_semantic_sets_incorrect = []
    for i in range(1, num_generations + 1):
        ln_predictive_entropy_auroc = sklearn.metrics.roc_auc_score(
            1 - result_df['correct'], result_df['average_predictive_entropy_on_subset_{}'.format(i)])
        aurocs.append(
            sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                          result_df['predictive_entropy_on_subset_{}'.format(i)]))
        ln_aurocs.append(ln_predictive_entropy_auroc)
        semantic_aurocs.append(
            sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                          result_df['semantic_predictive_entropy_on_subset_{}'.format(i)]))
        average_number_of_semantic_sets.append(result_df['number_of_semantic_sets_on_subset_{}'.format(i)].mean())
        average_number_of_semantic_sets_correct.append(
            result_df[result_df['correct'] == 1]['number_of_semantic_sets_on_subset_{}'.format(i)].mean())
        average_number_of_semantic_sets_incorrect.append(
            result_df[result_df['correct'] == 0]['number_of_semantic_sets_on_subset_{}'.format(i)].mean())

    result_dict['ln_predictive_entropy_auroc_on_subsets'] = ln_aurocs
    result_dict['predictive_entropy_auroc_on_subsets'] = aurocs
    result_dict['semantic_predictive_entropy_auroc_on_subsets'] = semantic_aurocs
    result_dict['average_number_of_semantic_sets_on_subsets'] = average_number_of_semantic_sets
    result_dict['average_number_of_semantic_sets_on_subsets_correct'] = average_number_of_semantic_sets_correct
    result_dict['average_number_of_semantic_sets_on_subsets_incorrect'] = average_number_of_semantic_sets_incorrect
    result_dict['model_name'] = args.model
    result_dict['run_name'] = run_name


    wandb.log({"result_dict": result_dict})

    overall_result_dict[run_id] = result_dict
    sequence_embeddings_dict[run_id] = sequence_embeddings

    wandb.finish()
    torch.cuda.empty_cache()

with open('overall_results.json', 'w') as f:
    json.dump(overall_result_dict, f)

with open('sequence_embeddings.pkl', 'wb') as f:
    pickle.dump(sequence_embeddings_dict, f)

# Store data frame as csv
accuracy_verification_df = result_df[['most_likely_generation', 'answer', 'correct']]
accuracy_verification_df.to_csv('accuracy_verification.csv')

print("INFO:-----DONE-----")
