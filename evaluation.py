from AlignScore.src.alignscore import AlignScore
from BARTScore.bart_score import BARTScorer
from transformers import BertForSequenceClassification, BertTokenizer
import argparse
import json
import numpy as np
import evaluate
import torch
import logging

logging.disable(logging.WARNING)

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='input.json')
    parser.add_argument('--output_file', type=str, default='output.json')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--alignscore_ckpt', type=str, default="./AlignScore/AlignScore-large.ckpt")
    parser.add_argument('--gpu', type=int, default=0)

    return parser


def main(args):
    input_file = args.input_file
    output_file = args.output_file

    file_name = {}
    file_name['file_name'] = input_file
    json_data = []
    with open(input_file, 'r') as file:
        for line in file:
            json_data.append(json.loads(line))

    predictions = []
    references = []
    sources = []
    i = 0
    for data in json_data:
        if not data["predicted"].strip(): continue
        predictions.append(data["predicted"].strip())
        references.append(data["gold"])
        sources.append(data["source"])
        i+=1

    total_result = {}

    # 1) AlignScore
    align_scorer = AlignScore(model='roberta-base', batch_size=args.batch_size, device='cuda:'+str(args.gpu), ckpt_path=args.alignscore_ckpt, evaluation_mode='nli_sp')
    alignscore_result = align_scorer.score(contexts=sources, claims=predictions)
    total_result['AlignScore'] = 100*np.mean(alignscore_result)
    print(total_result['AlignScore'])

    # 2) FactCC
    model_path = 'manueldeprada/FactCC'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device(f'cuda:'+str(args.gpu))
    model.to(device)

    pred_result = []
    for i in range(len(predictions)):
        text = sources[i]
        claim = predictions[i]

        input_dict = tokenizer(text, claim, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        # input_dict = tokenizer(text, claim, max_length=512, padding='max_length', truncation='only_first', return_tensors='pt')
        input_dict = input_dict.to(device)
        logits = model(**input_dict).logits
        pred = logits.argmax(dim=1)
        pred_result.append(pred.item())

    factcc_result = 1 - np.mean(pred_result)
    total_result['FactCC'] = 100*factcc_result
    print(total_result['FactCC'])

    # 3) BARTScore
    bart_scorer = BARTScorer(device='cuda:'+str(args.gpu), checkpoint='facebook/bart-large-cnn')
    bart_result = np.mean(bart_scorer.score(srcs=sources, tgts=predictions, batch_size=args.batch_size))
    total_result['BARTScore'] = bart_result
    print(total_result['BARTScore'])

    # 4) BS-Fact
    bert_scorer = evaluate.load("bertscore")
    bsfact_result = bert_scorer.compute(predictions=predictions, references=sources, lang="en",batch_size=args.batch_size)
    total_result['BS-Fact'] = 100*np.mean(bsfact_result["precision"])
    print(total_result['BS-Fact'])

    # 5) Rouge-L
    rouge_scorer = evaluate.load('rouge')
    rouge_result = rouge_scorer.compute(predictions=predictions, references=references)
    total_result['Rouge-L'] = 100*rouge_result['rougeL']
    print(total_result['Rouge-L'])

    print(total_result)

    with open(output_file, "a") as json_file:
        json_file.write(json.dumps(file_name))
        json_file.write("\n")
        json_file.write(json.dumps(total_result))
        json_file.write("\n")
        json_file.write("\n")


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    main(args)
